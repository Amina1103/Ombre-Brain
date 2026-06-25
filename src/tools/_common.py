"""
========================================
tools/_common.py — 跨工具共享的辅助逻辑
========================================

这个文件收纳被多个工具同时复用的、与具体工具语义无关的小工具：
配额检查（单桶字节上限 / pinned 数量上限）、合并或新建（hold/grow 共用）、
新桶疑似重复扫描、新事件触发的 plan 自动闭环判定。

关键行为：
- check_content_size / check_pinned_quota：读取 config.limits，超限返回中文提示串
- merge_or_create：先用语义检索找近似桶；超过阈值则合并（hold 用原文拼接，
  grow 用 LLM 压缩），否则新建；写完同步刷新 embedding 与脱水缓存
- iter 2.0：merge_or_create 接受 ``source_tool`` / ``grow_batch_id``，
  新建时写入 frontmatter；合并时不动原桶 source_tool，只追加 ``last_merged_by``
- check_duplicate_for：fire-and-forget 标记疑似重复对（不自动合并）
- check_plan_resolution：fire-and-forget 用向量预筛 + LLM 保守判断
  来把已完成的 active plan 标为 resolved

不做什么（边界）：
- 不持有任何全局对象，所有依赖都从 _runtime 取
- 不做日志格式化以外的副作用包装；调用方自行决定是否 await

对外暴露：limits_cfg / max_bucket_bytes / max_pinned / check_content_size /
         count_pinned / check_pinned_quota / merge_or_create /
         check_duplicate_for / check_plan_resolution
========================================
"""

from typing import Tuple
import asyncio
import hashlib

from . import _runtime as rt

_EMBED_WARN = (
    "向量化失败，该桶不参与语义检索，仅支持关键词匹配。请检查 OMBRE_EMBED_API_KEY。"
)

# ============================================================
# 常量 / Named constants
# ------------------------------------------------------------
# rule.md §①：禁止裸魔法数字。下面这些原本散在 helper 默认参数与
# 业务逻辑中，集中后：①调参一眼看完；②哲学阈值（importance≥9 上限）
# 明确可追。改这些值前请读 rule.md §1.0：“importance 稀缺才有意义”。
# ============================================================

# --- 桶与配额默认值 ---
_DEFAULT_MAX_BUCKET_BYTES = 50 * 1024  # 50 KB 单桶上限（超过建议走 grow 拆存）
_DEFAULT_MAX_PINNED = 20               # pinned 桶上限（哲学边界：重要必须稀缺）；与 config.example.yaml limits.max_pinned 同步

# --- importance≥9 配额（rule.md §1.0 哲学） ---
# ⚠ Amina 定制(上游 rebase 须带, 见迁移备忘第11项): 上游原值 HARD_CAP=24 / SOFT_WARN=22。
#   她库存有 ~125 条历史 importance=9(老版本无此配额时 Cyrus 自己存的真心标记), 24 太低会把
#   新的 9 全部静默降级。硬上限放宽到 175(留余量), 软阈值设 130: 库存 ~125, 让 Cyrus 一过 130
#   就开始收到"接近上限"提醒, 在 130~175 这段"斟酌区"里掂量新 9 值不值, 真降级才到 175。
_HIGH_IMP_THRESHOLD = 9                # importance 达到该值算“高重要度”
_HIGH_IMP_HARD_CAP = 175               # 高重要度桶硬上限 (上游 24, Amina 放宽)
_HIGH_IMP_SOFT_WARN = 130              # 过此数推 OB-W003"接近上限"提醒 = 斟酌区起点 (上游 22)
_HIGH_IMP_DEGRADE_TO = 8               # 超限时自动降到的 importance

# --- pinned 软阈值 ---
_PINNED_SOFT_GAP = 2                   # “软阈值 = cap - GAP”；cap=20 → soft=18

# --- check_duplicate_for / check_plan_resolution ---
_DUP_DEFAULT_THRESHOLD = 0.95          # 向量相似 >= 该值 → 标为疑似重复
_DUP_TOPK = 10                         # 检索前 N 个候选以判重复
_PLAN_VECTOR_TOPK = 20                 # plan 判定的向量预筛范围
_PLAN_VECTOR_THRESHOLD = 0.7           # 超过才交给 LLM 判定是否已完成
_PLAN_LLM_CONFIDENCE_MIN = 0.7         # LLM judgement.confidence 下限
_PLAN_FALLBACK_CAP = 10                # 无向量时直接送 LLM 的 plan 上限（防止过多 LLM 调用）

# --- 字段截断长度（下游存储 / 日志可读性）---
_RESOLUTION_REASON_MAX = 200           # 写入桶 frontmatter 的理由上限
_LOG_REASON_PREVIEW = 60               # 日志里预览的理由长度

# --- content lock 哈希 key 长度 ---
_CONTENT_LOCK_KEY_HEX = 16             # 64 bit 空间，碰撞概率徽不足道

# F-01 / F-08 fix: per-content-hash asyncio.Lock，防止并发同内容双新建。
# asyncio 单线程模型下 dict 访问本身是原子的，无需额外互斥。
# Lock 对象懒创建后不回收（单进程内独立 content hash 总量有限，不构成泄漏）。
_merge_content_locks: dict[str, asyncio.Lock] = {}


def _get_content_lock(content: str) -> asyncio.Lock:
    """Return (lazily created) per-content-hash asyncio.Lock."""
    key = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:_CONTENT_LOCK_KEY_HEX]
    if key not in _merge_content_locks:
        _merge_content_locks[key] = asyncio.Lock()
    return _merge_content_locks[key]


def _push_warning_safe(code: str, msg: str) -> None:
    """安全调用 errors.push_warning；import 失败时静默降级。

    原因：push_warning 在两个 quota helper 里被调 4 次，每次都要重复
    “三层 try/except import”的定位代码。集中后：
      ① 业务代码变成干净的一行调用；
      ② import 后退逻辑只需调一处；
      ③ 测试打档只需 patch 本函数。

    路径优先级（跟 imports.md 一致）：
      1. from errors        —— src/ 在 sys.path 顶层的生产/测试环境
      2. from ..errors      —— 包内相对导入的兑底
      3. 均失败 → 静默跳过（不能因 warning 传递失败让业务报错）
    """
    try:
        from errors import push_warning  # type: ignore
    except ImportError:
        try:
            from ..errors import push_warning  # type: ignore
        except Exception:  # pragma: no cover
            return
    try:
        push_warning(code, msg)
    except Exception:  # pragma: no cover
        # 警告通道崩了也不能拖垃业务路径
        pass


def limits_cfg() -> dict:
    """读 config.limits 段；缺省值与 1.6 spec §5 一致：50KB 单桶 / 20 pinned。"""
    return rt.config.get("limits", {}) or {}


def max_bucket_bytes() -> int:
    return int(limits_cfg().get("max_bucket_bytes") or _DEFAULT_MAX_BUCKET_BYTES)


def max_pinned() -> int:
    return int(limits_cfg().get("max_pinned") or _DEFAULT_MAX_PINNED)


def check_content_size(content: str) -> str | None:
    """超过单桶上限返回中文提示串；否则返回 None。"""
    cap = max_bucket_bytes()
    if cap <= 0:
        return None
    size = len(content.encode("utf-8"))
    if size > cap:
        return (
            f"内容过大（{size/1024:.1f} KB > 上限 {cap/1024:.0f} KB）。"
            "请改用 grow 拆分存入，或在 config.limits.max_bucket_bytes 调高上限。"
        )
    return None


async def count_pinned() -> int:
    """统计当前 pinned 桶数量。失败时返回 0（保守，不阻断）。"""
    try:
        all_b = await rt.bucket_mgr.list_all(include_archive=False)
        return sum(
            1 for b in all_b
            if b.get("metadata", {}).get("pinned")
            or b.get("metadata", {}).get("type") == "permanent"
        )
    except Exception as e:
        rt.logger.warning(f"count_pinned failed: {e}")
        return 0


async def check_pinned_quota() -> str | None:
    """到达 pinned 上限返回提示串；否则返回 None。

    （store_pinned 在严格模式下用此函数硬拒绝；新的"自动降级"路径请改用
    enforce_pinned_quota，达到上限时返回 (False, msg) 让调用方走普通桶。）"""
    cap = max_pinned()
    if cap <= 0:
        return None
    cur = await count_pinned()
    if cur >= cap:
        return (
            f"pinned 桶已达上限（{cur}/{cap}），建议先用 trace(bucket_id, pinned=0) "
            "清理低优先级钉选；或在 config.limits.max_pinned 调高上限。"
        )
    return None


# ============================================================
# 配额 helpers（统一错误体系 OB-W003/W004 + OB-I001/I002）
# ------------------------------------------------------------
# 设计：把"配额预警"和"自动降级"两步分开，分别对应 W 与 I。
# 业务代码调用前者拿到提示后，自动经 _push_warning_safe 送去 MCP 返回末尾。
# 阈值常量定义在文件顶部"常量"区，与 importance 哲学边界放在一起。
# ============================================================


async def count_high_importance() -> int:
    """统计 importance≥9 的非 pinned/protected 桶。失败时返回 0（不阻断写入）。"""
    try:
        all_b = await rt.bucket_mgr.list_all(include_archive=False)
        return sum(
            1 for b in all_b
            if int(b.get("metadata", {}).get("importance") or 0) >= _HIGH_IMP_THRESHOLD
            and not b.get("metadata", {}).get("pinned")
            and not b.get("metadata", {}).get("protected")
        )
    except Exception as e:
        rt.logger.warning(f"count_high_importance failed: {e}")
        return 0


async def enforce_high_importance_quota(importance: int) -> int:
    """importance≥9 配额检查 + 自动降级。

    - 当前数 ≥ 硬上限 → push OB-I001 并把 importance 降为 _HIGH_IMP_DEGRADE_TO
    - 当前数 ≥ 软阈值 → push OB-W003（仅提醒，不动数据）
    返回最终生效的 importance。
    """
    if importance < _HIGH_IMP_THRESHOLD:
        return importance
    cur = await count_high_importance()
    if cur >= _HIGH_IMP_HARD_CAP:
        rt.logger.info(
            f"op=quota phase=branch branch=imp_degrade requested={importance} "
            f"current={cur} cap={_HIGH_IMP_HARD_CAP} degraded_to={_HIGH_IMP_DEGRADE_TO}"
        )
        _push_warning_safe(
            "OB-I001",
            f"当前已有 {cur} 条 importance≥{_HIGH_IMP_THRESHOLD}（硬上限 {_HIGH_IMP_HARD_CAP}），新桶 importance 自动降级为 {_HIGH_IMP_DEGRADE_TO}",
        )
        return _HIGH_IMP_DEGRADE_TO
    if cur >= _HIGH_IMP_SOFT_WARN:
        _push_warning_safe(
            "OB-W003",
            f"当前已有 {cur} 条 importance≥{_HIGH_IMP_THRESHOLD}（硬上限 {_HIGH_IMP_HARD_CAP}），接近上限",
        )
    return importance


async def enforce_pinned_quota(pinned: bool) -> bool:
    """pinned 配额检查 + 自动退出。

    - 当前数 ≥ 硬上限 → push OB-I002 并返回 False（走普通桶）
    - 当前数 ≥ 软阈值 → push OB-W004（仅提醒，不动数据）
    传入 pinned=False 时直接返回 False。
    """
    if not pinned:
        return False
    cap = max_pinned()
    cur = await count_pinned()
    # 软阈值 = cap - GAP；cap=20、GAP=2 → soft=18。cap 太小（≤GAP）退化为硬上限。
    soft = max(1, cap - _PINNED_SOFT_GAP) if cap > _PINNED_SOFT_GAP else cap
    if cap > 0 and cur >= cap:
        rt.logger.info(
            f"op=quota phase=branch branch=pinned_degrade current={cur} cap={cap}"
        )
        _push_warning_safe(
            "OB-I002",
            f"当前已有 {cur} 条 pinned（硬上限 {cap}），本次未钉成功，已保留为普通桶",
        )
        return False
    if cap > 0 and cur >= soft:
        _push_warning_safe(
            "OB-W004",
            f"当前已有 {cur} 条 pinned（硬上限 {cap}），接近上限",
        )
    return True


def append_plan_change_log(old_history, action: str, **fields) -> list:
    """plan 桶 change_log 唯一写入口（iter 2.0 §10 U-01 修复）。

    把旧 history 复制一份、追加一条带 ISO 时间戳的新条目，返回新 list。
    所有写 plan change_log 的地方（plan_create / trace plan / dashboard plan action）
    都必须走这里，保证字段顺序、时间戳精度、复制语义一致。
    """
    from datetime import datetime as _dt
    history = list(old_history or [])
    entry = {"ts": _dt.now().isoformat(timespec="seconds"), "action": action}
    for k, v in fields.items():
        if v is not None:
            entry[k] = v
    history.append(entry)
    return history


async def merge_or_create(
    content: str,
    tags: list,
    importance: int,
    domain: list,
    valence: float,
    arousal: float,
    name: str = "",
    raw_merge: bool = False,
    why_remembered: str = "",
    source_tool: str = "",
    grow_batch_id: str = "",
) -> Tuple[str, bool, str]:
    """
    检查是否有相似桶可合并，有则合并，无则新建。返回 (桶ID或名称, 是否合并, embed警告信息)。

    raw_merge=True (hold)：原文追加，不调 LLM 压缩。
    raw_merge=False (grow)：LLM 压缩老+新内容。

    iter 2.0 来源追踪：
    - source_tool: "hold" | "grow"，作为新建桶的 source_tool 写入；
      合并路径下保留原桶 source_tool 不变，但写 last_merged_by=source_tool。
    - grow_batch_id: 仅 grow 路径会传，新建时写入；合并路径不覆盖原桶的 batch_id
      （原桶可能来自上一次 grow 或 hold，硬覆盖会丢失最初批次信息）。

    F-01 / F-08 fix：整个 search→create 路径在 per-content-hash Lock 下串行执行。
    同内容并发调用时后到的协程会阻塞，等前者写完后直接走合并分支，不产生重复桶。
    """
    async with _get_content_lock(content):
        return await _merge_or_create_inner(
            content=content, tags=tags, importance=importance, domain=domain,
            valence=valence, arousal=arousal, name=name, raw_merge=raw_merge,
            why_remembered=why_remembered, source_tool=source_tool,
            grow_batch_id=grow_batch_id,
        )


async def _merge_or_create_inner(
    content: str,
    tags: list,
    importance: int,
    domain: list,
    valence: float,
    arousal: float,
    name: str = "",
    raw_merge: bool = False,
    why_remembered: str = "",
    source_tool: str = "",
    grow_batch_id: str = "",
) -> Tuple[str, bool, str]:
    """实际的 search→merge/create 逻辑，由 merge_or_create 在 Lock 保护下调用。"""
    try:
        existing = await rt.bucket_mgr.search(content, limit=1, domain_filter=domain or None)
    except Exception as e:
        rt.logger.warning(f"Search for merge failed, creating new / 合并搜索失败，新建: {e}")
        existing = []

    if existing and existing[0].get("score", 0) > (rt.config.get("merge_threshold") or 75):
        bucket = existing[0]
        # --- 不合并到钉选/保护桶 ---
        if not (bucket["metadata"].get("pinned") or bucket["metadata"].get("protected")):
            try:
                if raw_merge:
                    # --- 原文拼接合并（hold 路径）---
                    old_text = bucket["content"].rstrip()
                    new_text = content.strip()
                    if new_text and new_text not in old_text:
                        merged = f"{old_text}\n\n---\n{new_text}" if old_text else new_text
                    else:
                        merged = old_text or new_text
                else:
                    # --- LLM 压缩合并（grow 路径）---
                    merged = await rt.dehydrator.merge(bucket["content"], content)
                old_v = bucket["metadata"].get("valence") or 0.5
                old_a = bucket["metadata"].get("arousal") or 0.3
                merged_valence = round((old_v + valence) / 2, 2) if 0 <= valence <= 1 else old_v
                merged_arousal = round((old_a + arousal) / 2, 2) if 0 <= arousal <= 1 else old_a
                update_kwargs = dict(
                    content=merged,
                    tags=list(set((bucket["metadata"].get("tags") or []) + tags)),
                    importance=max(bucket["metadata"].get("importance") or 5, importance),
                    domain=list(set((bucket["metadata"].get("domain") or []) + domain)),
                    valence=merged_valence,
                    arousal=merged_arousal,
                )
                # iter 2.0：合并时记录「最后一次合并由谁触发」，不动原桶的 source_tool。
                # 这样 dashboard 既能看到桶最初由谁创建，也能看到最近一次合并的来源。
                if source_tool:
                    update_kwargs["last_merged_by"] = source_tool
                await rt.bucket_mgr.update(bucket["id"], **update_kwargs)
                # --- 旧 content 的脱水缓存失效，让 breath 拿到合并后的新文本 ---
                try:
                    rt.dehydrator.invalidate_cache(bucket["content"])
                except Exception:
                    pass
                # --- 合并后刷新 embedding（best-effort，合并路径不返回 embed 警告）---
                try:
                    await rt.embedding_engine.generate_and_store(bucket["id"], merged)
                except Exception:
                    pass
                rt.logger.info(
                    f"op=merge_or_create phase=branch branch=merge bucket_id={bucket['id']} "
                    f"raw_merge={int(raw_merge)} source_tool={source_tool or '_'} "
                    f"score={existing[0].get('score', 0):.3f}"
                )
                return bucket["id"], True, ""
            except Exception as e:
                rt.logger.warning(f"Merge failed, creating new / 合并失败，新建: {e}")

    bucket_id = await rt.bucket_mgr.create(
        content=content,
        tags=tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=name or None,
        why_remembered=why_remembered,
        source_tool=source_tool,
        grow_batch_id=grow_batch_id,
    )
    # --- 新桶生成 embedding；失败时返回警告（降级，不报错）---
    # 允许降级：embedding 失败桶仍有效，仅丧失语义检索能力，设计上明确接受此降级
    embed_warn = ""
    try:
        ok = await rt.embedding_engine.generate_and_store(bucket_id, content)
        if not ok:
            embed_warn = _EMBED_WARN
            rt.logger.info(
                f"op=merge_or_create phase=branch branch=embed_degrade bucket_id={bucket_id} "
                f"reason=generate_returned_false"
            )
    except Exception as _embed_exc:
        embed_warn = _EMBED_WARN
        rt.logger.info(
            f"op=merge_or_create phase=branch branch=embed_degrade bucket_id={bucket_id} "
            f"reason={type(_embed_exc).__name__}"
        )
    rt.logger.info(
        f"op=merge_or_create phase=branch branch=create bucket_id={bucket_id} "
        f"source_tool={source_tool or '_'} grow_batch_id={grow_batch_id or '_'} "
        f"embed_ok={int(not embed_warn)}"
    )
    return bucket_id, False, embed_warn


async def check_duplicate_for(new_bucket_id: str, new_text: str, threshold: float = _DUP_DEFAULT_THRESHOLD) -> None:
    """fire-and-forget：新桶写完后，向量相似 > threshold 的旧桶标为疑似重复。

    iter 1.6 §4：不自动合并，只在两边各写 dup_candidate=<对端 id> + dup_score=<0~1>，
    Dashboard 在桶详情里显示「疑似重复」提示，由她/他手动确认是否合并。
    """
    try:
        if not rt.embedding_engine or not getattr(rt.embedding_engine, "enabled", False):
            return
        sims = await rt.embedding_engine.search_similar(new_text, top_k=_DUP_TOPK)
        for bid, score in sims:
            if bid == new_bucket_id:
                continue
            if score < threshold:
                continue
            try:
                await rt.bucket_mgr.update(
                    new_bucket_id, dup_candidate=bid, dup_score=round(float(score), 4)
                )
                await rt.bucket_mgr.update(
                    bid, dup_candidate=new_bucket_id, dup_score=round(float(score), 4)
                )
                rt.logger.info(
                    f"duplicate candidate: {new_bucket_id} ↔ {bid} (sim={score:.3f})"
                )
            except Exception as e:
                rt.logger.warning(f"dup mark failed: {e}")
            break  # 只标最相似的一对
    except Exception as e:
        rt.logger.warning(f"check_duplicate_for outer error: {e}")


async def check_plan_resolution(new_event_text: str, source_bucket_id: str = "") -> None:
    """fire-and-forget：扫描 active plan，向量相似 > 0.7 的让 LLM 保守判断是否完成。"""
    try:
        all_b = await rt.bucket_mgr.list_all(include_archive=False)
        active_plans = [
            b for b in all_b
            if b["metadata"].get("type") == "plan"
            and b["metadata"].get("status", "active") == "active"
        ]
        if not active_plans:
            return
        plan_candidates = []
        if rt.embedding_engine and getattr(rt.embedding_engine, "enabled", False):
            try:
                sims = await rt.embedding_engine.search_similar(new_event_text, top_k=_PLAN_VECTOR_TOPK)
                sim_map = {bid: sc for bid, sc in sims}
                for p in active_plans:
                    if sim_map.get(p["id"], 0.0) > _PLAN_VECTOR_THRESHOLD:
                        plan_candidates.append(p)
                # 向量预筛没命中任何 plan → fallback 到全量（上限保护）
                if not plan_candidates:
                    plan_candidates = active_plans[:_PLAN_FALLBACK_CAP]
            except Exception as e:
                rt.logger.warning(f"plan resolution: vector pre-filter failed, falling back: {e}")
                plan_candidates = active_plans[:_PLAN_FALLBACK_CAP]
        else:
            # 无向量后端：直接把所有 active plan 送 LLM 判定（上限防止过多调用）
            plan_candidates = active_plans[:_PLAN_FALLBACK_CAP]
        for p in plan_candidates:
            try:
                judgement = await rt.dehydrator.judge_plan_resolution(
                    p["content"], new_event_text
                )
                if judgement.get("resolved") and judgement.get("confidence", 0.0) >= _PLAN_LLM_CONFIDENCE_MIN:
                    await rt.bucket_mgr.update(
                        p["id"],
                        status="resolved",
                        resolution_reason=judgement.get("reason", "")[:_RESOLUTION_REASON_MAX],
                        resolved_by=source_bucket_id or "",
                    )
                    rt.logger.info(
                        f"plan auto-resolved: {p['id']} — {judgement.get('reason', '')[:_LOG_REASON_PREVIEW]}"
                    )
            except Exception as e:
                rt.logger.warning(f"plan resolution judgement failed for {p['id']}: {e}")
    except Exception as e:
        rt.logger.warning(f"check_plan_resolution outer error: {e}")


# ============================================================
# 显式 plan→bucket 联动（人工/Claude 路径）
# ------------------------------------------------------------
# 当 plan 桶被「人工或 Claude 显式」标为 resolved 时，把它指向的
# related_bucket / resolved_by 两个普通桶也同步标 resolved=True。
# 这是 rule.md §1 哲学落地：plan 是承诺，承诺被放下，承载这条承诺
# 的事件桶也不该再浮上来。
#
# 不联动的路径：check_plan_resolution（LLM 自动二判）—— 自动判定
# 的可信度低于人工/Claude 显式动作，避免把活的事件桶意外打沉。
#
# 反向不做：bucket trace(resolved=1) 不联动 plan（plan 是独立承诺，
# 单条事件结束不等于承诺达成）。
# ============================================================

async def cascade_plan_resolved_to_buckets(plan_meta: dict, plan_id: str) -> list[str]:
    """把 plan_meta 里 related_bucket / resolved_by 指向的普通桶标 resolved。

    入参：plan 桶的 metadata + plan_id（仅用于日志）。
    出参：实际被联动到的 bucket_id 列表（已存在、未删除、未本来就 resolved）。
    异常：单个桶失败不影响其他；外层异常仅记日志、返回已联动列表。
    """
    linked: list[str] = []
    if not isinstance(plan_meta, dict):
        return linked
    candidates: list[str] = []
    for key in ("related_bucket", "resolved_by"):
        val = (plan_meta.get(key) or "").strip() if isinstance(plan_meta.get(key), str) else ""
        # resolved_by 可能是 "manual" / "llm_judge"，不是 bucket_id，跳过
        if not val or val in ("manual", "llm_judge"):
            continue
        if val not in candidates:
            candidates.append(val)
    for bid in candidates:
        try:
            b = await rt.bucket_mgr.get(bid)
            if not b:
                continue
            meta = b.get("metadata", {})
            # 已经 resolved 就不重复操作（避免无意义 touch）
            if meta.get("resolved"):
                continue
            # plan 不联动 plan；letter 也跳过（永久保留）
            if meta.get("type") in ("plan", "letter"):
                continue
            ok = await rt.bucket_mgr.update(bid, resolved=True)
            if ok:
                linked.append(bid)
                rt.logger.info(
                    f"plan→bucket cascade: plan={plan_id} → bucket={bid} resolved=True"
                )
        except Exception as e:
            rt.logger.warning(
                f"plan→bucket cascade failed: plan={plan_id} bucket={bid} err={e}"
            )
    return linked


# 向后兼容：保留下划线别名（部分历史调用点用 _ 前缀）
_check_duplicate_for = check_duplicate_for
_check_plan_resolution = check_plan_resolution
