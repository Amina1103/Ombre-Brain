"""
========================================
tools/breath/search.py — 有 query 的检索模式
========================================

走 breath(query=...) 时进入这里。两路并行：bucket_manager 关键词检索 +
embedding_engine 向量近邻，结果合并去重，逐条 dehydrate 后塞 token 预算。

关键行为：
- domain/valence/arousal 作为过滤参数传给 bucket_mgr.search
- 向量通道阈值 sim>0.5；archived 桶不能从向量通道漂回（违反契约）
- 命中后调 touch()，记忆重构会把展示层 valence 按当前情绪做 ±0.1 微调
- 检索结果 < 3 时 40% 概率从低权重旧桶里随机漂出 1-3 条「忽然想起来」
- 命中 0 条时回 webhook 报空，并给出可操作的引导文案

不做什么（边界）：
- 不返回 feel/plan/letter（专用通道有自己的入口）
- pinned/protected/permanent 仍可被检索（也是记忆，只是同时在浮现模式置顶）
- dont_surface=True 在检索中保留——主动遗忘只限制无参浮现

对外暴露：surface_search(query, max_results, max_tokens, domain, valence,
                          arousal, tag_filter) → str
========================================
"""

import random

from .. import _runtime as rt
from utils import strip_wikilinks, count_tokens_approx


def _bucket_has_tags(meta: dict, tag_filter: list) -> bool:
    if not tag_filter:
        return True
    bucket_tags = set(meta.get("tags", []) or [])
    return all(t in bucket_tags for t in tag_filter)


async def surface_search(
    query: str,
    max_results: int,
    max_tokens: int,
    domain: str,
    valence: float,
    arousal: float,
    tag_filter: list,
) -> str:
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None

    try:
        matches = await rt.bucket_mgr.search(
            query,
            limit=max(max_results, 20),
            domain_filter=domain_filter,
            query_valence=q_valence,
            query_arousal=q_arousal,
        )
    except Exception as e:
        rt.logger.error(f"Search failed / 检索失败: {e}")
        return "检索过程出错，请稍后重试。"

    matches = [b for b in matches if b["metadata"].get("type") not in ("feel", "plan", "letter")]
    matches = [b for b in matches if _bucket_has_tags(b["metadata"], tag_filter)]

    # --- 向量通道 ---
    matched_ids = {b["id"] for b in matches}
    try:
        vector_results = await rt.embedding_engine.search_similar(query, top_k=max(max_results, 20))
        for bucket_id, sim_score in vector_results:
            if bucket_id not in matched_ids and sim_score > 0.65:
                bucket = await rt.bucket_mgr.get(bucket_id)
                if (
                    bucket
                    and bucket["metadata"].get("type") not in ("feel", "plan", "letter", "archived")
                    and _bucket_has_tags(bucket["metadata"], tag_filter)
                ):
                    bucket["score"] = round(sim_score * 100, 2)
                    bucket["vector_match"] = True
                    matches.append(bucket)
                    matched_ids.add(bucket_id)
    except Exception as e:
        rt.logger.warning(f"Vector search failed, using keyword only / 向量搜索失败: {e}")

    results = []
    token_used = 0
    for bucket in matches:
        if token_used >= max_tokens:
            break
        try:
            clean_meta = {k: v for k, v in bucket["metadata"].items() if k != "tags"}
            # --- 记忆重构：根据当前情绪微调展示层 valence（±0.1）---
            if q_valence is not None and "valence" in clean_meta:
                original_v = float(clean_meta.get("valence") or 0.5)
                shift = (q_valence - 0.5) * 0.2
                clean_meta["valence"] = max(0.0, min(1.0, original_v + shift))
            summary = await rt.dehydrator.dehydrate(strip_wikilinks(bucket["content"]), clean_meta)
            summary_tokens = count_tokens_approx(summary)
            if token_used + summary_tokens > max_tokens:
                break
            await rt.bucket_mgr.touch(bucket["id"])
            meta_b = bucket["metadata"]
            if meta_b.get("pinned") or meta_b.get("protected") or meta_b.get("type") == "permanent":
                summary = f"📌 [核心准则] [bucket_id:{bucket['id']}] {summary}"
            elif bucket.get("vector_match"):
                created = meta_b.get("created", "")[:10]  # Cyrus 定制：带创建日期
                summary = f"[语义关联] [{created}] [bucket_id:{bucket['id']}] {summary}"
            else:
                created = meta_b.get("created", "")[:10]  # Cyrus 定制：带创建日期
                summary = f"[{created}] [bucket_id:{bucket['id']}] {summary}"
            results.append(summary)
            token_used += summary_tokens
        except Exception as e:
            rt.logger.error(
                f"Failed to dehydrate search result / 检索结果脱水失败: {type(e).__name__}: {e}",
                exc_info=True,
            )
            continue

    # --- 检索结果 < 3 时 40% 概率随机浮现 ---
    if len(matches) < 3 and random.random() < 0.4:
        try:
            all_buckets = await rt.bucket_mgr.list_all(include_archive=False)
            matched_ids = {b["id"] for b in matches}
            low_weight = [
                b for b in all_buckets
                if b["id"] not in matched_ids
                and b["metadata"].get("type") not in ("feel", "plan", "letter")
                and rt.decay_engine.calculate_score(b["metadata"]) < 2.0
            ]
            if low_weight:
                drifted = random.sample(low_weight, min(random.randint(1, 3), len(low_weight)))
                drift_results = []
                for b in drifted:
                    clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                    summary = await rt.dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                    drift_results.append(f"[surface_type: random]\n{summary}")
                results.append("--- 忽然想起来 ---\n" + "\n---\n".join(drift_results))
        except Exception as e:
            rt.logger.warning(f"Random surfacing failed / 随机浮现失败: {e}")

    if not results:
        if rt.fire_webhook:
            await rt.fire_webhook("breath", {"mode": "empty", "matches": 0})
        return (
            f"没有匹配到「{query}」相关的记忆。\n"
            "可以换个关键词试试，或不带 query 看当下权重池；feel 用 breath(domain=\"feel\")，信件用 letter_read。"
        )

    final_text = "\n---\n".join(results)
    if rt.fire_webhook:
        await rt.fire_webhook("breath", {"mode": "ok", "matches": len(matches), "chars": len(final_text)})
    return final_text
