"""
========================================
tools/breath/importance.py — importance_min 模式
========================================

走 breath(importance_min=N) 时进入这里。跳过语义检索，按 importance
降序拉所有标过的核心事项（最多 20 条），让模型一次性扫一眼自己
认定为「重要」的桶。

关键行为：
- 列出所有非 feel/plan/letter、未主动遗忘、且 importance >= 阈值 的桶
- tags 过滤同样生效（AND）
- 按 importance 降序，截到 20 条，逐条 dehydrate 后塞进 max_tokens 预算

不做什么（边界）：
- 不做向量检索（这是「按重要度批量拉」而不是「找相似」）
- 不主动 touch（浮现行为不应重置衰减计时器）

对外暴露：surface_by_importance(importance_min, max_tokens, tag_filter) → str
========================================
"""

from .. import _runtime as rt
from utils import strip_wikilinks, count_tokens_approx


def _bucket_has_tags(meta: dict, tag_filter: list) -> bool:
    if not tag_filter:
        return True
    bucket_tags = set(meta.get("tags", []) or [])
    return all(t in bucket_tags for t in tag_filter)


async def surface_by_importance(importance_min: int, max_tokens: int, tag_filter: list) -> str:
    try:
        all_buckets = await rt.bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        return f"记忆系统暂时无法访问: {e}"
    filtered = [
        b for b in all_buckets
        if int(b.get("metadata", {}).get("importance") or 0) >= importance_min
        and b.get("metadata", {}).get("type") not in ("feel", "plan", "letter")
        and not b.get("metadata", {}).get("dont_surface", False)
        and _bucket_has_tags(b.get("metadata", {}), tag_filter)
    ]
    filtered.sort(key=lambda b: int(b.get("metadata", {}).get("importance") or 0), reverse=True)
    filtered = filtered[:20]
    if not filtered:
        return f"没有重要度 >= {importance_min} 的记忆。"
    results = []
    token_used = 0
    for b in filtered:
        if token_used >= max_tokens:
            break
        try:
            clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
            try:
                summary = await rt.dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
            except Exception as dehy_err:
                rt.logger.warning(f"importance_min dehydrate failed / 脱水失败: {dehy_err}")
                is_pinned = b["metadata"].get("pinned") or b["metadata"].get("protected")
                if is_pinned:
                    # pinned 桶脱水失败时降级展示原文，确保核心准则可见
                    summary = strip_wikilinks(b["content"])[:300].strip() or "（空记忆）"
                else:
                    continue
            t = count_tokens_approx(summary)
            if token_used + t > max_tokens:
                break
            imp = b["metadata"].get("importance", 0)
            created = b["metadata"].get("created", "")[:10]  # Cyrus 定制：带创建日期
            results.append(f"[importance:{imp}] [{created}] [bucket_id:{b['id']}] {summary}")
            token_used += t
        except Exception as e:
            rt.logger.warning(f"importance_min bucket processing failed: {e}")
    return "\n---\n".join(results) if results else "没有可以展示的记忆。"
