"""
========================================
tools/grow/core.py — grow 长内容主路径（digest + merge）
========================================

长内容（≥30 字）走这里。先调 dehydrator.digest 把整段拆成 2~6 条
事件项，每条独立尝试 merge_or_create。

关键行为：
- digest 失败（API key 不可用）时直接 RuntimeError，不创建任何桶
- 逐条调 merge_or_create（grow 路径用 LLM merge，会压缩老+新）
- iter 2.0：每次 grow 调用生成一个 ``grow_batch_id``，同批次新建桶共享，
  source_tool 一律为 ``grow``；合并到的老桶不改 source_tool
- 单条失败不影响其他；按字节上限校验单条尺寸
- embedding 失败时桶正常创建，返回追加向量化降级警告
- 末尾 fire-and-forget 触发 plan 自动闭环（用整段原文做匹配）

不做什么（边界）：
- 不写 feel：grow 是事件归档，不是反思
- 不做 pinned 标记：grow 拆出来的事件桶都是 dynamic
- 不接受 why_remembered：grow 是整理，拆出来的每条桶就是事件本身，是 why 本身

对外暴露：grow_core(content) → str
========================================
"""

import asyncio
import uuid

from .. import _runtime as rt
from .._common import (
    merge_or_create,
    check_content_size,
    check_duplicate_for,
    check_plan_resolution,
)


async def grow_core(content: str) -> str:
    try:
        items = await rt.dehydrator.digest(content)
    except Exception as e:
        rt.logger.error(f"Diary digest failed / 日记整理失败: {e}")
        raise RuntimeError(
            f"API key 未配置或调用失败，日记拆分无法完成，桶未创建。请检查 OMBRE_COMPRESS_API_KEY。（错误：{e}）"
        ) from e

    if not items:
        return "内容为空或整理失败。"

    # iter 2.0 来源追踪：同一次 grow 拆出的所有桶共享同一个 batch_id，
    # dashboard 可按 grow_batch_id 聚合显示「这次日记一共归档了哪些事件」。
    # 用 12 位 hex 与 bucket_id 长度对齐，加 g_ 前缀方便人眼区分。
    batch_id = f"g_{uuid.uuid4().hex[:12]}"

    results = []
    created = 0
    merged = 0
    embed_warnings = []

    for item in items:
        try:
            size_err = check_content_size(item.get("content", ""))
            if size_err:
                results.append(f"⚠️{item.get('name', '?')}（{size_err}）")
                continue
            result_name, is_merged, embed_warn = await merge_or_create(
                content=item["content"],
                tags=item.get("tags") or [],
                importance=item.get("importance") or 5,
                domain=item.get("domain") or ["未分类"],
                valence=item.get("valence") or 0.5,
                arousal=item.get("arousal") or 0.3,
                name=item.get("name", ""),
                source_tool="grow",
                grow_batch_id=batch_id,
            )
            if embed_warn and embed_warn not in embed_warnings:
                embed_warnings.append(embed_warn)

            if is_merged:
                results.append(f"📎{result_name}")
                merged += 1
            else:
                # 露出新建桶的 bucket_id (不只是名字): 让调用方 grow 完能直接拿 id 写 feel
                # (上游只回 name → 想对刚 grow 的桶写 feel 还得 breath 搜一趟拿 id)。Amina 定制。
                results.append(f"📝{item.get('name', result_name)}→{result_name}")
                created += 1
                asyncio.create_task(check_duplicate_for(result_name, item["content"]))
        except Exception as e:
            rt.logger.warning(
                f"Failed to process diary item / 日记条目处理失败: "
                f"{item.get('name', '?')}: {e}"
            )
            results.append(f"⚠️{item.get('name', '?')}")

    asyncio.create_task(check_plan_resolution(content))
    summary = f"{len(items)}条|新{created}合{merged} batch:{batch_id}\n" + "\n".join(results)
    if embed_warnings:
        summary += f"\n⚠️ {embed_warnings[0]}"
    return summary
