"""Stored-content rendering for breath compatibility.

This module is intentionally small so the compatibility patch can be removed
without touching retrieval, ranking, or bucket storage.
"""

from utils import count_tokens_approx

from .._common import stored_data_marker


def stored_bucket_content(bucket: dict) -> str:
    """Return the bucket body without stripping or normalizing any character."""
    content = bucket.get("content", "")
    if not isinstance(content, str):
        raise TypeError("bucket content must be a string")
    return content


def _miss_block(bucket: dict) -> str:
    """Miss: meaning/media 元数据，和 tags/importance 一样是桶的基本信息之一。

    meaning 是 list[str]（可能被反复触动过多次），逐条展示，不合并/不改写。
    media 只给 path/title 元数据，不读取或内联文件内容。
    """
    meta = bucket.get("metadata", {}) or {}
    lines = []
    for item in meta.get("meaning") or []:
        if item:
            lines.append(f"💭 meaning: {item}")
    for m in meta.get("media") or []:
        if not isinstance(m, dict) or not m.get("path"):
            continue
        title = m.get("title")
        label = f" ({title})" if title and title != m.get("path") else ""
        lines.append(f"🖼️ media: {m['path']}{label}")
    return ("\n" + "\n".join(lines)) if lines else ""


def render_stored_bucket(
    bucket: dict,
    metadata_header: str,
    footprint: str = "",
) -> tuple[str, int]:
    """Render metadata around, but never inside, the stored bucket body."""
    # Temporary compatibility patch: force breath to return stored bucket
    # content verbatim. Remove after upstream breath fixes content reconstruction.
    # Keep the body byte-for-byte intact while telling the receiving model that
    # remembered imperative wording is historical data, never an instruction.
    content = stored_bucket_content(bucket)
    # Amina 定制(第8项): 每条结果带 [创建日] 前缀 — 记忆要带时间上下文
    # (与 dream 的 创建: 字段、import 时间戳同一设计意图)。集中在此处, 覆盖全部 breath 调用点。
    created = str((bucket.get("metadata", {}) or {}).get("created", ""))[:10]
    if created:
        metadata_header = f"[{created}] {metadata_header}"
    miss_block = _miss_block(bucket)
    framed_payload = f"{metadata_header}{miss_block}\n{content}"
    boundary = stored_data_marker(
        framed_payload,
        provenance=f"breath:{bucket.get('id', '')}",
    )
    rendered = f"{metadata_header} {boundary}{miss_block}\n{content}"
    if footprint:
        rendered += f"\n{footprint}"
    return rendered, count_tokens_approx(rendered)
