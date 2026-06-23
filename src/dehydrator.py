"""
========================================
dehydrator.py — 调用 LLM 做「脱水压缩 / 合并 / 打标 / 拆分」
========================================

这个文件包住对外部 LLM 的所有 prompt 和调用。tools/hold、tools/grow、
tools/dream 等都通过它来「让模型做内容理解」，自身不直接拼 prompt。

关键行为：
- dehydrate(content)：把长内容压成高密度摘要，省 token
- merge(old, new)：揉合新旧内容并保持桶体积大致恒定
- analyze(content)：返回 {domain, valence, arousal, tags, suggested_name, importance}
- digest(content)：把日记/长文拆成 2~6 条独立条目（grow 用）
- 走 OpenAI 兼容客户端（DeepSeek / Ollama / LM Studio / vLLM / Gemini 都行）
- SQLite 缓存脱水结果，避免对相同内容重复调用 API

不做什么（边界）：
- 不读写记忆桶文件（不知道 bucket 是什么形态）
- 不决定何时调用、不做去重判断（hold/grow 决定）
- 没 API key 时不报错，返回降级结果（让上层决定怎么办）

对外暴露：Dehydrator 类（dehydrate / merge / analyze / digest）和默认 prompt 字符串
========================================
"""


import os
import re
import json
import hashlib
import sqlite3
import logging
from typing import Optional

from openai import AsyncOpenAI

from utils import count_tokens_approx

logger = logging.getLogger("ombre_brain.dehydrator")


# ============================================================
# 调参面板 / Tunable constants
# ------------------------------------------------------------
# rule.md §①：禁裸魔法数字。这些原本散在五个 _api_* 方法中，
# 集中后调参一眼看完；prompt 模板本身仍在下面以可读性优先。
# ============================================================

# --- LLM 默认参数 ---
_DEFAULT_MODEL = "gemini-2.0-flash"
_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_DEFAULT_MAX_TOKENS = 8192      # Cyrus 定制：脱水/合并输出放宽（原 1024），让记忆存得更全
_DEFAULT_TEMPERATURE = 0.1
_API_TIMEOUT_SECONDS = 60.0

# --- 该多长才需要压缩（低于该 token 数直接走原文）---
_DEHYDRATE_MIN_TOKENS = 100

# --- 各 API 调用的内容截断上限（防 prompt token 超范围）---
_DEHYDRATE_INPUT_LIMIT = 10000   # Cyrus 定制：存更全的记忆，输入截断放宽（原 3000）
_MERGE_INPUT_LIMIT = 10000       # Cyrus 定制（原 2000，新旧各一份）
_ANALYZE_INPUT_LIMIT = 10000     # Cyrus 定制（原 2000）
_DIGEST_INPUT_LIMIT = 20000      # Cyrus 定制：一天日记量大（原 5000）
_PLAN_JUDGE_INPUT_LIMIT = 1500  # plan 与 new event 各一份

# --- 各专用调用的 max_tokens 覆盖 ---
_ANALYZE_MAX_TOKENS = 4096      # Gemini 2.5 thinking 会消耗大量 token，需留足余量
_DIGEST_MAX_TOKENS = 8192       # 日记拆条内容多，thinking + 输出都需要足量空间
_PLAN_JUDGE_MAX_TOKENS = 2048   # thinking 模型下 200 token 完全不够
_PLAN_JUDGE_TEMPERATURE = 0.0   # 判定需确定性
_DIGEST_TEMPERATURE = 0.0       # 拆条需确定性

# --- 默认情感坐标（与 bucket_manager 中保持一致）---
_DEFAULT_VALENCE = 0.5  # 0=极负, 1=极正
_DEFAULT_AROUSAL = 0.3  # 0=完全平静, 1=极激动

# --- 输出截断长度 ---
_TAGS_MAX = 15           # tags 最多保留几个
_DOMAIN_MAX = 3          # domain 最多保留几个（rule.md 推荐选 1~2 个）
_NAME_MAX_CHARS = 20     # suggested_name 上限
_PLAN_REASON_MAX = 200   # plan 判定 reason 上限
_PARSE_ERR_PREVIEW = 200  # JSON 解析失败时日志中 raw 预览长度

# --- importance 范围（与哲学边界一致）---
_IMPORTANCE_MIN = 1
_IMPORTANCE_MAX = 10
_DEFAULT_IMPORTANCE = 5


# --- Dehydration prompt: instructs cheap LLM to compress information ---
# --- 脱水提示词：指导廉价 LLM 压缩信息 ---
DEHYDRATE_PROMPT = """你是一个信息压缩专家。请将以下内容脱水为紧凑摘要。

压缩规则：
1. 提取所有核心事实，去除冗余修饰和重复
2. 保留最新的情绪状态和态度
3. 保留所有待办/未完成事项
4. 关键数字、日期、名称必须保留
5. 目标压缩率 > 70%
6. 用「Amina」和「Cyrus」标注说话人；原文中的「Claude」「AI」「assistant」一律写成「Cyrus」，「用户」「human」「user」一律写成「Amina」；禁止用「她」「他」「我」「他们」「伴侣」等代词替代，避免混淆

输出格式（纯 JSON，无其他内容）：
{
  "core_facts": ["事实1", "事实2"],
  "emotion_state": "当前情绪关键词",
  "todos": ["待办1", "待办2"],
  "keywords": ["关键词1", "关键词2"],
  "summary": "50字以内的核心总结"
}"""


# --- Diary digest prompt: split daily notes into independent memory entries ---
# --- 日记整理提示词：把一大段日常拆分成多个独立记忆条目 ---
DIGEST_PROMPT = """你是一个日记整理专家。她/他会发送一段包含今天各种事情的文本（可能很杂乱），请你将其拆分成多个独立的记忆条目。

整理规则：
1. 每个条目应该是一个独立的主题/事件（不要混在一起）
2. 为每个条目自动分析元数据
3. 去除无意义的口水话和重复信息，保留核心内容
4. 同一主题的零散信息应合并为一个条目
5. 如果有待办事项，单独提取为一个条目
6. 单个条目内容不少于50字，过短的零碎信息合并到最相关的条目中
7. 总条目数控制在 2~6 个，避免过度碎片化
8. 在 content 中对人名、地名、专有名词用 [[双链]] 标记（如 [[人名]]、[[专有名词]]），普通词汇不要加
9. 在 content 中用「Amina」和「Cyrus」标注说话人；原文的「Claude」「AI」「assistant」一律写成「Cyrus」，「用户」「human」「user」一律写成「Amina」；禁止用代词替代，避免混淆

输出格式（纯 JSON 数组，无其他内容）：
[
  {
    "name": "条目标题（10字以内）",
    "content": "整理后的内容",
    "domain": ["主题域1"],
    "valence": 0.7,
    "arousal": 0.4,
    "tags": ["核心词1", "核心词2", "扩展词1", "扩展词2"],
    "importance": 5
  }
]

tags 生成规则：先从原文精准提取 3~5 个核心词，再引申扩展 5~8 个语义相关词（近义词、上位词、关联场景词），合并为一个数组。

主题域可选（选最精确的 1~2 个，只选真正相关的）：
  日常: ["饮食", "穿搭", "出行", "居家", "购物"]
  人际: ["家庭", "恋爱", "友谊", "社交"]
  成长: ["工作", "学习", "考试", "求职"]
  身心: ["健康", "心理", "睡眠", "运动"]
  兴趣: ["游戏", "影视", "音乐", "阅读", "创作", "手工"]
  数字: ["编程", "AI", "硬件", "网络"]
  事务: ["财务", "计划", "待办"]
  内心: ["情绪", "回忆", "梦境", "自省"]
importance: 1-10，根据内容重要程度判断
valence: 0~1（0=消极, 0.5=中性, 1=积极）
arousal: 0~1（0=平静, 0.5=普通, 1=激动）"""


# --- Merge prompt: instruct LLM to blend old and new memories ---
# --- 合并提示词：指导 LLM 揉合新旧记忆 ---
MERGE_PROMPT = """你是一个信息合并专家。请将旧记忆与新内容合并为一份统一的简洁记录。

合并规则：
1. 新内容与旧记忆冲突时，以新内容为准
2. 去除重复信息
3. 保留所有重要事实
4. 总长度尽量不超过旧记忆的 120%
5. 对出现的人名、地名、专有名词用 [[双链]] 标记（如 [[人名]]、[[专有名词]]），普通词汇不要加
6. 用「Amina」和「Cyrus」标注说话人；「Claude」「AI」「assistant」一律写成「Cyrus」，「用户」「human」「user」一律写成「Amina」；禁止用代词替代

直接输出合并后的文本，不要加额外说明。"""


# --- Auto-tagging prompt: analyze content for domain and emotion coords ---
# --- 自动打标提示词：分析内容的主题域和情感坐标 ---
ANALYZE_PROMPT = """你是一个内容分析器。请分析以下文本，输出结构化的元数据。

分析规则：
1. domain（主题域）：选最精确的 1~2 个，只选真正相关的
   日常: ["饮食", "穿搭", "出行", "居家", "购物"]
   人际: ["家庭", "恋爱", "友谊", "社交"]
   成长: ["工作", "学习", "考试", "求职"]
   身心: ["健康", "心理", "睡眠", "运动"]
   兴趣: ["游戏", "影视", "音乐", "阅读", "创作", "手工"]
   数字: ["编程", "AI", "硬件", "网络"]
   事务: ["财务", "计划", "待办"]
   内心: ["情绪", "回忆", "梦境", "自省"]
2. valence（情感效价）：0.0~1.0，0=极度消极 → 0.5=中性 → 1.0=极度积极
3. arousal（情感唤醒度）：0.0~1.0，0=非常平静 → 0.5=普通 → 1.0=非常激动
4. tags（关键词标签）：分两步生成，合并为一个数组：
   第一步—精准提取：从原文抽取 3~5 个真正的核心词，不泛化、不遗漏
   第二步—引申扩展：自动补充 8~10 个与当前场景语义相关的词，包括近义词、上位词、关联场景词、她/他可能用不同措辞搜索的词
   两步合并为一个 tags 数组，总计 10~15 个
5. suggested_name（建议桶名）：10字以内的简短标题
6. 在 tags 和 suggested_name 中不要使用 [[]] 双链标记

输出格式（纯 JSON，无其他内容）：
{
  "domain": ["主题域1", "主题域2"],
  "valence": 0.7,
  "arousal": 0.4,
  "tags": ["核心词1", "核心词2", "扩展词1", "扩展词2", "..."],
  "suggested_name": "简短标题"
}"""


class Dehydrator:
    """
    Data dehydrator + content analyzer.
    Three capabilities: dehydration / merge / auto-tagging (domain + emotion).
    API-only: every public method requires a working LLM API.
    If the API is unavailable, methods raise RuntimeError so callers can
    surface the failure to the user instead of silently producing low-quality results.
    数据脱水器 + 内容分析器。
    三大能力：脱水压缩 / 新旧合并 / 自动打标。
    仅走 API：API 不可用时直接抛出 RuntimeError，调用方明确感知。
    （根据 BEHAVIOR_SPEC.md 三、降级行为表决策：无本地降级）
    """

    def __init__(self, config: dict):
        # --- Read dehydration API config / 读取脱水 API 配置 ---
        dehy_cfg = config.get("dehydration", {})
        self.api_key = dehy_cfg.get("api_key", "")
        self.model = dehy_cfg.get("model", _DEFAULT_MODEL)
        self.base_url = dehy_cfg.get("base_url", _DEFAULT_BASE_URL)
        self.max_tokens = dehy_cfg.get("max_tokens", _DEFAULT_MAX_TOKENS)
        self.temperature = dehy_cfg.get("temperature", _DEFAULT_TEMPERATURE)
        # api_format: "openai_compat" (default) | "gemini" | "anthropic"
        self.api_format = dehy_cfg.get("api_format", "openai_compat")
        # Auto-detect new Google AI Studio key format (AQ.*): these keys are not accepted
        # by the OpenAI-compat endpoint (/v1beta/openai/) and must use the native
        # generateContent API. Switch automatically so users don't need to set api_format manually.
        if (
            self.api_format == "openai_compat"
            and self.api_key.startswith("AQ.")
            and "generativelanguage.googleapis.com" in (self.base_url or "")
        ):
            self.api_format = "gemini"
            logger.info("AQ.* key + generativelanguage.googleapis.com detected — auto-switching to native Gemini API")
        # thinking_budget: 仅 Gemini 2.5+/3.x「思考型」模型生效。默认 0 = 关闭思考。
        # 关键：gemini-3.5-flash 等模型默认会先消耗 output token 做「思考」，当
        # max_tokens 较小时思考会吃光预算 → 返回空文本（这正是脱水/抽取偶发返回
        # 空、报 "LLM extraction failed" 的根因）。脱水/抽取是机械式转换，不需要
        # 思考，关掉它既修了空输出、又更快更省。设为 None 可彻底不发该字段（兼容
        # 不支持 thinkingConfig 的老模型）。
        self.thinking_budget = dehy_cfg.get("thinking_budget", 0)

        # --- API availability / 是否有可用的 API ---
        self.api_available = bool(self.api_key)

        # --- Initialize OpenAI-compatible client (only for openai_compat format) ---
        # --- 初始化 OpenAI 兼容客户端（仅 openai_compat 格式使用）---
        self.client: Optional[AsyncOpenAI] = None
        if self.api_available and self.api_format == "openai_compat":
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=_API_TIMEOUT_SECONDS,
            )

        # --- SQLite dehydration cache ---
        # --- SQLite 脱水缓存：content hash → summary ---
        db_path = os.path.join(config["buckets_dir"], "dehydration_cache.db")
        self.cache_db_path = db_path
        self._cache_conn: sqlite3.Connection = self._init_cache_db()

    def _init_cache_db(self) -> sqlite3.Connection:
        """Open (or create) the dehydration cache DB; return a persistent connection."""
        os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)
        # check_same_thread=False is safe here: asyncio runs on one thread and all
        # cache calls are synchronous helper methods called from that same thread.
        conn = sqlite3.connect(self.cache_db_path, check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dehydration_cache (
                content_hash TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        return conn

    def _get_cached_summary(self, content: str) -> str | None:
        """Look up cached dehydration result by content hash."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        row = self._cache_conn.execute(
            "SELECT summary FROM dehydration_cache WHERE content_hash = ?",
            (content_hash,)
        ).fetchone()
        return row[0] if row else None

    def _set_cached_summary(self, content: str, summary: str):
        """Store dehydration result in cache."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        self._cache_conn.execute(
            "INSERT OR REPLACE INTO dehydration_cache (content_hash, summary, model) VALUES (?, ?, ?)",
            (content_hash, summary, self.model)
        )
        self._cache_conn.commit()

    def invalidate_cache(self, content: str):
        """Remove cached summary for specific content (call when bucket content changes)."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        self._cache_conn.execute(
            "DELETE FROM dehydration_cache WHERE content_hash = ?", (content_hash,)
        )
        self._cache_conn.commit()

    # ---------------------------------------------------------
    # 内部 helpers / Internal helpers
    # ---------------------------------------------------------
    def _require_api(self) -> None:
        """API 不可用时抛出统一文案的 RuntimeError。

        原本 dehydrate / merge / analyze / digest 各处都重复
        `if not self.api_available: raise RuntimeError("...")`，
        统一后调用方一行 `self._require_api()` 即可，且文案改一处全部生效。
        """
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用，请检查 config.yaml 中的 dehydration 配置")

    async def _chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """统一的 OpenAI-compatible chat 调用。

        原本 5 个 _api_* 方法重复了同样的样板：
          * 构造 messages
          * 调 client.chat.completions.create
          * 检查 response.choices 非空
          * 取 choices[0].message.content 并兜底空字符串
        统一后：
          * 调用方传入 system + user prompt 与可选的 max_tokens / temperature
          * 默认值取 self.max_tokens / self.temperature（由 config.yaml 决定）
          * 始终返回 str（response 异常时返回空串，调用方各自决策）

        参数：
            system, user — Chat completion 的 system/user 消息
            max_tokens   — 覆盖默认（如 analyze 用 256，digest 用 2048）
            temperature  — 覆盖默认（如 digest / plan_judge 需要 0.0）
        """
        if self.api_format == "gemini":
            return await self._chat_gemini(system, user, max_tokens=max_tokens, temperature=temperature)
        if self.api_format == "anthropic":
            return await self._chat_anthropic(system, user, max_tokens=max_tokens, temperature=temperature)
        if self.api_format != "openai_compat":
            logger.warning(f"Unknown api_format '{self.api_format}', falling back to openai_compat")
        # openai_compat (default)
        if self.client is None:
            return ""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
        )
        if not response.choices:
            return ""
        text = response.choices[0].message.content or ""
        # Cyrus 定制：剥掉 DeepSeek 等推理模型吐的 <think>...</think>，不让思考过程混进记忆
        text = re.sub(r'<think>[\s\S]*?</think>', '', text)
        return text.replace('</think>', '').strip()

    async def _chat_gemini(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Native Gemini generateContent API call (no OpenAI-compat wrapper)."""
        if not self.api_key:
            return ""
        import httpx
        # Strip any accidental "models/" prefix — Google rejects double-prefix in the URL
        model_id = self.model.removeprefix("models/").strip()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
        payload: dict = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens if max_tokens is not None else self.max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
            },
        }
        # 关闭/限制思考预算（见 __init__ 的 thinking_budget 说明）。
        if self.thinking_budget is not None:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        async with httpx.AsyncClient(timeout=_API_TIMEOUT_SECONDS) as client:
            r = await client.post(url, params={"key": self.api_key}, json=payload)
            r.raise_for_status()
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return parts[0].get("text", "") if parts else ""

    async def _chat_anthropic(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Native Anthropic Messages API call."""
        if not self.api_key:
            return ""
        import httpx
        base = self.base_url.rstrip("/") if self.base_url else "https://api.anthropic.com"
        url = f"{base}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": temperature if temperature is not None else self.temperature,
        }
        async with httpx.AsyncClient(timeout=_API_TIMEOUT_SECONDS) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
        data = r.json()
        content = data.get("content", [])
        if not content:
            return ""
        first = content[0]
        return first.get("text", "") if isinstance(first, dict) else ""

    @staticmethod
    def _strip_md_fence(raw: str) -> str:
        """剥掉 LLM 偶尔会包的 ```...``` 代码块外壳。

        DeepSeek / Gemini 在被要求"返回纯 JSON"时仍偶尔把 JSON 包进
        ```json\n{...}\n``` 里。三处 JSON 解析都得做这层剥离，
        所以统一抽到这里。原始字符串不含围栏时原样返回。
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
        return cleaned

    @staticmethod
    def _clamp_va(
        meta: dict,
        default_v: float = _DEFAULT_VALENCE,
        default_a: float = _DEFAULT_AROUSAL,
    ) -> tuple[float, float]:
        """读取 meta 中的 valence / arousal 并钳制到 [0, 1]。

        三处 LLM 返回校验逻辑相同（_format_output / _parse_analysis / _parse_digest），
        集中后保证三处行为一致：解析失败一律回 (默认 V, 默认 A)。
        """
        try:
            v = max(0.0, min(1.0, float(meta.get("valence", default_v))))
            a = max(0.0, min(1.0, float(meta.get("arousal", default_a))))
            return v, a
        except (ValueError, TypeError):
            return default_v, default_a

    # ---------------------------------------------------------
    # Dehydrate: compress raw content into concise summary
    # 脱水：将原始内容压缩为精简摘要
    # API only (no local fallback)
    # 仅通过 API 脱水（无本地回退）
    # ---------------------------------------------------------
    async def dehydrate(self, content: str, metadata: Optional[dict] = None) -> str:
        """
        Dehydrate/compress memory content.
        Returns formatted summary string ready for Claude context injection.
        Uses SQLite cache to avoid redundant API calls.
        对记忆内容做脱水压缩。
        返回格式化的摘要字符串，可直接注入 Claude 上下文。
        使用 SQLite 缓存避免重复调用 API。
        """
        if not content or not content.strip():
            return "（空记忆 / empty memory）"

        # --- Content is short enough, no compression needed ---
        # --- 内容已经很短，不需要压缩 ---
        if count_tokens_approx(content) < _DEHYDRATE_MIN_TOKENS:
            return self._format_output(content, metadata)

        # --- Check cache first ---
        # --- 先查缓存 ---
        cached = self._get_cached_summary(content)
        if cached:
            return self._format_output(cached, metadata)

        # --- API dehydration (no local fallback) ---
        # --- API 脱水（无本地降级）---
        self._require_api()

        result = await self._api_dehydrate(content)
        # --- Cache the result ---
        self._set_cached_summary(content, result)
        return self._format_output(result, metadata)

    # ---------------------------------------------------------
    # Merge: blend new content into existing bucket
    # 合并：将新内容揉入已有桶，保持体积恒定
    # ---------------------------------------------------------
    async def merge(self, old_content: str, new_content: str) -> str:
        """
        Merge new content with old memory, preventing infinite bucket growth.
        将新内容与旧记忆合并，避免桶无限膨胀。
        """
        if not old_content and not new_content:
            return ""
        if not old_content:
            return new_content or ""
        if not new_content:
            return old_content

        # --- API merge (no local fallback) ---
        self._require_api()
        try:
            result = await self._api_merge(old_content, new_content)
            if result:
                return result
            raise RuntimeError("API 合并返回空结果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 合并失败，请检查 API 连接: {e}") from e

    # ---------------------------------------------------------
    # API call: dehydration
    # API 调用：脱水压缩
    # ---------------------------------------------------------
    async def _api_dehydrate(self, content: str) -> str:
        """
        Call LLM API for intelligent dehydration (via OpenAI-compatible client).
        调用 LLM API 执行智能脱水。
        """
        return await self._chat(DEHYDRATE_PROMPT, content[:_DEHYDRATE_INPUT_LIMIT])

    # ---------------------------------------------------------
    # API call: merge
    # API 调用：合并
    # ---------------------------------------------------------
    async def _api_merge(self, old_content: str, new_content: str) -> str:
        """
        Call LLM API for intelligent merge (via OpenAI-compatible client).
        调用 LLM API 执行智能合并。
        """
        user_msg = (
            f"旧记忆：\n{old_content[:_MERGE_INPUT_LIMIT]}\n\n"
            f"新内容：\n{new_content[:_MERGE_INPUT_LIMIT]}"
        )
        return await self._chat(MERGE_PROMPT, user_msg)

    # ---------------------------------------------------------
    # Output formatting
    # 输出格式化
    # Wraps dehydrated result with bucket name, tags, emotion coords
    # 把脱水结果包装成带桶名、标签、情感坐标的可读文本
    # ---------------------------------------------------------

    def _format_output(self, content: str, metadata: Optional[dict] = None) -> str:
        """
        Format dehydrated result into context-injectable text.
        将脱水结果格式化为可注入上下文的文本。
        """
        header = ""
        if metadata and isinstance(metadata, dict):
            name = metadata.get("name", "未命名")
            domains = ", ".join(metadata.get("domain", []))
            valence, arousal = self._clamp_va(metadata)
            header = f"📌 记忆桶: {name}"
            if domains:
                header += f" [主题:{domains}]"
            header += f" [情感:V{valence:.1f}/A{arousal:.1f}]"
            # Show model's perspective if available (valence drift)
            model_v = metadata.get("model_valence")
            if model_v is not None:
                try:
                    header += f" [我的视角:V{float(model_v):.1f}]"
                except (ValueError, TypeError):
                    pass
            if metadata.get("digested"):
                header += " [已消化]"
            header += "\n"

        # 去掉 keywords 字段：LLM 返回的 JSON 里 keywords 是内部索引用途，不暴露给上下文
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "keywords" in parsed:
                parsed.pop("keywords", None)
                content = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass  # 非 JSON 内容直接透传
        content = re.sub(r'\[\[([^\]]+)\]\]', r'\1', content)
        return f"{header}{content}"

    # ---------------------------------------------------------
    # Auto-tagging: analyze content for domain + emotion + tags
    # 自动打标：分析内容，输出主题域 + 情感坐标 + 标签
    # Called by server.py when storing new memories
    # 存新记忆时由 server.py 调用
    # ---------------------------------------------------------
    async def analyze(self, content: str) -> dict:
        """
        Analyze content and return structured metadata.
        分析内容，返回结构化元数据。

        Returns: {"domain", "valence", "arousal", "tags", "suggested_name"}
        """
        if not content or not content.strip():
            return self._default_analysis()

        # --- API analyze (no local fallback) ---
        self._require_api()
        try:
            result = await self._api_analyze(content)
            if result:
                return result
            raise RuntimeError("API 打标返回空结果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 打标失败，请检查 API 连接: {e}") from e

    # ---------------------------------------------------------
    # API call: auto-tagging
    # API 调用：自动打标
    # ---------------------------------------------------------
    async def _api_analyze(self, content: str) -> dict:
        """
        Call LLM API for content analysis / tagging.
        调用 LLM API 执行内容分析打标。
        """
        raw = await self._chat(
            ANALYZE_PROMPT,
            content[:_ANALYZE_INPUT_LIMIT],
            max_tokens=_ANALYZE_MAX_TOKENS,
            temperature=_DEFAULT_TEMPERATURE,
        )
        if not raw.strip():
            return self._default_analysis()
        return self._parse_analysis(raw)

    # ---------------------------------------------------------
    # Parse API JSON response with safety checks
    # 解析 API 返回的 JSON，做安全校验
    # Ensure valence/arousal in 0~1, domain/tags valid
    # ---------------------------------------------------------
    def _parse_analysis(self, raw: str) -> dict:
        """
        Parse and validate API tagging result.
        解析并校验 API 返回的打标结果。
        """
        try:
            cleaned = self._strip_md_fence(raw)
            result = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, ValueError):
            logger.warning(f"API tagging JSON parse failed / JSON 解析失败: {raw[:_PARSE_ERR_PREVIEW]}")
            return self._default_analysis()

        if not isinstance(result, dict):
            return self._default_analysis()

        # --- Validate and clamp value ranges / 校验并钳制数值范围 ---
        valence, arousal = self._clamp_va(result)

        return {
            "domain": result.get("domain", ["未分类"])[:_DOMAIN_MAX],
            "valence": valence,
            "arousal": arousal,
            "tags": result.get("tags", [])[:_TAGS_MAX],
            "suggested_name": str(result.get("suggested_name", ""))[:_NAME_MAX_CHARS],
        }

    # ---------------------------------------------------------
    # Default analysis result (empty content or total failure)
    # 默认分析结果（内容为空或完全失败时用）
    # ---------------------------------------------------------
    def _default_analysis(self) -> dict:
        """
        Return default neutral analysis result.
        返回默认的中性分析结果。
        """
        return {
            "domain": ["未分类"],
            "valence": _DEFAULT_VALENCE,
            "arousal": _DEFAULT_AROUSAL,
            "tags": [],
            "suggested_name": "",
        }

    # ---------------------------------------------------------
    # Diary digest: split daily notes into independent memory entries
    # 日记整理：把一大段日常拆分成多个独立记忆条目
    # For the "grow" tool — "dump a day's content and it gets organized"
    # 给 grow 工具用，"一天结束发一坨内容"靠这个
    # ---------------------------------------------------------
    async def digest(self, content: str) -> list[dict]:
        """
        Split a large chunk of daily content into independent memory entries.
        将一大段日常内容拆分成多个独立记忆条目。

        Returns: [{"name", "content", "domain", "valence", "arousal", "tags", "importance"}, ...]
        """
        if not content or not content.strip():
            return []

        # --- API digest (no local fallback) ---
        self._require_api()
        try:
            result = await self._api_digest(content)
            if result:
                return result
            raise RuntimeError("API 日记整理返回空结果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 日记整理失败，请检查 API 连接: {e}") from e

    # ---------------------------------------------------------
    # API call: diary digest
    # API 调用：日记整理
    # ---------------------------------------------------------
    async def _api_digest(self, content: str) -> list[dict]:
        """
        Call LLM API for diary organization.
        调用 LLM API 执行日记整理。
        """
        raw = await self._chat(
            DIGEST_PROMPT,
            content[:_DIGEST_INPUT_LIMIT],
            max_tokens=_DIGEST_MAX_TOKENS,
            temperature=_DIGEST_TEMPERATURE,
        )
        if not raw.strip():
            return []
        return self._parse_digest(raw)

    # ---------------------------------------------------------
    # Parse diary digest result with safety checks
    # 解析日记整理结果，做安全校验
    # ---------------------------------------------------------
    def _parse_digest(self, raw: str) -> list[dict]:
        """
        Parse and validate API diary digest result.
        解析并校验 API 返回的日记整理结果。
        """
        try:
            cleaned = self._strip_md_fence(raw)
            items = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, ValueError):
            logger.warning(f"Diary digest JSON parse failed / JSON 解析失败: {raw[:_PARSE_ERR_PREVIEW]}")
            return []

        if not isinstance(items, list):
            return []

        validated = []
        for item in items:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            try:
                importance = max(
                    _IMPORTANCE_MIN,
                    min(_IMPORTANCE_MAX, int(item.get("importance", _DEFAULT_IMPORTANCE))),
                )
            except (ValueError, TypeError):
                importance = _DEFAULT_IMPORTANCE
            valence, arousal = self._clamp_va(item)

            validated.append({
                "name": str(item.get("name", ""))[:_NAME_MAX_CHARS],
                "content": str(item.get("content", "")),
                "domain": item.get("domain", ["未分类"])[:_DOMAIN_MAX],
                "valence": valence,
                "arousal": arousal,
                "tags": item.get("tags", [])[:_TAGS_MAX],
                "importance": importance,
            })
        return validated

    # ---------------------------------------------------------
    # API call: judge whether a new event resolves an active plan
    # API 调用：判断新事件是否完成了某个 active plan
    # ---------------------------------------------------------
    async def judge_plan_resolution(self, plan_text: str, new_event_text: str) -> dict:
        """
        Conservative judgement (鼓励漏报，避免误报).
        保守判断：仅在新事件明确表示 plan 已完成时返回 resolved=True。
        Returns: {"resolved": bool, "confidence": float, "reason": str}
        Returns {"resolved": False} silently when API unavailable.
        """
        if not self.api_available:
            return {"resolved": False, "confidence": 0.0, "reason": "API 不可用"}
        system = (
            "你是一个保守的计划完成判断器。给定一条 plan 和一条新事件，"
            "只在新事件明确表示该 plan 已被完成、放弃或不再相关时，输出 resolved=true；"
            "其它情况一律 false。返回严格 JSON：{\"resolved\": true/false, \"confidence\": 0~1, \"reason\": \"...\"}。"
            "不要解释、不要 markdown、不要多余文本。"
        )
        user = (
            f"PLAN:\n{plan_text[:_PLAN_JUDGE_INPUT_LIMIT]}\n\n"
            f"NEW EVENT:\n{new_event_text[:_PLAN_JUDGE_INPUT_LIMIT]}"
        )
        try:
            raw = await self._chat(
                system,
                user,
                max_tokens=_PLAN_JUDGE_MAX_TOKENS,
                temperature=_PLAN_JUDGE_TEMPERATURE,
            )
            if not raw:
                return {"resolved": False, "confidence": 0.0, "reason": "空响应"}
            cleaned = self._strip_md_fence(raw)
            data = json.loads(cleaned)
            return {
                "resolved": bool(data.get("resolved", False)),
                "confidence": float(data.get("confidence", 0.0)),
                "reason": str(data.get("reason", ""))[:_PLAN_REASON_MAX],
            }
        except Exception as e:
            logger.warning(f"judge_plan_resolution failed: {e}")
            return {"resolved": False, "confidence": 0.0, "reason": str(e)}
