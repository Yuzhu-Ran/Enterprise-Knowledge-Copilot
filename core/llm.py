"""
统一的大模型调用入口：通过百炼的 OpenAI 兼容接口调用 Qwen。
"""
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import settings


@lru_cache
def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """返回缓存的 Qwen 客户端，支持异步调用和流式输出。"""
    api_key = settings.dashscope_api_key.strip()
    if not api_key:
        raise ValueError(
            "未配置 DASHSCOPE_API_KEY，请在项目根目录的 .env 中填写百炼密钥。"
        )

    return ChatOpenAI(
        model=settings.qwen_model,
        api_key=api_key,
        base_url=settings.dashscope_base_url,
        temperature=temperature,
        extra_body={"enable_thinking": False},
        use_responses_api=False,
        timeout=60,
        max_retries=2,
    )
