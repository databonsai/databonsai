# llm_providers/__init__.py
from .llm_provider import LLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider


# def get_llm_provider(provider_name: str) -> LLMProvider:
#     if provider_name == "openai":
#         return OpenAIProvider()
#     # elif provider_name == 'anthropic':
#     #     return AnthropicProvider()
#     else:
#         raise ValueError(f"Unsupported LLM provider: {provider_name}")
