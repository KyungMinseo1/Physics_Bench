from __future__ import annotations

from typing import Any, Mapping

from src.contracts import ChatPrompt
from src.serving.api_common import ApiCallError, ApiEngineBase

__all__ = ["OpenRouterEngine"]

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterEngine(ApiEngineBase):
    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        *,
        max_workers: int = 20,
        max_retries: int = 5,
    ) -> None:
        super().__init__(model, api_key_env, max_workers=max_workers, max_retries=max_retries)

    def _call_one(self, prompt: ChatPrompt, api_key: str, gen_kwargs: Mapping[str, Any]) -> str:
        # 지연 import: openai 패키지가 없는 환경(예: GPU 전용 학습 노드)에서도
        # 이 모듈 자체는 문제없이 import 할 수 있어야 한다 -- 실제로 이
        # 엔진을 쓸 때만(즉 provider: openrouter 로 설정했을 때만) 필요하다.
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(base_url=_BASE_URL, api_key=api_key)
        completion = client.chat.completions.create(
            model=self.model,
            messages=list(prompt),
            temperature=gen_kwargs.get("temperature"),
            max_tokens=gen_kwargs.get("max_tokens"),
        )
        text = completion.choices[0].message.content
        if not text:
            raise ApiCallError("OpenRouter가 빈 응답을 반환했습니다.")
        return text
