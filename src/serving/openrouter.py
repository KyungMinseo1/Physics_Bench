"""OpenRouter(여러 모델 제공사를 하나의 OpenAI 호환 API로 묶어주는 라우팅
서비스) 기반 튜터 엔진. **시뮬레이션 진단 전용** (``api_common.py`` 모듈
독스트링 참고 -- GRPO/PPO 학습 루프에는 들어가지 않는다).

참조한 외부 repo(KyungMinseo1/Dynamic-Peda-RL)의
``src/inference_providers/open_router_inference.py`` 와 같은 서비스·같은
``openai`` 클라이언트를 쓰지만, vLLM ``RequestOutput`` 흉내를 내지 않고
``contracts.Engine`` 을 직접 만족한다(``ApiEngineBase`` 참고).
"""

from __future__ import annotations

from typing import Any, Mapping

from src.contracts import ChatPrompt
from src.serving.api_common import ApiCallError, ApiEngineBase

__all__ = ["OpenRouterEngine"]

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterEngine(ApiEngineBase):
    """튜터 프로파일 전용 OpenRouter 엔진. ``registry.tutor`` 자리에 바로
    꽂아 쓴다(``_TutorAdapter`` 로 감쌀 필요 없음 -- 이미
    ``contracts.Engine`` 을 만족한다).

    Args:
        model: OpenRouter 모델 id (예: ``"google/gemini-2.5-pro"``,
            ``"anthropic/claude-sonnet-4.5"``). OpenRouter 대시보드의 모델
            목록에 있는 문자열 그대로 써야 한다.
        api_key_env: API 키를 읽어올 환경변수 이름. 기본값
            ``"OPENROUTER_API_KEY"``. 같은 접두어의 다른 환경변수
            (``OPENROUTER_API_KEY_2`` 등)가 있으면 재시도 때 로테이션 대상에
            자동으로 포함된다 (``api_common.resolve_api_keys`` 참고).
    """

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
