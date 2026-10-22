"""Google Gemini API 기반 튜터 엔진. **시뮬레이션 진단 전용**
(``api_common.py`` 모듈 독스트링 참고 -- GRPO/PPO 학습 루프에는 들어가지
않는다).

``google-genai``(``from google import genai``) 를 쓴다 -- 참조한 외부
repo(KyungMinseo1/Dynamic-Peda-RL)의 ``gemini_api_inference.py`` 는 이제
Google이 지원을 완전히 종료한 구 SDK(``google-generativeai``,
``genai.configure``/``GenerativeModel`` 계열)를 쓰지만, 그 패키지를 실제로
설치해서 호출해보니 "All support for the `google.generativeai` package has
ended" 경고가 떠서 신 SDK로 바로 옮겼다. 신 SDK는 ``genai.Client(api_key=...)``
가 매 호출마다 독립된 클라이언트 인스턴스라서, 구 SDK의
``genai.configure()``(모듈 전역 설정)를 여러 스레드가 동시에 건드리며 생기던
키 로테이션 경합도 자연히 없어진다.

이 프로젝트의 실제 튜터 프롬프트(``persona.build_tutor_prompt``,
``persona._chat``)는 항상 정확히 ``[system, user]`` 두 메시지뿐이고(대화
전사는 이미 user 메시지 본문 텍스트 안에 통째로 펼쳐져 있다) 멀티턴 상태를
유지할 필요가 없으므로, 그 경우 ``client.models.generate_content`` 단발
호출로 처리한다. ``ChatPrompt`` 가 그 이상의 멀티턴 메시지를 담고 있는
경우에도 동작하도록(향후 재사용 대비) system 메시지는
``GenerateContentConfig.system_instruction`` 으로 분리하고 나머지는
``client.chats.create(history=...)`` 로 히스토리를 구성한다.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.contracts import ChatPrompt
from src.serving.api_common import ApiCallError, ApiEngineBase

__all__ = ["GeminiEngine"]


def _split_prompt(prompt: ChatPrompt) -> tuple[str | None, list[dict[str, Any]], str]:
    """``prompt`` 를 (system_instruction, 히스토리, 마지막 user 메시지 본문)
    으로 분리한다. 히스토리 원소의 ``{"role": ..., "parts": [{"text": ...}]}``
    형태는 ``google.genai.types.ContentDict``/``PartDict`` 와 그대로 맞는다
    (라이브러리 타입을 직접 구성하지 않고 dict로 넘겨도 pydantic이 검증해서
    받아준다). 마지막 메시지가 system 이면(있을 수 없는 형태지만 방어적으로)
    에러를 낸다 -- 보낼 내용이 없다는 뜻이므로."""
    system_parts = [str(m.get("content", "")) for m in prompt if m.get("role") == "system"]
    non_system = [m for m in prompt if m.get("role") != "system"]
    if not non_system:
        raise ApiCallError("prompt에 system 메시지뿐이라 보낼 내용이 없습니다.")

    history = [
        {
            "role": "user" if m.get("role") == "user" else "model",
            "parts": [{"text": str(m.get("content", ""))}],
        }
        for m in non_system[:-1]
    ]
    last_content = str(non_system[-1].get("content", ""))
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, history, last_content


class GeminiEngine(ApiEngineBase):
    """튜터 프로파일 전용 Gemini 엔진 (``google-genai`` SDK). ``registry.tutor``
    자리에 바로 꽂아 쓴다(``_TutorAdapter`` 로 감쌀 필요 없음).

    Args:
        model: Gemini 모델 이름 (예: ``"gemini-2.5-pro"``,
            ``"gemini-2.5-flash"``, ``"gemini-3.1-flash-lite"``).
        api_key_env: API 키를 읽어올 환경변수 이름. 기본값
            ``"GEMINI_API_KEY"``.
    """

    def __init__(
        self,
        model: str,
        api_key_env: str = "GEMINI_API_KEY",
        *,
        max_workers: int = 20,
        max_retries: int = 5,
    ) -> None:
        super().__init__(model, api_key_env, max_workers=max_workers, max_retries=max_retries)

    def _call_one(self, prompt: ChatPrompt, api_key: str, gen_kwargs: Mapping[str, Any]) -> str:
        # 지연 import: google-genai 가 없는 환경에서도 이 모듈 자체는
        # 문제없이 import 할 수 있어야 한다 (provider: gemini 로 설정했을
        # 때만 실제로 필요).
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        client = genai.Client(api_key=api_key)
        system_instruction, history, last_content = _split_prompt(prompt)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=gen_kwargs.get("temperature"),
            max_output_tokens=gen_kwargs.get("max_tokens"),
        )

        if history:
            chat = client.chats.create(model=self.model, config=config, history=history)
            response = chat.send_message(last_content)
        else:
            response = client.models.generate_content(
                model=self.model, contents=last_content, config=config
            )

        text = response.text
        if not text:
            raise ApiCallError("Gemini가 빈 응답을 반환했습니다.")
        return text
