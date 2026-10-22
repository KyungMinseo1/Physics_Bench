"""OpenRouter/Gemini 같은 closed API 기반 튜터 엔진들이 공유하는 실행
뼈대. **시뮬레이션 진단 전용** — GRPO/PPO + LoRA로 학습시키는 실제 튜터
정책은 여전히 로컬 vLLM(``src/serving/parallel_vllm.py``)만 쓴다. 이 모듈은
"학습 전 base 모델이 few-shot 예시를 그대로 베끼는 문제인지, prompt/환경
설계 결함인지"를 구분하기 위해 같은 roster를 더 강한 모델로 한 번 돌려보는
용도로만 쓰인다 (docs 참고 없음 -- 이번 진단 세션에서 결정된 범위).

참고한 외부 repo(KyungMinseo1/Dynamic-Peda-RL)의
``src/inference_providers/{gemini_api_inference,open_router_inference}.py``
는 vLLM의 ``RequestOutput``/``CompletionOutput`` 타입을 그대로 흉내내
반환한다 (그 repo 전체가 vLLM 타입에 묶여 있어서). 이 프로젝트는
``contracts.Engine``(``run_batch(prompts, profile, overrides) -> list[str]``)
로 vLLM 타입을 이미 완전히 은닉해뒀고 ``src/serving/mock.py`` 의
``MockEngine`` 이 그 프로토콜을 직접 구현하는 선례이므로, 여기서도 같은
패턴을 따른다 -- API 전용 엔진은 vLLM을 전혀 import 하지 않는다.

튜터 프로파일(``tutor``/``regression``)만 지원한다: guided-JSON도 ``n>1``
샘플링도 쓰지 않는 프로파일이라서다 (``sampling.py`` 의
``_PROFILE_DEFAULTS``/``_GUIDED_SCHEMAS`` 참고). 학생 역할(bid/speak/judge/
posttest)은 guided-JSON 스키마 강제가 필요해 범위 밖이고, 학생 엔진은
계속 로컬 vLLM만 쓴다.
"""

from __future__ import annotations

import concurrent.futures
import os
import random
import time
from typing import Any, Callable, Mapping, Sequence

from src.contracts import ChatPrompt, SamplingProfile
from src.serving.sampling import profile_generation_kwargs

__all__ = ["ApiCallError", "resolve_api_keys", "run_prompts_concurrently", "ApiEngineBase"]

_VALID_PROFILES: frozenset[str] = frozenset({"tutor", "regression"})


class ApiCallError(RuntimeError):
    """프롬프트 1개에 대한 API 호출이 재시도를 다 소진하고도 실패했을 때
    내부적으로만 쓰는 예외. ``run_prompts_concurrently`` 밖으로는 절대
    전파되지 않는다 -- 최종적으로는 빈 문자열로 낮춰진다(모듈 독스트링
    참고, ``parse_tutor_output`` 이 빈 입력을 이미 안전하게 처리한다)."""


def resolve_api_keys(api_key_env: str) -> list[str]:
    """``api_key_env`` 자체와, 같은 접두어로 시작하는 다른 환경변수
    (``{api_key_env}_2``, ``{api_key_env}_3`` 처럼 여러 키를 등록해 rate
    limit을 분산하고 싶을 때 쓰는 확장 규칙)를 전부 모아 반환한다. 참조
    repo(Dynamic-Peda-RL)의 fallback 키 로테이션 규칙과 동일하다.

    맨 앞 값이 항상 ``api_key_env`` 자기 자신이어야 한다(첫 시도는 항상 이
    키를 쓴다는 뜻) -- 나머지는 정렬 순서로 뒤따른다.
    """
    primary = os.getenv(api_key_env)
    if not primary:
        raise ValueError(f"{api_key_env} 환경변수가 설정되어 있지 않습니다.")
    fallbacks = sorted(
        k for k in os.environ if k.startswith(api_key_env) and k != api_key_env
    )
    keys = [primary] + [os.environ[k] for k in fallbacks]
    return keys


def run_prompts_concurrently(
    prompts: Sequence[ChatPrompt],
    call_one: Callable[[ChatPrompt, str], str],
    api_keys: Sequence[str],
    *,
    max_workers: int = 20,
    max_retries: int = 5,
    max_backoff_seconds: float = 30.0,
) -> list[str]:
    """``prompts`` 각각에 대해 ``call_one(prompt, api_key)`` 를 스레드풀로
    동시에 실행한다 (vLLM의 연속 배칭과 달리 API 호출은 프롬프트마다
    별도의 HTTP 요청이라서 -- 참조 repo와 동일한 접근).

    실패하면 지수 백오프(+jitter)로 재시도하고, 매 재시도마다
    ``api_keys`` 를 무작위로 바꿔가며 시도한다. ``max_retries`` 번을 다
    써도 실패하면(참조 repo는 사실상 무한 재시도지만, 이 프로젝트는 실패를
    항상 안전한 기본값으로 낮추는 쪽을 선호한다 -- ``parse_bid``/
    ``parse_speak`` 와 같은 철학) 그 프롬프트 자리는 빈 문자열로 채운다.
    프롬프트 순서는 그대로 보존된다.
    """

    def _execute_one(prompt: ChatPrompt) -> str:
        backoff = 1.0
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            api_key = random.choice(api_keys)
            try:
                return call_one(prompt, api_key)
            except Exception as e:  # noqa: BLE001 -- 어떤 provider 예외든 재시도 대상
                last_error = e
                if attempt == max_retries:
                    break
                time.sleep(min(backoff, max_backoff_seconds) + random.uniform(0, 1))
                backoff = min(backoff * 2, max_backoff_seconds)
        # 재시도를 다 소진함 -- 예외를 전파하지 않고 빈 문자열로 낮춘다.
        del last_error
        return ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_execute_one, prompts))


class ApiEngineBase:
    """``contracts.Engine`` 을 만족하는 API 기반 엔진의 공통 뼈대.
    ``src/serving/registry.py`` 의 ``_BaseAdapter``/``_TutorAdapter`` 분리
    패턴과 같은 이유로 나눴다 -- profile 검증과 빈 입력 처리를 한 곳에
    모으고, 하위 클래스(``OpenRouterEngine``/``GeminiEngine``)는 실제 API
    호출 1건(``_call_one``)만 구현하면 된다.

    ``_TutorAdapter`` 를 감싸지 않고 이 클래스가 직접 ``registry.tutor`` 로
    쓰인다 -- 이미 ``contracts.Engine`` 을 만족하므로 어댑터가 필요 없다.
    """

    def __init__(
        self,
        model: str,
        api_key_env: str,
        *,
        max_workers: int = 20,
        max_retries: int = 5,
    ) -> None:
        # .env 파일(있다면)에서 API 키를 읽어 os.environ 에 채워 넣는다.
        # 이미 환경변수로 직접 설정돼 있으면(예: 배포 환경) 이 호출은
        # 아무것도 덮어쓰지 않는다(dotenv 기본 동작).
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()

        self.model = model
        self._api_key_env = api_key_env
        self._max_workers = max_workers
        self._max_retries = max_retries

    def run_batch(
        self,
        prompts: Sequence[ChatPrompt],
        profile: SamplingProfile,
        overrides: Mapping[str, Any] | None = None,
    ) -> list[str]:
        if profile not in _VALID_PROFILES:
            raise ValueError(
                f"tutor API 엔진에는 profile={profile!r} 을 쓸 수 없습니다. "
                f"허용: {sorted(_VALID_PROFILES)}"
            )
        prompts = list(prompts)
        if not prompts:
            return []

        gen_kwargs = profile_generation_kwargs(profile, overrides)
        api_keys = resolve_api_keys(self._api_key_env)

        def _call_one(prompt: ChatPrompt, api_key: str) -> str:
            return self._call_one(prompt, api_key, gen_kwargs)

        return run_prompts_concurrently(
            prompts,
            _call_one,
            api_keys,
            max_workers=self._max_workers,
            max_retries=self._max_retries,
        )

    def _call_one(self, prompt: ChatPrompt, api_key: str, gen_kwargs: Mapping[str, Any]) -> str:
        """실제 API 호출 1건. 하위 클래스가 구현한다. 실패하면 그대로
        예외를 던지면 된다 -- 재시도는 ``run_prompts_concurrently`` 가
        처리한다."""
        raise NotImplementedError

    def sleep(self) -> None:
        """``Classroom.rollout()`` 이 라운드마다 제네릭하게 호출한다
        (2026-08-16 도입, ``registry._BaseAdapter.sleep`` 참고). API 엔진은
        GPU에 상주하는 가중치 자체가 없으므로 재울 것도 없어서 아무 것도
        하지 않는다."""
        pass

    def wake_up(self) -> None:
        """``sleep()`` 과 대칭인 no-op."""
        pass

    def shutdown(self) -> None:
        """``EngineRegistry.shutdown()`` 이 제네릭하게 순회하며 호출한다
        (vLLM 엔진과 동일한 인터페이스를 맞추기 위함). API 엔진은 정리할
        GPU/프로세스 자원이 없으므로 아무 것도 하지 않는다."""
        pass
