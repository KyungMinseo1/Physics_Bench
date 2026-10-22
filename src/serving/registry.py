"""``contracts.Engine`` 규격에 맞춰 실제 vLLM 엔진을 감싸는 어댑터, 그리고
튜터용/학생용 엔진을 만들어주는 팩토리(생성 함수) 모음.

이 아래에서 쓰는 실제 서빙 엔진(``ParallelVLLMEngine``, vLLM 기반)은 그
동작 방식이 완전히 확정되어 있지 않은 부분이 있다. 이 모듈이 그런
불확실한 지점 세 가지를 전부 감싸안아서, 이 모듈보다 위쪽(호출하는 쪽)
코드는 그 불확실성을 신경 쓸 필요가 없게 만든다 (관련 근거는
student_simulator_plan.md §8.3, contracts.py 의 ``Engine`` 독스트링 참고).

1. vLLM 이 돌려주는 ``RequestOutput`` (vLLM 자체의 결과 객체) 에서 실제
   텍스트 문자열만 뽑아낸다 (``.outputs[0].text``). 다만 한 프롬프트당 여러 개의
   답을 생성하는(``n>1``) ``posttest``(사후검사) 의 경우는 ``.outputs`` 안에
   담긴 여러 결과를 프롬프트 순서를 유지한 채로 하나의 평평한 리스트로 펼친다.
2. ``profile``(어떤 종류의 호출인지를 나타내는 이름, 타입은
   ``SamplingProfile``) 을 실제 vLLM 이 요구하는 ``SamplingParams`` 객체로
   변환하는 일은 ``src/serving/sampling.py`` 에 맡긴다.
3. 대화 메시지 목록(chat message list) 형태를 보장하는 일. 이 저장소에서는
   프롬프트가 항상 ``ChatPrompt`` (즉 메시지 목록) 형태로만 들어오므로, 별도
   변환 없이 그대로 ``ParallelVLLMEngine.run_batch`` 에 전달하면 된다(문자열
   프롬프트로 바꿔줄 필요가 없다).

그리고 이 모듈의 진짜 핵심 책임은 두 가지다. 첫째, **튜터용/학생용 엔진
2개를 만들되 둘 다 같은 물리 GPU(device 0)를 가리키게 하는 팩토리**를
제공하는 것 (student_simulator_plan.md §8.3). 둘째, **JSON 형식을 강제하는
guided-decoding 이 실패했을 때 한 번만 재시도하는 로직**을 담당하는 것
(student_simulator_plan.md §8.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.contracts import ChatPrompt, Engine, SamplingProfile
from src.serving.parallel_vllm import ParallelVLLMEngine
from src.serving.sampling import (
    bid_is_valid,
    build_sampling_params,
    judge_is_valid,
    speak_is_valid,
)

__all__ = ["EngineRegistry", "build_registry"]

_DEFAULT_CONFIG_PATH = Path("config/models.yaml")


def _extract_text(request_output: Any) -> str:
    """vLLM 이 돌려주는 ``RequestOutput`` 객체에서 생성된 텍스트 1개만 뽑아낸다
    (``.outputs[0].text`` 위치에 들어있다).

    vLLM 이 정확히 어떤 형태의 객체를 반환하는지는 우리 쪽에서 완전히
    확정할 수 없는 부분이라, 그 불확실성을 다른 코드로 퍼지지 않게 여기
    한 곳에 모아둔 것이다 (contracts.Engine 독스트링, reference_notes
    §1.4의 두 번째 항목 참고).
    """
    return request_output.outputs[0].text


def _flatten_n(request_outputs: Sequence[Any]) -> list[str]:
    """한 프롬프트당 여러 개의 답을 요청하는(``n>1``) posttest(사후검사)
    프로파일의 경우, 각 결과의 ``.outputs`` 안에 담긴 여러 답변들을 프롬프트
    순서를 유지한 채로 하나의 평평한(중첩 없는) 리스트로 펼친다.

    결과 리스트의 길이는 ``len(request_outputs) * n`` 이 되고, 순서는
    ``[p0s0, p0s1, ..., p0s{n-1}, p1s0, ...]`` (프롬프트 0의 샘플 0, 1, ...,
    다음 프롬프트 1의 샘플 0, ... 순서) 이다. 이 형식은 contracts.Engine
    독스트링에 정의되어 있다.
    """
    flattened: list[str] = []
    for req_out in request_outputs:
        for candidate in req_out.outputs:
            flattened.append(candidate.text)
    return flattened


# --------------------------------------------------------------------------
# 역할(튜터/학생)별 Engine 어댑터
# --------------------------------------------------------------------------


class _BaseAdapter:
    """튜터 어댑터와 학생 어댑터가 공통으로 쓰는 기본 처리를 모아둔 클래스.
    빈 입력 배치 처리, 허용된 프로파일인지 검증, 실제 ParallelVLLMEngine
    으로의 위임을 담당한다."""

    _VALID_PROFILES: frozenset[str] = frozenset()
    _role_name = "engine"

    def __init__(self, engine: ParallelVLLMEngine) -> None:
        self._engine = engine

    def sleep(self) -> None:
        """``Classroom.rollout()`` 이 이번 라운드에 이 엔진이 필요 없을 때
        호출한다(GPU 메모리 절약, 2026-08-16 도입). ``self._engine`` 이
        ``ParallelVLLMEngine`` 이면 실제로 재우고, API 기반 엔진
        (``OpenRouterEngine``/``GeminiEngine``, GPU 자원을 안 씀)이면
        그쪽의 no-op ``sleep()`` 이 그대로 불린다."""
        self._engine.sleep()

    def wake_up(self) -> None:
        """``sleep()`` 의 반대. ``Classroom.rollout()`` 이 이 엔진을 쓰기
        직전에 호출한다."""
        self._engine.wake_up()

    def _check_profile(self, profile: SamplingProfile) -> None:
        if profile not in self._VALID_PROFILES:
            raise ValueError(
                f"{self._role_name} 엔진에는 profile={profile!r} 을 쓸 수 없습니다. "
                f"허용: {sorted(self._VALID_PROFILES)}"
            )

    def _generate(
        self,
        prompts: Sequence[ChatPrompt],
        profile: SamplingProfile,
        overrides: Mapping[str, Any] | None,
    ) -> list[str]:
        sp = build_sampling_params(profile, overrides)
        raw = self._engine.run_batch(list(prompts), sp)
        return [_extract_text(o) for o in raw]

    def _run_with_retry(
        self,
        prompts: list[ChatPrompt],
        profile: SamplingProfile,
        overrides: Mapping[str, Any] | None,
        is_valid: Any,
    ) -> list[str]:
        """받은 응답들 중에서 guided-JSON 형식을 지키지 못한 것으로 보이는
        응답만 골라내어, 그것들만 한 번 다시 호출(재시도)한다
        (student_simulator_plan.md §8.4절 규칙). ``_StudentAdapter``(bid/
        speak/judge)와 ``_JudgeAdapter``(judge가 별도 모델일 때)가 공유한다
        -- 둘 다 guided-JSON 프로파일에 똑같은 재시도 규칙을 적용해야 하므로
        여기 ``_BaseAdapter`` 로 올려 뒀다.

        "이 응답이 형식을 어긴 것으로 보인다"는 판정 자체는 여기서 하지 않고
        ``sampling.py`` 에 있는 순수 함수(``bid_is_valid`` / ``speak_is_valid``/
        ``judge_is_valid``)에 맡긴다. 이 메서드는 그 판정 결과를 받아 엔진을
        실제로 다시 호출하는 일(재시도 실행)만 담당한다. 이렇게 역할을 나눈
        이유는, §8.4절의 방침대로 파서가 있는 sampling.py 쪽은 엔진에
        접근하지 않는 순수한 코드로 유지하고, 실제로 엔진을 다시 불러야 하는
        재시도 처리는(엔진 접근이 필요하므로) 이쪽 registry.py 에 두기
        위해서다.
        """
        sp = build_sampling_params(profile, overrides)
        raw = self._engine.run_batch(prompts, sp)
        texts = [_extract_text(o) for o in raw]

        retry_idx = [i for i, t in enumerate(texts) if not is_valid(t)]
        if retry_idx:
            retry_prompts = [prompts[i] for i in retry_idx]
            retry_raw = self._engine.run_batch(retry_prompts, sp)
            retry_texts = [_extract_text(o) for o in retry_raw]
            for i, t in zip(retry_idx, retry_texts):
                texts[i] = t
        return texts


class _TutorAdapter(_BaseAdapter):
    """튜터(교사 역할) 엔진을 감싸는 어댑터. 튜터 발화용 ``tutor`` 프로파일과
    회귀 테스트용 ``regression`` 프로파일, 이 두 가지 호출만 허용한다."""

    _VALID_PROFILES = frozenset({"tutor", "regression"})
    _role_name = "tutor"

    def __init__(
        self, engine: ParallelVLLMEngine, extra_overrides: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(engine)
        #: 엔진 생성 시점에 고정되는 기본 오버라이드(예: ``config/models.yaml``
        #: 의 ``tutor.max_tokens`` -- ``enable_thinking`` 을 켠 reasoning
        #: 모델은 사고 과정 때문에 기본 512토큰으로 답까지 다 못 끝내고
        #: 잘리기 쉬워서, 필요하면 이 값을 늘려 잡는다). 매 ``run_batch``
        #: 호출의 ``overrides`` 에 항상 병합되고, 호출부가 명시적으로 같은
        #: 키를 넘기면 그게 우선한다.
        self._extra_overrides = dict(extra_overrides or {})

    def run_batch(
        self,
        prompts: Sequence[ChatPrompt],
        profile: SamplingProfile,
        overrides: Mapping[str, Any] | None = None,
    ) -> list[str]:
        self._check_profile(profile)
        prompts = list(prompts)
        if not prompts:
            return []
        merged_overrides = {**self._extra_overrides, **(overrides or {})}
        return self._generate(prompts, profile, merged_overrides)


class _StudentAdapter(_BaseAdapter):
    """학생 역할 엔진을 감싸는 어댑터. 입찰(bid)/발화(speak)/judge(발화가
    특정 오개념/KC를 겨냥했는지 판정하는 별도 심판 호출) 응답이 guided-JSON
    형식을 어겼다고 판단되면 한 번만 다시 호출(재시도)해보고, 그 결과를 그대로
    반환한다. judge는 GRPO로 학습시킬 튜터 정책과 완전히 분리돼야 하므로
    LoRA가 적용되지 않는 이 학생 엔진을 그대로 재사용한다(``build_registry``
    참고 -- 별도 엔진/GPU를 새로 띄우지 않는다).

    .. note::
       한 번 재시도를 해도 여전히 형식이 깨진(malformed) 응답일 수 있다. 이
       어댑터는 문자열 목록(``list[str]``, contracts.Engine 이 정한 반환 규약)만
       돌려주도록 되어 있어서, "파싱 최종 실패 시 쓸 기본값"(예:
       ``Bid(level="low", is_fallback=True)``)을 이 클래스 안에서 만들어 넣지는
       않는다. 그래서 **이 어댑터가 반환한 문자열을 실제로 사용하는 쪽에서는
       반드시** ``src/serving/sampling.py`` 의 ``parse_bid`` / ``parse_speak``
       함수로 다시 한번 읽어야 한다. 그래야 비로소 student_simulator_plan.md
       §8.4절이 정한 fallback 규칙(엔진 재시도 1회 → 그래도 실패하면 bid는
       low로, 사후검사는 오답으로 처리)이 완전히 적용된다. ``parse_bid``/
       ``parse_speak`` 는 형식이 깨진 입력이 들어와도 항상 안전하게 기본값으로
       처리하도록 설계돼 있으므로, "여기서의 엔진 재시도"와 "최종 사용처에서의
       관대한 파싱", 이 두 단계를 합쳐야 §8.4 규약 전체가 완성된다.
    """

    _VALID_PROFILES = frozenset(
        {"student_speak", "student_bid", "judge", "posttest", "regression"}
    )
    _role_name = "student"

    def __init__(self, engine: ParallelVLLMEngine, posttest_n_samples: int = 4) -> None:
        super().__init__(engine)
        self._posttest_n_samples = posttest_n_samples

    def run_batch(
        self,
        prompts: Sequence[ChatPrompt],
        profile: SamplingProfile,
        overrides: Mapping[str, Any] | None = None,
    ) -> list[str]:
        self._check_profile(profile)
        prompts = list(prompts)
        if not prompts:
            return []

        if profile == "posttest":
            return self._run_posttest(prompts, overrides)
        if profile == "student_bid":
            return self._run_with_retry(prompts, profile, overrides, bid_is_valid)
        if profile == "student_speak":
            return self._run_with_retry(prompts, profile, overrides, speak_is_valid)
        if profile == "judge":
            return self._run_with_retry(prompts, profile, overrides, judge_is_valid)
        # regression(회귀 테스트) 프로파일은 형식 제약이 없는 자유 텍스트라 재시도가 필요 없다.
        return self._generate(prompts, profile, overrides)

    def _run_posttest(
        self, prompts: list[ChatPrompt], overrides: Mapping[str, Any] | None
    ) -> list[str]:
        merged: dict[str, Any] = {"n": self._posttest_n_samples}
        if overrides:
            merged.update(overrides)
        sp = build_sampling_params("posttest", merged)
        raw = self._engine.run_batch(prompts, sp)
        return _flatten_n(raw)


class _JudgeAdapter(_BaseAdapter):
    """judge(발화 하나가 특정 오개념/KC를 얼마나 정확히 겨냥했는지 판정하는
    별도 심판 LLM 호출) 전용 어댑터. ``judge`` 프로파일과 회귀 테스트용
    ``regression`` 프로파일만 허용한다.

    ``config/models.yaml`` 의 ``judge.model_path`` 가 비어 있으면(기본값)
    judge는 여전히 ``_StudentAdapter`` (학생 엔진)를 그대로 재사용한다 --
    ``_StudentAdapter`` 도 이미 ``judge`` 프로파일을 처리하므로 이 클래스를
    쓸 필요가 없다. 이 어댑터는 ``judge.model_path`` 가 설정돼서 judge가
    **진짜 별도 모델/엔진**일 때만 쓰인다(design_v2.md "미결 사항": "Judge
    모델 — 학생 LLM과 분리된 모델을 쓸 것(같은 모델이면 자기 선호 편향)").
    별도 어댑터로 분리한 이유는 순전히 안전장치다 -- profile 검증을 좁게
    걸어서, student_speak/student_bid/posttest 프로파일 호출이 실수로 이
    judge 전용 엔진으로 새어 들어가는 걸 원천 차단한다."""

    _VALID_PROFILES = frozenset({"judge", "regression"})
    _role_name = "judge"

    def run_batch(
        self,
        prompts: Sequence[ChatPrompt],
        profile: SamplingProfile,
        overrides: Mapping[str, Any] | None = None,
    ) -> list[str]:
        self._check_profile(profile)
        prompts = list(prompts)
        if not prompts:
            return []
        if profile == "judge":
            return self._run_with_retry(prompts, profile, overrides, judge_is_valid)
        # regression(회귀 테스트) 프로파일은 형식 제약이 없는 자유 텍스트라 재시도가 필요 없다.
        return self._generate(prompts, profile, overrides)


# --------------------------------------------------------------------------
# 레지스트리(엔진들을 담는 컨테이너) + 그것을 만드는 팩토리 함수
# --------------------------------------------------------------------------


@dataclass
class EngineRegistry:
    """튜터용/학생용/judge용 엔진을 함께 담아두는 컨테이너. 셋 다
    ``contracts.Engine`` 규격을 만족한다.

    사용하는 쪽에서는 ``registry.tutor`` / ``registry.student`` /
    ``registry.judge`` 처럼 속성으로 접근하면 된다.
    """

    tutor: Engine
    student: Engine
    #: judge(발화가 특정 오개념/KC를 겨냥했는지 판정하는 별도 심판 LLM
    #: 호출) 전용 엔진. ``config/models.yaml`` 의 ``judge.model_path`` 가
    #: 비어 있으면(기본값) 이 필드는 ``student`` 와 **같은 객체**다(별도
    #: GPU/메모리를 추가로 안 씀, 기존 동작 그대로) -- design_v2.md
    #: "미결 사항"의 "Judge 모델 — 학생 LLM과 분리된 모델을 쓸 것"이 아직
    #: 충족 안 된 기본 상태다. ``model_path`` 를 채우면 별도 vLLM 인스턴스로
    #: 분리된다.
    judge: Engine
    #: 가공되지 않은(어댑터로 감싸지 않은) 원본 엔진 인스턴스들을 그대로
    #: 들고 있는 필드다(길이는 2 또는 3 -- judge가 student를 재사용하면
    #: 중복 등록을 피해 2개). 프로세스를 정상적으로 종료(shutdown)하거나
    #: 저수준 디버깅을 할 때만 사용한다. 평소 롤아웃(시뮬레이션 진행)
    #: 코드는 이 필드를 직접 건드리지 말고 ``tutor``/``student``/``judge``
    #: 세 어댑터만 통해서 엔진을 써야 한다. 학생 쪽은 항상
    #: ``ParallelVLLMEngine`` 이지만, 튜터 쪽은 ``config/models.yaml`` 의
    #: ``tutor.provider`` 설정에 따라 ``ParallelVLLMEngine`` 이 아니라
    #: ``OpenRouterEngine``/``GeminiEngine``
    #: (``src/serving/{openrouter,gemini}.py`` -- 시뮬레이션 진단 전용)일 수
    #: 있으므로 타입을 ``Engine`` 으로 넓혀 둔다. 이 API 기반 엔진들도
    #: ``shutdown()`` 을 (no-op으로) 구현하므로 아래 반복문은 그대로 동작한다.
    _raw_engines: tuple[Engine, ...] = field(repr=False)

    def shutdown(self) -> None:
        """엔진들이 띄운 워커 프로세스를 모두 종료한다."""
        for engine in self._raw_engines:
            engine.shutdown()


def _load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _tutor_engine_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    tutor_cfg = cfg["tutor"]
    vllm_cfg = tutor_cfg["vllm"]
    lora_cfg = tutor_cfg.get("lora") or {}
    use_lora = bool(lora_cfg.get("enable", False))
    lora_path = lora_cfg.get("path")
    # LoRA(대형 모델을 통째로 재학습하지 않고 적은 파라미터만 추가 학습해
    # 파인튜닝하는 경량 기법)가 config 상으로는 켜져 있는데(lora.enable=true)
    # 정작 학습된 어댑터 파일 경로(path)가 아직 없는 경우가 있다. 이는 아직
    # LoRA 어댑터를 학습시키기 전인 프로젝트 초기 단계(M0)에서 흔히 벌어지는
    # 상황이다. 이럴 때는 엔진 자체는 enable_lora=True 로 "LoRA 를 받아들일 수
    # 있는 상태"로 띄우되, 실제로 어댑터를 붙이는 LoRARequest 는 만들지 않아서
    # 결과적으로 LoRA 를 적용하지 않은 기본(base) 모델처럼 동작하게 한다. 이
    # fallback(대체) 동작 방식은 reference_notes 문서에서 확인된 사실이 아니라,
    # 이번에 이 코드를 이식하면서 우리가 새로 내린 설계 판단이다 (최종 보고서의
    # "확인 필요" 항목 참조).
    return dict(
        model_path=tutor_cfg["model_path"],
        gpus_per_instance=vllm_cfg.get("number_of_gpus_per_instance", 1),
        n_instances=vllm_cfg.get("n_instances", None),
        max_number_of_instances=vllm_cfg.get("max_number_of_instances", -1),
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.40),
        max_model_len=vllm_cfg.get("max_model_len", 12000),
        max_num_seqs=vllm_cfg.get("max_num_seqs", 512),
        enable_sleep_mode=vllm_cfg.get("enable_sleep_mode", False),
        enforce_eager=vllm_cfg.get("enforce_eager", False),
        use_lora=use_lora,
        lora_path=lora_path,
        max_lora_rank=lora_cfg.get("rank", 64),
        enable_prefix_caching=vllm_cfg.get("enable_prefix_caching", True),
        kv_cache_dtype=vllm_cfg.get("kv_cache_dtype"),
        quantization=vllm_cfg.get("quantization"),
        use_v0=vllm_cfg.get("use_v0", False),
        enable_thinking=vllm_cfg.get("enable_thinking", False),
        use_awq=vllm_cfg.get("use_awq", False),
        bits_and_bytes=vllm_cfg.get("bits_and_bytes", False),
        gpu_ids=vllm_cfg.get("gpu_ids"),
    )


def _student_engine_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    student_cfg = cfg["student"]
    vllm_cfg = student_cfg["vllm"]
    # LoRA(경량 파인튜닝 기법)는 튜터 전용 기능이다. config 파일의 student
    # 섹션에는 애초에 lora 라는 키 자체가 존재하지 않으므로, 아래에서
    # use_lora/lora_path 는 절대 채워지지 않는다(항상 False/None). 즉 "학생
    # 엔진에 실수로 LoRA 가 새어 들어가는 일"이 코드 로직이 아니라 config
    # 파일의 구조 자체로 원천 차단되어 있다.
    return dict(
        model_path=student_cfg["model_path"],
        gpus_per_instance=vllm_cfg.get("number_of_gpus_per_instance", 1),
        n_instances=vllm_cfg.get("n_instances", None),
        max_number_of_instances=vllm_cfg.get("max_number_of_instances", -1),
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.45),
        max_model_len=vllm_cfg.get("max_model_len", 12000),
        max_num_seqs=vllm_cfg.get("max_num_seqs", 512),
        enforce_eager=vllm_cfg.get("enforce_eager", False),
        enable_sleep_mode=vllm_cfg.get("enable_sleep_mode", False),
        use_lora=False,
        lora_path=None,
        enable_prefix_caching=vllm_cfg.get("enable_prefix_caching", True),
        kv_cache_dtype=vllm_cfg.get("kv_cache_dtype"),
        quantization=vllm_cfg.get("quantization"),
        use_v0=vllm_cfg.get("use_v0", False),
        enable_thinking=vllm_cfg.get("enable_thinking", False),
        use_awq=vllm_cfg.get("use_awq", False),
        bits_and_bytes=vllm_cfg.get("bits_and_bytes", False),
        gpu_ids=vllm_cfg.get("gpu_ids"),
    )


def _judge_engine_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    judge_cfg = cfg["judge"]
    vllm_cfg = judge_cfg["vllm"]
    return dict(
        model_path=judge_cfg["model_path"],
        gpus_per_instance=vllm_cfg.get("number_of_gpus_per_instance", 1),
        n_instances=vllm_cfg.get("n_instances", None),
        max_number_of_instances=vllm_cfg.get("max_number_of_instances", -1),
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.30),
        max_model_len=vllm_cfg.get("max_model_len", 12000),
        max_num_seqs=vllm_cfg.get("max_num_seqs", 512),
        enable_sleep_mode=vllm_cfg.get("enable_sleep_mode", False),
        enforce_eager=vllm_cfg.get("enforce_eager", False),
        use_lora=False,
        lora_path=None,
        enable_prefix_caching=vllm_cfg.get("enable_prefix_caching", True),
        kv_cache_dtype=vllm_cfg.get("kv_cache_dtype"),
        quantization=vllm_cfg.get("quantization"),
        use_v0=vllm_cfg.get("use_v0", False),
        enable_thinking=vllm_cfg.get("enable_thinking", False),
        use_awq=vllm_cfg.get("use_awq", False),
        bits_and_bytes=vllm_cfg.get("bits_and_bytes", False),
        gpu_ids=vllm_cfg.get("gpu_ids"),
    )


def _build_judge_engine(
    cfg: Mapping[str, Any], student_engine: Engine
) -> tuple[Engine, Engine | None]:
    """``(registry.judge 에 넣을 엔진, _raw_engines 에 추가로 등록할 원본
    엔진 또는 None)`` 을 만든다.

    ``cfg["judge"]["model_path"]`` 가 비어 있으면(기본값, 기존 설정 파일에
    ``judge:`` 블록 자체가 없어도 동일하게 동작한다) student 엔진을 그대로
    돌려준다 -- 지금까지의 동작(judge가 student 엔진을 재사용)과 완전히
    같고, 두 번째 값은 None이라 ``_raw_engines`` 에 중복으로 잡히지 않는다
    (student 엔진은 이미 별도로 등록되므로 여기서 또 넣으면 shutdown()이
    두 번 불린다). 값이 있으면 vLLM 인스턴스를 하나 더 띄운다 --
    design_v2.md "미결 사항"의 "Judge는 학생과 분리된 모델을 쓸 것"에
    대응한다. GPU 메모리를 추가로 쓰므로, 필요하면 tutor/student 쪽
    ``gpu_memory_utilization`` 도 같이 낮춰야 할 수 있다.

    openrouter/gemini 같은 API 기반 judge는 아직 지원하지 않는다 -- judge
    프로파일은 guided-JSON(``JudgeSchema``) 구조화 출력이 필요한데,
    ``src/serving/api_common.py`` 의 ``ApiEngineBase`` 는 (튜터 전용으로
    설계돼) 그 기능이 없다.
    """
    judge_cfg = cfg.get("judge") or {}
    model_path = judge_cfg.get("model_path")
    if not model_path:
        return student_engine, None
    raw = ParallelVLLMEngine(**_judge_engine_kwargs(cfg))
    return _JudgeAdapter(raw), raw


def _build_tutor_engine(cfg: Mapping[str, Any]) -> tuple[Engine, Engine]:
    """``(registry.tutor 에 넣을 엔진, _raw_engines/shutdown용 원본 엔진)``
    튜플을 만든다. ``cfg["tutor"]["provider"]``(기본값 ``"vllm"``, 즉 기존
    설정 파일은 이 필드가 없어도 지금까지와 동일하게 동작한다)로 분기한다.

    ``provider: vllm`` 이면 두 값이 다르다 -- ``_TutorAdapter`` 로 감싼 것과
    원본 ``ParallelVLLMEngine`` (``shutdown()`` 은 원본에만 있다). 그 외
    (``openrouter``/``gemini``, ``src/serving/{openrouter,gemini}.py`` --
    **시뮬레이션 진단 전용**, GRPO/PPO로 학습시키는 실제 튜터 정책은 여전히
    vLLM만 쓴다)는 두 값이 같은 객체다 -- 이미 ``contracts.Engine`` 을 직접
    만족하므로(``src/serving/api_common.py`` 의 ``ApiEngineBase``) 별도
    어댑터가 필요 없다.
    """
    provider = cfg["tutor"].get("provider", "vllm")
    if provider == "vllm":
        raw = ParallelVLLMEngine(**_tutor_engine_kwargs(cfg))
        # reasoning으로 토큰을 더 쓰는 모델(enable_thinking=true)을 위한
        # tutor 프로파일 max_tokens 오버라이드. null/미설정이면 sampling.py의
        # 기본값(512)을 그대로 쓴다.
        max_tokens = cfg["tutor"].get("max_tokens")
        extra_overrides = {"max_tokens": max_tokens} if max_tokens is not None else None
        return _TutorAdapter(raw, extra_overrides=extra_overrides), raw
    if provider == "openrouter":
        from src.serving.openrouter import OpenRouterEngine  # noqa: PLC0415

        engine = OpenRouterEngine(**cfg["tutor"]["openrouter"])
        return engine, engine
    if provider == "gemini":
        from src.serving.gemini import GeminiEngine  # noqa: PLC0415

        engine = GeminiEngine(**cfg["tutor"]["gemini"])
        return engine, engine
    raise ValueError(
        f"config/models.yaml 의 tutor.provider={provider!r} 를 알 수 없습니다. "
        "vllm/openrouter/gemini 중 하나여야 합니다."
    )


def build_registry(config_path: str | Path = _DEFAULT_CONFIG_PATH) -> EngineRegistry:
    """``config/models.yaml`` 설정 파일을 읽어서 튜터용·학생용·judge용
    엔진을 만들고, 셋을 :class:`EngineRegistry` 로 감싸서 반환한다. judge는
    ``judge.model_path`` 가 비어 있으면(기본값) 별도 엔진을 안 띄우고
    student 엔진을 그대로 재사용한다(``_build_judge_engine`` 참고).

    .. warning::
       **``ParallelVLLMEngine`` 인스턴스는(최대 3개 -- 튜터/학생/judge)
       반드시 이 함수 안에서, 하나씩 순서대로(동시가 아니라 순차적으로)
       생성되어야 한다.** 그 이유는 다음과 같다. ``ParallelVLLMEngine.__init__``
       은 각 엔진의 워커 프로세스를 띄울 때 부모 프로세스가 갖고 있는
       ``CUDA_VISIBLE_DEVICES``(어떤 GPU를 쓸지 지정하는 환경변수)의 값을
       전역으로(그 프로세스 전체에 영향을 주는 방식으로) 덮어써 버린다
       (reference_notes §1.3 참고). 만약 엔진 인스턴스 여러 개를 동시에
       (스레드나 asyncio, 별도 프로세스 등으로 병렬로) 생성하면, 이
       환경변수 값을 서로 먼저 쓰려고 경합하게 되어 하나 이상이 잘못된
       GPU를 바라보게 될 수 있다. **그러므로 이 함수를 스레드풀이나
       asyncio.gather 같은 병렬 실행 수단으로 동시에 여러 번 호출해서는
       안 된다.** 바로 아래에서 코드가 순서대로(한 줄씩 차례로) 실행되게
       작성해둔 것 자체가 이 순서를 강제하는 장치다. (``tutor.provider``
       가 vllm이 아니면 튜터 쪽은, judge가 student를 재사용하면 judge
       쪽은 GPU/``CUDA_VISIBLE_DEVICES`` 를 전혀 건드리지 않으므로 이
       경고와 무관하다 -- 그래도 이 함수 자체를 병렬 호출하지 않는 규칙은
       여전히 지킨다.)
    """
    cfg = _load_config(config_path)

    # 순서가 중요하다: 튜터 -> 학생 -> judge 순서로 만든다. (위 경고 참조)
    tutor_engine, tutor_raw = _build_tutor_engine(cfg)
    tutor_raw.sleep()  # 튜터 엔진은 시뮬레이션 시작 전까지는 안 쓰므로 바로 재우기
    student_raw = ParallelVLLMEngine(**_student_engine_kwargs(cfg))
    posttest_n = cfg.get("posttest", {}).get("n_samples", 4)
    student_engine = _StudentAdapter(student_raw, posttest_n_samples=posttest_n)
    student_raw.sleep()  # 학생 엔진도 시뮬레이션 시작 전까지는 안 쓰므로 바로 재우기

    judge_engine, judge_raw = _build_judge_engine(cfg, student_engine)
    if judge_raw is not None:
        judge_raw.sleep()  # judge 엔진도 시뮬레이션 시작 전까지는 안 쓰므로 바로 재우기

    raw_engines = [tutor_raw, student_raw]
    if judge_raw is not None:
        raw_engines.append(judge_raw)

    return EngineRegistry(
        tutor=tutor_engine,
        student=student_engine,
        judge=judge_engine,
        _raw_engines=tuple(raw_engines),
    )

if __name__ == "__main__":
    # 이 모듈을 단독으로 실행하면, config/models.yaml 을 읽어서 튜터/학생 엔진을
    # 띄우고, 그 엔진들을 shutdown() 하고 종료한다. (실제 시뮬레이션은 하지 않는다.)
    registry = build_registry()
    registry.shutdown()