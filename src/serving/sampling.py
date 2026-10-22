"""이 파일에는 (1) 호출 종류(샘플링 프로파일)별 생성 파라미터 정의, (2) 모델
응답이 지켜야 할 JSON 형식을 강제하는 "guided-JSON" 스키마, (3) 응답이 형식을
안 지켰을 때도 최대한 봐주며 읽어내는 관대한 파서가 들어 있다.

``contracts.SamplingProfile`` 에 정의된 6가지 호출 종류(``tutor``=튜터 발화,
``student_speak``=학생 발화, ``student_bid``=학생이 발언권을 얻으려 입찰하는 것,
``judge``=발화 하나가 특정 오개념/KC를 얼마나 정확히 겨냥했는지 판정하는 별도
심판 호출, ``posttest``=사후검사, ``regression``=회귀 테스트용)마다 실제로 어떤
파라미터(temperature, max_tokens 등)를 쓸지는 이 파일이 정한다. ``tutor``/
``student_speak``/``student_bid``/``posttest``/``regression`` 다섯 종류와 각각의
규칙은 student_simulator_plan.md 8장("서빙 구성", 특히 8.4절 "구조화 출력"과
8.5절 "재현성")에 정의되어 있고, 아래 코드 곳곳의 §8.4/§8.5 같은 표기는 그
문서의 해당 절을 가리킨다. ``judge``는 그 문서에 없는, 방향 그래프 기반
지식 갱신 설계(2026-08-15)에서 새로 추가된 프로파일이다.

이 모듈 안의 함수들은 전부 **엔진(실제로 모델을 호출하는 객체)에 직접 접근하지
않는 순수 함수**다. vLLM(GPU에서 모델을 서빙하는 라이브러리) 관련 import 는
:func:`build_sampling_params` 함수 안에서만, 그 함수가 실제로 호출될 때 이뤄진다
(지연 import). 이렇게 해두면 vLLM 이 설치되어 있지 않은 환경(GPU 없이 테스트용
MockEngine 만 쓰는 경우 등)에서도, 이 파일에 있는 스키마 정의나 파서 함수들은
아무 문제 없이 import 하고 호출할 수 있다.

"응답이 형식을 안 지켰을 때 엔진을 다시 호출해서 재시도한다"는 로직(재시도
오케스트레이션)은 이 파일의 책임이 아니다. ``parse_bid`` / ``parse_speak`` 는
**주어진 텍스트 한 번**에 대해서만 최선을 다해 해석해보고, 실패하면 곧바로
정해진 기본값(fallback)을 반환하는 순수 함수일 뿐이다. "실패했으니 엔진을 한
번 더 호출해서 다시 받아보자"는 판단과 실제 재호출은 ``src/serving/registry.py``
의 어댑터 코드가 담당한다. 그 어댑터는 이 파일의 :func:`bid_is_valid` /
:func:`speak_is_valid` 함수로 "이 응답이 재시도가 필요할 만큼 깨졌는지"를
판정한다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from src.contracts import Bid, KCUpdate, SamplingProfile, SelfReport

if TYPE_CHECKING:  # pragma: no cover - 타입 힌트 전용
    from vllm import SamplingParams

__all__ = [
    "BidSchema",
    "SelfReportSchema",
    "SpeakSchema",
    "JudgeSchema",
    "JudgeProposal",
    "profile_generation_kwargs",
    "build_sampling_params",
    "bid_is_valid",
    "speak_is_valid",
    "judge_is_valid",
    "parse_bid",
    "parse_speak",
    "parse_judge",
]


# --------------------------------------------------------------------------
# guided-JSON 스키마: 모델이 반드시 이 형식의 JSON 으로만 답하도록 vLLM 에
# 강제시키는 데 쓰는 pydantic 클래스들이다 (student_simulator_plan.md §8.4).
# --------------------------------------------------------------------------


class KCUpdateItemSchema(BaseModel):
    """``BidSchema.kc_updates`` 의 원소 하나. ``kc_id`` 는 그 학생의 bid
    프롬프트에 ``[Checking your own understanding]`` 블록으로 표시된 태그
    중 하나여야 하지만, 이 스키마 자체는 그걸 강제하지 않는다 -- guided
    JSON 스키마는 학생마다 달라지는 kc_id 목록을 알 수 없으므로(프로파일당
    스키마 하나), 실제로 그 학생이 갖고 있는 mastered가 아닌 KC인지는
    ``apply_knowledge_update`` 가 사후에 검증한다."""

    model_config = ConfigDict(extra="ignore")

    kc_id: str
    resolved: bool


class BidSchema(BaseModel):
    """학생의 "입찰(bid)" 응답이 지켜야 할 JSON 형식. 여기서 쓰는 값(high/medium/low)의
    범위는 ``contracts.BidLevel`` 에 정의된 값 범위와 반드시 일치해야 한다.

    ``intent`` 는 PEARL 논문의 z^t(latent behavioral intent)에 대응하는
    필드로, 새 LLM 콜을 추가하지 않고 기존 bid 응답에 실어 보낸다
    (``contracts.StudentIntent`` 참고). 기본값 ``"neutral"`` 은 이 필드가
    없던 기존 손으로 쓴 JSON(테스트 등)과의 호환을 위한 것이다.

    ``kc_review``/``kc_updates`` 는 이번 라운드에 이 학생이 자기 knowledge
    상태를 스스로 재검토한 결과다(2026-08-14 재설계 — self_report 대신 이
    필드가 갱신 신호원이 된다, ``src.student.knowledge_update`` 참고).
    ``kc_review`` 는 JSON 키 순서상 ``kc_updates`` 보다 앞에 둬서, guided
    decoding이 최종 배열을 커밋하기 전에 먼저 이 자유 문장 필드로 "생각을
    정리"하도록 유도한다(구조화 출력 안에서 reasoning-then-answer 순서를
    강제하는 흔한 패턴).

    (2026-08-14 필드 순서 재조정) ``intent`` 를 ``kc_review``/``kc_updates``
    **뒤**로 옮겼다. guided decoding은 vLLM에 넘기는 이 클래스의
    ``model_json_schema()`` 순서 그대로 토큰 단위로 키를 강제하므로(프롬프트
    예시 텍스트의 순서와는 무관하다 -- ``student_bid.j2`` 도 이 순서에 맞춰
    같이 고쳤다), 원래 순서(``intent`` 가 먼저)에서는 모델이 kc_review/
    kc_updates로 자기 이해 상태를 실제로 되짚어보기도 전에 intent부터
    커밋해야 했다. 200-에피소드 스윕에서 misconceived kc_updates 제안 12건
    중 12건, absent kc_updates 제안 287건 중 205건이 같은 응답 안에서
    intent="reinforce"와 함께 나와 자기모순으로 처리돼 버려졌는데(``src.
    student.knowledge_update`` 의 ``*_INTENT_MULTIPLIER`` 참고), 이게
    진짜 자기모순이 아니라 순서상 intent를 먼저 정해야 해서 생긴 인공물일
    가능성이 있어 순서를 바꿔서 재검증해보는 것이다 -- kc_review/kc_updates로
    먼저 자기 이해를 정리한 다음에 intent를 고르게 하면, intent가 그
    자기평가와 더 일관되게 나올 것이라는 가설이다.
    """

    model_config = ConfigDict(extra="ignore")

    bid: Literal["high", "medium", "low"]
    reason: str
    kc_review: str = ""
    kc_updates: list[KCUpdateItemSchema] = Field(default_factory=list)
    intent: Literal["reinforce", "waver", "neutral"] = "neutral"


class SelfReportSchema(BaseModel):
    """학생 발화에 함께 붙는 "자기 보고(self-report)"의 JSON 형식. 학생이 스스로
    "내 생각이 바뀌었다" 같은 걸 보고하는 부가 정보로, 필드는 ``contracts.SelfReport``
    와 1대1로 대응한다."""

    model_config = ConfigDict(extra="ignore")

    changed_mind_about: str | None = None
    addressing: str | None = None


class SpeakSchema(BaseModel):
    """학생 발화(student_speak) 응답이 지켜야 할 JSON 형식.

    이 스키마가 밖으로 내보내는 JSON 키 이름은 ``_self_report`` 여야 한다.
    이는 contracts.py 의 ``SamplingProfile`` 독스트링(및 student_simulator_plan.md
    §6.2절 "STUDENT_SPEAK")이 정한 응답 형식 규칙이다. 그런데 pydantic v2 는
    필드 이름이 밑줄(_)로 시작하면 "라이브러리 내부용 필드(private attribute)"로
    취급해서 스키마나 JSON 변환 결과에서 아예 빼버린다. 그래서 밑줄로 시작하지
    않는 ``self_report`` 라는 파이썬 필드명을 쓰되, ``Field(alias=...)`` 로
    "JSON 으로 나갈 때는 ``_self_report`` 라는 이름을 써라"라고 별칭을 지정해준
    것이다. 이건 pydantic 자체의 제약 때문에 필요한 구현상의 우회일 뿐이고,
    참조 문서(reference_notes)와는 관련이 없다.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    utterance: str
    self_report: SelfReportSchema = Field(alias="_self_report", default_factory=SelfReportSchema)


class JudgeEdgeItemSchema(BaseModel):
    """judge(발화 하나가 특정 오개념/KC를 얼마나 정확히 겨냥했는지 판정하는
    별도 심판 LLM 호출) 응답의 ``edges`` 배열 원소 하나. ``target``/``kc_id``/
    ``misconception_id`` 가 실제 이 에피소드의 student_id/problem.kcs/
    problem.applicable_misconceptions 안에 있는 값인지는 이 스키마가
    강제하지 않는다(judge 프로파일 전체에 스키마 하나뿐이라 에피소드마다
    달라지는 목록을 guided JSON 스키마에 담을 수 없다) -- 실제 검증은
    ``Episode.record_judge_output`` 이 사후에 방어적으로 한다."""

    model_config = ConfigDict(extra="ignore")

    target: str
    kc_id: str
    misconception_id: str | None = None
    relation: Literal["targeted", "generic"]


class JudgeSchema(BaseModel):
    """judge 응답이 지켜야 할 JSON 형식."""

    model_config = ConfigDict(extra="ignore")

    edges: list[JudgeEdgeItemSchema] = Field(default_factory=list)


@dataclass(frozen=True)
class JudgeProposal:
    """judge 응답 파싱 결과 하나. ``Episode.record_judge_output`` 이 이걸
    검증해서 ``contracts.ChallengeEdge`` 로 승격시키기 전까지는 순수하게
    "judge의 제안"일 뿐이다."""

    target: str
    kc_id: str
    misconception_id: str | None
    relation: Literal["targeted", "generic"]


# --------------------------------------------------------------------------
# 프로파일 기본 파라미터
# --------------------------------------------------------------------------

#: max_tokens(모델이 한 번에 생성할 수 있는 최대 토큰 수) 값은
#: student_simulator_plan.md 에도 reference_notes 에도 정확한 수치가 나와 있지
#: 않다. 그래서 아래 값들은 근거 있는 계산값이 아니라 **잠정적인 자리표시자
#: (placeholder)일 뿐임을 분명히 밝혀둔다.** 실제 프로젝트 초기 단계(M0)에서
#: 학생 발화·입찰(bid)·사후검사 응답이 실제로 얼마나 긴지 확인한 뒤 조정해야
#: 한다 (자세한 절차는 docs/m0_checklist.md 참조).
_PLACEHOLDER_MAX_TOKENS: dict[SamplingProfile, int] = {
    "tutor": 512,
    "student_speak": 512,
    "student_bid": 512,
    "judge": 512,
    "regression": 512,
}

_PROFILE_DEFAULTS: dict[SamplingProfile, dict[str, Any]] = {
    "tutor": {"temperature": 1.0, "max_tokens": _PLACEHOLDER_MAX_TOKENS["tutor"]},
    "student_speak": {"temperature": 0.6, "max_tokens": _PLACEHOLDER_MAX_TOKENS["student_speak"]},
    "student_bid": {"temperature": 0.6, "max_tokens": _PLACEHOLDER_MAX_TOKENS["student_bid"]},
    # judge는 분류(classification) 과제다 -- 학생/튜터 발화처럼 다양한 표현이
    # 나올 필요가 없고, 오히려 같은 입력에 판정이 왔다갔다 하면 안 되므로
    # temperature=0.0(결정적)으로 고정한다.
    "judge": {"temperature": 0.0, "max_tokens": _PLACEHOLDER_MAX_TOKENS["judge"]},
    # temperature=0(greedy)이면 n=4가 전부 동일한 텍스트를 낸다 — 그러면
    # grader.aggregate_majority()가 4개 샘플로 다수결/평균을 내는 것(§2.6
    # "측정 노이즈 관리")이 아무 의미가 없어지고, 최신 vLLM은 이 조합
    # (temperature=0 + n>1) 자체를 에러로 막는다. 그래서 실제로 다른 샘플
    # 4개를 얻도록 student_speak/student_bid와 같은 temperature=0.6을 쓴다.
    "posttest": {"temperature": 0.6, "n": 4, "max_tokens": _PLACEHOLDER_MAX_TOKENS["posttest"]},
    # student_simulator_plan.md §8.5("재현성")에 따르면 regression(회귀 테스트용)
    # 프로파일은 temperature=0(무작위성 없이 항상 가장 확률 높은 답만 고르는 설정)이
    # 핵심이다. contracts.py 의 독스트링은 이에 더해 "max_num_seqs=1"(동시에 처리하는
    # 요청 수를 1개로 제한)도 언급하는데, 그건 여기 SamplingParams(생성 파라미터)가
    # 아니라 **엔진을 만들 때 넘기는 생성자 인자**(ParallelVLLMEngine.max_num_seqs) 쪽
    # 설정이다. 즉 동시에 여러 요청을 처리하지 못하게 만드는 제약이라는 뜻이다.
    # 다만 이 regression 프로파일만을 위해 동시성을 1로 낮춘 별도의 엔진 인스턴스를
    # 이번 작업 범위에서는 아직 만들지 않았다. registry.py 쪽 주석과 최종 보고서의
    # "확인 필요" 항목을 참고할 것.
    "regression": {"temperature": 0.0, "max_tokens": _PLACEHOLDER_MAX_TOKENS["regression"]},
}

#: 아래 딕셔너리에는 guided decoding(JSON 형식 강제) 스키마가 적용되는
#: 프로파일만 등록한다. tutor(튜터 발화)/posttest(사후검사)/regression(회귀
#: 테스트)은 형식 제약이 없는 자유 텍스트로 응답받는다. posttest 응답을
#: 채점하는 로직은 이 파일이 아니라 src/domain/base.py 의 Domain.grade 가 담당한다.
_GUIDED_SCHEMAS: dict[SamplingProfile, dict[str, Any]] = {
    "student_bid": BidSchema.model_json_schema(),
    "student_speak": SpeakSchema.model_json_schema(),
    "judge": JudgeSchema.model_json_schema(),
}


def profile_generation_kwargs(
    profile: SamplingProfile, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """프로파일의 기본 생성 파라미터(``temperature``/``max_tokens``/필요하면
    ``n``)에 ``overrides`` 를 얹은 순수 dict 를 돌려준다. vLLM 을 전혀 import
    하지 않는다 -- ``build_sampling_params`` (vLLM 경로)와 vLLM 에 의존하지
    않는 API 기반 엔진(``src.serving.api_common`` 참고, OpenRouter/Gemini 등
    시뮬레이션 전용 진단 경로) 둘 다 같은 기본값을 이 함수 하나로 공유한다.

    guided-JSON 스키마는 여기 포함되지 않는다 -- vLLM 전용 개념이라
    :func:`build_sampling_params` 에서만 별도로 얹는다.
    """
    spec = dict(_PROFILE_DEFAULTS[profile])
    if overrides:
        spec.update(overrides)
    return spec


def build_sampling_params(
    profile: SamplingProfile, overrides: Mapping[str, Any] | None = None
) -> "SamplingParams":
    """프로파일 이름을 받아서, vLLM 이 실제로 요구하는 ``SamplingParams`` 객체
    (생성 파라미터를 담는 vLLM 자체 타입)로 변환해 돌려준다.

    ``overrides`` 인자로 ``max_tokens`` 같은 값은 기본값을 덮어쓸 수 있다.
    다만 guided-JSON 스키마(응답이 지켜야 할 JSON 형식)만큼은 **항상 그
    프로파일에 정해진 값을 마지막에 다시 강제로 적용한다**(overrides 를
    먼저 병합한 다음, 그 위에 스키마를 다시 얹는 순서로 처리한다). 이렇게
    하는 이유는, 호출하는 쪽이 실수로 overrides 에 다른 스키마 값을 흘려
    넣더라도 "이 프로파일은 반드시 이 JSON 형식을 지켜야 한다"는 규칙이
    깨지지 않도록 보장하기 위해서다 (contracts.Engine.run_batch 의 독스트링에
    "구조화 출력 스키마는 여기서 바꾸지 말 것"이라고 명시되어 있다).

    vLLM 라이브러리는 이 함수가 실제로 호출될 때만 import 한다(지연 import).
    """
    from vllm import SamplingParams  # noqa: PLC0415

    spec = profile_generation_kwargs(profile, overrides)

    guided_schema = _GUIDED_SCHEMAS.get(profile)
    if guided_schema is not None:
        # vLLM은 이 인자를 v0.8 전후로 리네이밍했다: 구버전은
        # `GuidedDecodingParams` + `SamplingParams(guided_decoding=...)`,
        # 신버전(예: 0.27)은 `StructuredOutputsParams` +
        # `SamplingParams(structured_outputs=...)`. 이 프로젝트는 vLLM 버전을
        # 못박지 않으므로(pyproject.toml 참조) 두 API를 모두 지원한다.
        try:
            from vllm.sampling_params import GuidedDecodingParams  # noqa: PLC0415

            spec["guided_decoding"] = GuidedDecodingParams(json=guided_schema)
        except ImportError:
            from vllm.sampling_params import StructuredOutputsParams  # noqa: PLC0415

            spec["structured_outputs"] = StructuredOutputsParams(json=guided_schema)

    return SamplingParams(**spec)


# --------------------------------------------------------------------------
# JSON 추출 + 관대한 파서: 응답이 형식을 어겼을 때 fallback 처리를 하는 부분
# (student_simulator_plan.md §8.4 "구조화 출력"에서 정의된 규칙)
# --------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Any:
    """모델이 낸 텍스트 출력에서 JSON 객체를 최대한 찾아내 뽑아낸다.

    guided decoding(JSON 형식을 강제하는 디코딩 방식)이 정상적으로 걸렸다면
    ``text`` 자체가 그대로 유효한 JSON이어야 정상이다. 하지만 다음과 같은
    경우들 때문에 그렇지 않을 수 있다: (a) max_tokens 제한에 걸려 응답이
    중간에 잘린 경우, (b) 어떤 이유로 guided decoding 자체가 적용되지 않은
    경로를 탄 경우, (c) 상대적으로 작은 8B 모델이 형식을 못 지키고 실패하는
    경우. 이런 경우를 대비해, 먼저 텍스트 전체를 JSON 으로 파싱해보고 실패하면
    텍스트 안에서 처음 나오는 ``{...}`` 블록(앞뒤에 다른 잡음 텍스트가 섞여
    있어도 그 부분만)을 찾아서 다시 한번 파싱을 시도한다.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError("no JSON object found in model output")
    return json.loads(match.group(0))


def bid_is_valid(text: str) -> bool:
    """``text`` 가 ``BidSchema`` 형식(입찰 응답이 지켜야 할 JSON 형식)을
    만족하는 유효한 JSON인지 판정한다.

    ``registry.py`` 가 "이 응답이 재시도가 필요할 만큼 깨졌는지"를 판단하는
    데 이 함수를 쓴다.
    """
    try:
        BidSchema.model_validate(_extract_json(text))
        return True
    except Exception:
        return False


def speak_is_valid(text: str) -> bool:
    """``text`` 가 ``SpeakSchema`` 형식(학생 발화 응답이 지켜야 할 JSON 형식)을
    만족하는 유효한 JSON인지 판정한다."""
    try:
        SpeakSchema.model_validate(_extract_json(text))
        return True
    except Exception:
        return False


def judge_is_valid(text: str) -> bool:
    """``text`` 가 ``JudgeSchema`` 형식을 만족하는 유효한 JSON인지 판정한다."""
    try:
        JudgeSchema.model_validate(_extract_json(text))
        return True
    except Exception:
        return False


def parse_bid(text: str) -> Bid:
    """입찰(bid) 응답을 최대한 너그럽게 해석하는 파서. **어떤 입력이 와도
    절대 예외(에러)를 던지지 않는다.**

    파싱에 실패하면 ``Bid(level="low", is_fallback=True)``, 즉 "낮은 입찰로
    처리하고 fallback 이 발동했다"는 표시가 담긴 값을 반환한다. 이는
    student_simulator_plan.md §8.4절이 정한 규칙이다: "파싱 실패 시에는
    fallback 으로 처리하며, bid 는 low(낮음)를 기본값으로 한다." 이 함수
    자체는 재시도를 하지 않는다 — 재시도는 이 함수가 아니라 ``registry.py``
    의 책임이다. ``registry.py`` 는 :func:`bid_is_valid` 로 응답이 깨졌는지
    확인한 뒤, 깨졌으면 엔진을 다시 한번 호출하고, 그 새로운 결과에 대해
    다시 이 함수를 적용한다. 두 번째 시도까지도 실패하면 그때는 자연스럽게
    다시 이 fallback 값이 나오게 된다.
    """
    try:
        data = BidSchema.model_validate(_extract_json(text))
        kc_updates = tuple(
            KCUpdate(kc_id=u.kc_id, resolved=u.resolved) for u in data.kc_updates
        )
        return Bid(
            level=data.bid,
            reason=data.reason,
            intent=data.intent,
            kc_updates=kc_updates,
            is_fallback=False,
        )
    except Exception:
        return Bid(
            level="low",
            reason="fallback: guided JSON parse failed",
            intent=None,
            kc_updates=(),
            is_fallback=True,
        )


def parse_speak(text: str) -> tuple[str, SelfReport]:
    """학생 발화 응답을 최대한 너그럽게 해석하는 파서. **어떤 입력이 와도
    절대 예외(에러)를 던지지 않는다.**

    JSON 으로 파싱하는 데 실패하면, 모델이 낸 원문 텍스트를 그대로 발화
    내용으로 사용한다(완전히 침묵 처리하는 것보다는, 형식은 어겼어도 사람이
    읽을 수 있는 텍스트일 가능성이 높은 원문을 그대로 쓰는 편이 낫다는
    판단이다). 이때 ``self_report``(자기 보고 필드)는 빈 값으로 채운다.
    ``SelfReport`` 는 어차피 로깅(기록) 전용으로만 쓰이고(contracts.py 참고)
    학생의 상태나 보상 계산에는 전혀 영향을 주지 않으므로, 이 fallback 이
    빈 값이어도 다른 로직에 영향이 없다.
    """
    try:
        data = SpeakSchema.model_validate(_extract_json(text))
        sr = data.self_report
        return data.utterance, SelfReport(
            changed_mind_about=sr.changed_mind_about, addressing=sr.addressing
        )
    except Exception:
        return text.strip(), SelfReport(changed_mind_about=None, addressing=None)


def parse_judge(text: str) -> tuple[JudgeProposal, ...]:
    """judge 응답을 최대한 너그럽게 해석하는 파서. **어떤 입력이 와도 절대
    예외(에러)를 던지지 않는다.**

    파싱에 실패하면 빈 튜플을 반환한다 -- "이 발화에서는 증거가 안
    나왔다"는 뜻으로, 판정 대상이 없었던 것과 똑같이 취급된다(0/60
    베이스라인보다 나빠지지 않는 안전한 축소). 이 함수 자체는 재시도를
    하지 않는다 -- 재시도는 ``registry.py`` 의 책임이다(``bid``/``speak``와
    동일한 구조)."""
    try:
        data = JudgeSchema.model_validate(_extract_json(text))
        return tuple(
            JudgeProposal(
                target=e.target,
                kc_id=e.kc_id,
                misconception_id=e.misconception_id,
                relation=e.relation,
            )
            for e in data.edges
        )
    except Exception:
        return ()
