"""GPU 없이 전체 파이프라인을 테스트로 돌리기 위한 가짜(모의) 엔진.

실제로는 vLLM(GPU에서 대형 언어모델을 서빙하는 라이브러리)을 호출해야 하는
자리에, 이 ``MockEngine`` 을 대신 꽂으면 GPU도 vLLM 설치도 없이 코드를
끝까지 실행해 볼 수 있다.

``MockEngine`` 은 ``contracts.Engine`` 이 요구하는 메서드(``run_batch``)를
그대로 구현하고 있어서, 별도의 상속 없이도 "``Engine`` 타입이다"라고 간주된다
(파이썬의 구조적 타이핑 방식 — ``isinstance(MockEngine(), Engine)`` 이 참이 된다).
이 클래스는 vLLM 에 **전혀 의존하지 않는다.** ``src/serving/sampling.py`` 가
정의한 pydantic 스키마(응답 JSON 형태를 검증하는 클래스)만 재사용하는데,
그 파일 쪽에서도 vLLM 은 함수 안에서만 필요할 때 import 하므로(지연 import)
여기서 가져다 써도 vLLM 미설치 환경에서 문제가 없다.

이 클래스를 만든 핵심 목적은 "테스트하기 쉽게 만들기"다. 그래서 생성자에
다음 세 가지 테스트용 훅을 뒀다.

- ``scripted`` : 특정 프로파일(호출 종류)에 대해 미리 정해둔 응답을 순서대로
  내보내게 한다. 값은 응답 목록(list)이거나, 매 호출마다 실행되는 함수(callable)일
  수 있다. 예를 들어 "학생 전원이 low를 불렀을 때 침묵 처리가 되는지" 테스트하려면
  ``MockEngine(scripted={"student_bid": lambda prompt, i: '{"bid":"low","reason":"t"}'})``
  처럼 bid 프로파일 응답을 강제로 low 로 고정하면 된다.
- ``malformed_rate`` : ``student_bid`` / ``student_speak`` 응답을 일부러 깨진
  형태(불완전한 JSON)로 만들어서, ``registry.py`` 에 있는 "응답이 깨졌을 때 한 번
  재시도하고 그래도 안 되면 기본값으로 넘어간다"는 로직이 실제로 잘 동작하는지
  테스트할 수 있게 한다.
- ``calls`` : 지금까지 들어온 모든 ``run_batch`` 호출 내역을 기록해둔다. 프롬프트에
  민감한 정보가 실수로 섞여 들어가지 않았는지 등을 테스트에서 검사할 때 쓴다.

결정성(같은 입력이면 항상 같은 출력이 나오는 성질)에 대해: 같은 ``seed``
값으로, 같은 순서(어떤 profile을, 몇 개의 프롬프트로, 어떤 overrides로 호출했는지)로
호출하면 항상 같은 결과가 나온다. 다만 프롬프트의 *내용*(텍스트 자체)은 난수를
만드는 시드에 섞어 넣지 않는다. 만약 섞는다면 프롬프트 텍스트가 한 글자만
바뀌어도 "일부러 깨뜨리는" 패턴 자체가 완전히 달라져 버려서 테스트를 안정적으로
작성하기 어려워지기 때문이다. 대신 "이 프로파일이 몇 번째로 호출됐는가"라는
누적 카운터를 시드에 섞어서, 프롬프트 내용과 무관하게 안정적인 결과를 낸다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.contracts import ChatPrompt, SamplingProfile
from src.serving.sampling import BidSchema, JudgeSchema, SelfReportSchema, SpeakSchema

__all__ = ["MockEngine", "CallRecord"]

# scripted 딕셔너리의 값 하나는 아래 두 형태 중 하나가 올 수 있다. 그래서 타입을
# (Mapping 의 값 타입을) 구체적으로 좁히지 않고 Any 로 넓게 잡아뒀다.
#   - list[str]                                  : 먼저 넣은 응답부터 순서대로(FIFO) 하나씩 꺼내 쓰는 고정 응답 목록
#   - Callable[[ChatPrompt, int], str | None]     : 호출마다 실행되는 함수. None 을 반환하면 기본 생성 로직으로 대신 처리한다


@dataclass(frozen=True)
class CallRecord:
    """``run_batch`` 호출 1건을 기록한 값. 테스트 코드가 ``engine.calls`` 를 통해
    "무엇이 호출됐는지"를 나중에 확인할 때 읽는다 (예: 민감정보 유출 여부 검사)."""

    profile: SamplingProfile
    prompts: list[ChatPrompt]
    overrides: Mapping[str, Any] | None


class MockEngine:
    """``contracts.Engine`` 규격을 만족하는, GPU와 vLLM 없이 동작하는 가짜 엔진.

    Args:
        seed: 결정적 생성(같은 입력이면 항상 같은 출력)을 위한 난수 시드.
            같은 seed + 같은 순서로 호출하면 항상 같은 결과가 나온다.
        scripted: ``{profile 이름: 응답목록 또는 함수}`` 형태의 딕셔너리로, 정해진
            응답을 강제로 내보내고 싶을 때 쓴다. 값이 목록(list)이면, 그 프로파일로
            들어오는 호출에 대해 프롬프트 1개당 응답 1개씩(여러 번의 ``run_batch``
            호출에 걸쳐서도 누적해서) 순서대로 꺼내 쓴다. 목록이 다 소진되면 이후에는
            기본 생성 로직으로 넘어간다(폴백). 값이 함수(callable)이면
            ``(prompt, call_index) -> str | None`` 형태로 프롬프트마다 호출되며,
            ``None`` 을 반환하면 역시 기본 생성 로직으로 넘어간다.
        malformed_rate: ``malformed_profiles`` 에 속한 프로파일 중, scripted 로
            채워지지 않고 새로 생성된 응답을 이 확률로 일부러 깨뜨린다. "응답이
            깨졌을 때의 재시도/fallback 처리"를 테스트하기 위한 옵션이다. 0.0이면
            아무 응답도 깨지지 않는다.
        malformed_profiles: ``malformed_rate`` 를 적용할 프로파일들의 집합. 기본값은
            guided-JSON(정해진 JSON 형식을 강제하는 디코딩 방식)이 적용되는
            ``student_bid`` / ``student_speak`` 두 가지뿐이다.
    """

    def __init__(
        self,
        seed: int = 0,
        scripted: Mapping[str, Any] | None = None,
        malformed_rate: float = 0.0,
        malformed_profiles: Sequence[str] = ("student_bid", "student_speak"),
    ) -> None:
        self._seed = seed
        self._scripted: dict[str, Any] = dict(scripted or {})
        self._scripted_cursors: dict[str, int] = {}
        self._malformed_rate = malformed_rate
        self._malformed_profiles = frozenset(malformed_profiles)
        self._profile_call_counter: dict[str, int] = {}
        self.calls: list[CallRecord] = []
        #: ``sleep()``/``wake_up()`` 호출 기록("sleep"/"wake_up" 문자열이
        #: 호출 순서대로 쌓인다). ``Classroom.rollout()`` 이 2026-08-16부터
        #: 라운드마다 필요한 엔진만 깨우고 재우는 오케스트레이션을 하므로
        #: (student_simulator_plan.md §8.1 재검토), 테스트가 그 순서를
        #: 검증할 수 있게 ``.calls``(run_batch 전용)와는 별도로 남겨둔다.
        #: GPU가 없는 MockEngine에서는 이 메서드들이 실제로 아무것도 재우지
        #: 않는 no-op이다 -- 그래도 "실제로 상태가 바뀐 호출"만 기록해서,
        #: "튜터 차례엔 튜터만 깨어있고 학생은 자고 있었는가" 같은 걸
        #: 확인할 수 있게 한다(이미 그 상태인데 또 부르면 기록하지 않는다
        #: -- 아래 ``_asleep`` 가드와 같은 이유. 진짜 vLLM 엔진(
        #: ParallelVLLMEngine)이 이미 잠든 상태에서 sleep()을 또 부르면
        #: 실제로 죽는 것을 실측했으므로, MockEngine도 같은 가드 동작을
        #: 그대로 흉내내야 Classroom.rollout()의 오케스트레이션 버그를
        #: GPU 없이도 테스트에서 잡아낼 수 있다).
        self.sleep_wake_calls: list[str] = []
        self._asleep = False

    # -- contracts.Engine 이 요구하는 인터페이스 --------------------------

    def sleep(self) -> None:
        if self._asleep:
            return
        self.sleep_wake_calls.append("sleep")
        self._asleep = True

    def wake_up(self) -> None:
        if not self._asleep:
            return
        self.sleep_wake_calls.append("wake_up")
        self._asleep = False

    def run_batch(
        self,
        prompts: Sequence[ChatPrompt],
        profile: SamplingProfile,
        overrides: Mapping[str, Any] | None = None,
    ) -> list[str]:
        prompts = list(prompts)
        self.calls.append(
            CallRecord(profile=profile, prompts=prompts, overrides=dict(overrides) if overrides else None)
        )
        if not prompts:
            return []

        n = 4 if profile == "posttest" else 1
        if overrides and "n" in overrides:
            n = int(overrides["n"])

        out: list[str] = []
        for prompt in prompts:
            call_idx = self._profile_call_counter.get(profile, 0)
            self._profile_call_counter[profile] = call_idx + 1
            for sample_idx in range(n):
                out.append(self._resolve(prompt, profile, call_idx, sample_idx))
        return out

    # -- 아래는 클래스 내부에서만 쓰는 헬퍼 메서드 --------------------------

    def _resolve(
        self, prompt: ChatPrompt, profile: SamplingProfile, call_idx: int, sample_idx: int
    ) -> str:
        scripted_value = self._take_scripted(prompt, profile, call_idx)
        if scripted_value is not None:
            return scripted_value

        text = self._generate(prompt, profile, call_idx, sample_idx)

        if profile in self._malformed_profiles and self._malformed_rate > 0:
            rng = random.Random(f"malformed:{self._seed}:{profile}:{call_idx}:{sample_idx}")
            if rng.random() < self._malformed_rate:
                text = self._corrupt(text)
        return text

    def _take_scripted(
        self, prompt: ChatPrompt, profile: SamplingProfile, call_idx: int
    ) -> str | None:
        entry = self._scripted.get(profile)
        if entry is None:
            return None
        if callable(entry):
            return entry(prompt, call_idx)
        # 여기서부터는 entry 를 list[str] 로 취급한다. 앞에서부터 순서대로 하나씩
        # 꺼내 쓰고(FIFO), 다 꺼내 썼으면 None 을 반환해 기본 생성 로직으로 넘긴다.
        cursor = self._scripted_cursors.get(profile, 0)
        if cursor < len(entry):
            self._scripted_cursors[profile] = cursor + 1
            return entry[cursor]
        return None

    def _generate(
        self, prompt: ChatPrompt, profile: SamplingProfile, call_idx: int, sample_idx: int
    ) -> str:
        """프로파일(호출 종류)마다, 그 프로파일이 기대하는 형식에 맞는 그럴듯한
        가짜 응답을 만든다. seed 가 같으면 항상 같은 응답이 나온다."""
        if profile == "student_bid":
            rng = random.Random(f"gen:{self._seed}:{profile}:{call_idx}")
            level = rng.choice(["high", "medium", "low"])
            intent = rng.choice(["reinforce", "waver", "neutral"])
            # kc_updates: 대부분 빈 채로 두고(가장 흔한 실제 상황 -- 대개
            # 아무것도 안 바뀐다), 가끔 kc1~kc4 중 하나를 무작위로 제안한다.
            # 이 값은 그 학생의 실제 latent와 무관한 값일 수 있지만,
            # apply_knowledge_update가 이미 갖고 있지 않은/mastered인
            # kc_id는 방어적으로 무시하므로 안전하다.
            n_updates = rng.choice([0, 0, 0, 1, 2])
            kc_updates = [
                {"kc_id": rng.choice(["kc1", "kc2", "kc3", "kc4"]), "resolved": rng.random() < 0.3}
                for _ in range(n_updates)
            ]
            payload = BidSchema(
                bid=level,
                reason=f"mock reason #{call_idx}",
                intent=intent,
                kc_review=f"mock kc review #{call_idx}",
                kc_updates=kc_updates,
            )
            return payload.model_dump_json(by_alias=True)

        if profile == "student_speak":
            snippet = _last_user_text(prompt)[:80]
            utterance = f"[mock utterance #{call_idx}] {snippet}".strip()
            payload = SpeakSchema(utterance=utterance, self_report=SelfReportSchema())
            return payload.model_dump_json(by_alias=True)

        if profile == "posttest":
            return f"mock-answer-{call_idx}-{sample_idx}"

        if profile == "judge":
            # judge 기본 생성은 "아무 것도 겨냥하지 않았다"(빈 edges)로 둔다 --
            # student_bid의 kc_updates 기본값(대부분 빈 채로 둠)과 같은
            # 이유로, 이게 실제로 가장 흔한 판정 결과이기도 하다. 특정
            # 간선을 검증하려는 테스트는 ``scripted={"judge": [...]}`` 로
            # 원하는 응답을 직접 지정한다.
            return JudgeSchema(edges=[]).model_dump_json()

        if profile == "tutor":
            # 튜터(교사 역할) 응답은 앞에 반드시 "SAY:" 접두사를 붙여야 한다.
            # 이 접두사가 없으면 응답을 해석하는 ``parse_tutor_output`` 함수가
            # 매번 파싱에 실패하고, 그러면 mock 으로 굴리는 테스트 전체가
            # "파싱 실패했을 때의 대체 경로"만 타게 된다. 그 결과 GPU 없이
            # 돌릴 수 있는 유일한 통합 테스트 경로에서 정상적인 성공 경로를
            # 단 한 번도 검증하지 못하는 문제가 생긴다(실제로 8턴 중 8턴 모두
            # 실패하는 것을 확인했다). 아래 응답에는 특정 학생을 지목하는
            # "지명"은 포함하지 않으므로, 매 턴 학생들이 서로 발언권을 얻으려
            # 경쟁하는 BID_STAGE(입찰 단계) 상태로 이어진다. 특정 학생을
            # 지명하는 분기를 테스트하려면 후보 학생 id 를 알아야 하므로,
            # 그런 경우는 ``scripted={"tutor": [...]}`` 로 원하는 응답을
            # 직접 지정해서 검증한다.
            return f"SAY: [mock tutor response #{call_idx}]"

        # regression(회귀 테스트) 프로파일: 형식 제약 없는 자유 텍스트로 응답한다.
        return f"[mock {profile} response #{call_idx}]"

    @staticmethod
    def _corrupt(text: str) -> str:
        """JSON 문자열을 일부러 깨뜨린다. 뒷부분을 잘라내 닫는 괄호가 없는
        불완전한 문자열로 만드는 방식이다."""
        if len(text) <= 2:
            return "{not json"
        return text[: max(2, len(text) // 2)]


def _last_user_text(prompt: ChatPrompt) -> str:
    """대화 메시지 목록(chat message list)에서 가장 마지막 user(사용자) 메시지의
    내용(content)을 꺼낸다. user 메시지가 하나도 없으면 빈 문자열을 반환한다."""
    for msg in reversed(list(prompt)):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""
