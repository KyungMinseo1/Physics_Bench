"""물리적으로 GPU 1장짜리 서버 위에서, vLLM 엔진(GPU에서 대형 언어모델을
서빙하는 라이브러리) 1개를 백그라운드 프로세스로 계속 띄워두고 배치(여러
프롬프트를 한꺼번에 묶어) 추론을 요청하는 래퍼(감싸는 클래스) 모듈이다.

이 코드는 우리가 참고하는 원본 프로젝트(참조 repo)의 ``ParallelvLLMInference``
클래스(원본 경로: ``src/vllm/data_parallel_vllm.py``) 패턴을 우리 프로젝트로
가져온(이식한) 것이다. 원본 코드를 정리한 내용은
``docs/reference_notes.md`` 문서의 1.2~1.5절에 자세히 적혀 있다. 원본을 그대로
베끼지 않고, 우리가 실제로 배포하는 형태(GPU 1장만 사용, 튜터용·학생용 엔진
2개를 동시에 그 위에 띄움, 별도의 학습 루프는 없음)에 맞춰 필요한 부분만
정리했다. reference_notes.md 1.6절 표에 정리된, 원본과 다르게 의도적으로
바꾼 지점은 다음과 같다.

- ``model_save_path``(학습 중 갱신되는 모델 체크포인트 경로) 관련 기능과, 그
  경로를 주기적으로 확인(폴링)해서 워커를 재시작하는 ``_reload_workers`` 관련
  코드는 **모두 제거했다.** 지금 단계(시뮬레이션 단계)에서는 필요 없는
  기능이기 때문이다. 나중에 강화학습(RL) 단계(M6 이후)에서 다시 필요해지면,
  :meth:`run_batch` 메서드 안에 있는 "RELOAD HOOK"이라고 표시해둔 주석 위치에
  그 로직을 되살리면 된다.
- ``load_and_unload``(모델을 쓸 때만 불러오고 안 쓸 때는 내려두는 기능)는
  우리 코드에서는 상수로 항상 ``False`` 로 고정해뒀다 -- 그 경로 자체를
  아예 없앤 것이다.
- ``enable_sleep_mode``(모델을 재우고 깨우는 기능)는 **애초에는(student_
  simulator_plan.md §8.1 원안) 이것도 항상 꺼둘 계획이었다** -- 여러 모델을
  GPU 한 장에 계속 올려둬도 되니 필요 없다고 판단했었다. 하지만 실제로
  튜터/학생/judge 셋을 동시에 GPU 몇 장에 나눠 올리다 보니 메모리를 아낄
  필요가 생겨서, 2026-08-16부터 실제로 켜서 쓴다: 엔진 생성 시
  ``enable_sleep_mode=True`` 로 vLLM에게 "나중에 재워도 된다"고 허락해두고,
  ``ParallelVLLMEngine.sleep()``/``wake_up()`` 을 ``Classroom.rollout()``
  이 라운드마다 그 순간 필요 없는 엔진에 호출한다(아래 ``sleep``/
  ``wake_up`` 메서드, ``classroom.py`` 참고). ``enable_sleep_mode=True``
  로 띄우기만 하고 이 메서드들을 아무도 안 부르면 아무 일도 안 일어난다
  는 점에 유의할 것 -- vLLM의 sleep은 "자동으로 유휴 시 재우는" 기능이
  아니라, 명시적으로 호출해야만 재워지는 기능이다.
- ``enable_prefix_caching``(프롬프트 앞부분이 같으면 그 계산 결과를 재사용해
  속도를 높이는 캐싱 기능)은 생성자 인자로 밖에 노출해뒀고, 기본값은
  ``True``(켜짐)로 정했다. 원본 참조 repo 에서는 이 값이 ``False``로
  하드코딩(고정)되어 있었지만, 우리 상황에서는 대화 전사(기록)가 턴마다
  계속 누적되고 system prompt(시스템 지시문)도 항상 동일하기 때문에 캐싱으로
  얻는 속도 이득이 클 것으로 판단해 기본값을 바꿨다(1.6절 참고). 다만 실제로
  켰을 때와 껐을 때 어느 쪽이 나은지는 프로젝트 초기 단계(M0)에서 직접 측정해봐야
  한다 — ``docs/m0_checklist.md`` 참조.
- ``sampling_params``(생성 파라미터)를 워커 프로세스에 넘길 때, 우리는 딕셔너리
  (dict)가 아니라 **vLLM 자체의 ``SamplingParams`` 객체를 그대로** 만들어서
  넘긴다. 원본 참조 repo 가 dict 형태를 받아서 그걸 그대로 vLLM 내부까지
  전달하는지, 아니면 중간에 객체로 변환하는지는 reference_notes §1.5 절
  발췌 범위 안에서 확인되지 않았다(불확실한 부분). 그래서 우리는 확실하게
  안전한 방식인 "객체를 직접 만들어서 전달"하는 쪽을 택했다. 이 결정 때문에
  실제 ``SamplingParams`` 객체를 만드는 코드는 ``src/serving/registry.py`` 와
  ``src/serving/sampling.py`` 쪽에 있다.

vLLM 라이브러리는 **이 모듈에서 지연 import 한다**(실제로 필요한 순간에만
import 한다는 뜻). 즉 이 모듈 자체는, 워커 프로세스가 실제로 떠서 vLLM 을
불러 쓰기 전까지는 vLLM 이 설치되어 있지 않아도 문제없이 import 될 수 있어야
한다(GPU 없이 테스트하는 ``mock.py`` 경로가 이 성질에 의존한다).
"""

from __future__ import annotations

import math
import os
from multiprocessing import get_context
from queue import Empty
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - 타입 힌트 전용, 런타임에는 로드되지 않는다
    from vllm import SamplingParams

__all__ = ["ParallelVLLMEngine"]


def _is_chat_prompt(prompts: Sequence[Any]) -> bool:
    """``prompts`` 에 들어있는 원소 전체가 대화 메시지 목록(chat message list)
    형태인지 판정한다 (reference_notes §1.5 참고).

    모든 원소가 ``{"role": ..., "content": ...}`` (발화자 역할과 내용을 담은)
    딕셔너리들의 나열이면 "채팅 형태"로 간주해서 vLLM 의 ``llm.chat`` 경로를
    타게 하고, 그렇지 않으면 일반 텍스트 프롬프트로 보고 ``llm.generate``
    경로를 탄다. 우리 저장소에서는 ``contracts.ChatPrompt`` 규약에 따라
    프롬프트가 **항상 채팅 형태**이지만, 원본 참조 repo 에 있던 이 분기
    처리 자체는 만약을 대비해(방어적으로) 그대로 남겨뒀다.
    """
    if not prompts:
        return False
    for p in prompts:
        if not isinstance(p, (list, tuple)):
            return False
        for msg in p:
            if not isinstance(msg, Mapping) or "role" not in msg or "content" not in msg:
                return False
    return True


def _compute_gpu_groups(
    n_instances: int,
    gpus_per_instance: int,
    gpu_ids: Sequence[int] | None,
) -> list[list[Any]]:
    """워커 인스턴스마다 실제로 쓸 물리 GPU 번호 목록을 계산한다
    (``ParallelVLLMEngine.__init__`` 에서 뽑아낸 순수 함수 -- vLLM/GPU 없이
    단위 테스트할 수 있게 하기 위해서다, 2026-08-17 도입).

    우선순위:

    1. **``gpu_ids`` 가 명시적으로 주어지면 그걸 그대로 쓴다**
       (``config/models.yaml`` 의 역할별 ``tutor.vllm.gpu_ids`` 등 -- 튜터/
       학생/judge를 서로 다른 물리 GPU에 명시적으로 배정하고 싶을 때 쓴다.
       ``n_instances * gpus_per_instance`` 개수만큼 정확히 필요하고,
       ``gpus_per_instance`` 개씩 잘라서 워커 인스턴스별로 나눠 준다 --
       같은 GPU 번호를 여러 번 넣으면 여러 워커가 그 GPU를 공유한다(예:
       ``gpu_ids=[0, 0]`` + ``n_instances=2`` 는 워커 둘 다 GPU 0을 쓴다).
       ``CUDA_VISIBLE_DEVICES`` 환경변수보다 우선한다 -- 환경변수는 프로세스
       전체에 걸리는 전역 설정이라 엔진마다 다르게 줄 수 없지만, 이 값은
       엔진(역할)마다 다르게 넘길 수 있는 명시적 파라미터이기 때문이다.
    2. 없으면 ``CUDA_VISIBLE_DEVICES`` 환경변수를 읽어서 그 목록을 잘라
       쓴다(기존 동작).
    3. 그것도 없으면 0번부터 순서대로 쓴다(기존 동작 -- 이 프로세스에서
       보이는 GPU가 곧 물리 GPU 번호라고 가정한다).
    """
    required = n_instances * gpus_per_instance

    if gpu_ids is not None:
        source: list[Any] = list(gpu_ids)
        if len(source) < required:
            raise ValueError(
                f"gpu_ids에 최소 {required}개의 GPU 번호가 필요한데 "
                f"{len(source)}개만 주어졌습니다: {source}"
            )
        source = source[:required]
        return [source[i : i + gpus_per_instance] for i in range(0, required, gpus_per_instance)]

    designated_gpu_groups = os.environ["CUDA_VISIBLE_DEVICES"].split(",") if "CUDA_VISIBLE_DEVICES" in os.environ else []

    if designated_gpu_groups:
        if len(designated_gpu_groups) < required:
            raise ValueError(f"Need {required} GPUs, only {len(designated_gpu_groups)} available")
        designated_gpu_groups = designated_gpu_groups[:required]
        return [
            designated_gpu_groups[i : i + gpus_per_instance] for i in range(0, required, gpus_per_instance)
        ]

    return [list(range(i, i + gpus_per_instance)) for i in range(0, required, gpus_per_instance)]


def _worker_loop_fn(gpu_group: list[int], task_queue: Any, result_queue: Any, config: dict) -> None:
    """워커(작업) 프로세스가 시작될 때 실제로 실행되는 함수다. 이 워커
    프로세스는 파이썬 멀티프로세싱의 ``spawn`` 방식(부모와 메모리를 공유하지
    않고 완전히 새 파이썬 인터프리터를 띄우는 방식)으로 시작되므로, 이 모듈
    자체가 그 새 프로세스에서 처음부터 다시 import 된다.

    vLLM 라이브러리의 import 는 반드시 이 함수 **안에서** 일어나야 한다. 왜냐하면
    부모 프로세스가 이 모듈을 import 하는 시점, 혹은 이 모듈이 처음 로드되는
    시점에는 vLLM 이 설치되어 있지 않아도 아무 문제가 없어야 하기 때문이다
    (실제로 vLLM 이 필요해지는 건 워커가 뜬 뒤, 이 함수가 실행될 때뿐이다).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_group)

    print(
        f"[WORKER] PID={os.getpid()} "
        f"gpu_group={gpu_group} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )

    if config.get("use_v0"):
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM  # noqa: PLC0415  (일부러 함수 안에서 import 하는 지연 import 이다)

    llm_kwargs: dict[str, Any] = dict(
        model=config["model_path"],
        tensor_parallel_size=config["gpus_per_instance"],
        trust_remote_code=True,
        gpu_memory_utilization=config["gpu_memory_utilization"],
        max_model_len=config["max_model_len"],
        max_num_seqs=config["max_num_seqs"],
        enable_sleep_mode=config["enable_sleep_mode"],
        enforce_eager=config["enforce_eager"],
        enable_prefix_caching=config["enable_prefix_caching"],
    )
    if config.get("kv_cache_dtype"):
        llm_kwargs["kv_cache_dtype"] = config["kv_cache_dtype"]
    if config.get("quantization"):
        # if is_awq or bits_and_bytes, this will be overridden below
        llm_kwargs["quantization"] = config["quantization"]
    if config.get("limit_mm_per_prompt") is not None:
        llm_kwargs["limit_mm_per_prompt"] = config["limit_mm_per_prompt"]
    if config["use_lora"]:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = config["max_lora_rank"]

    is_awq = config["use_awq"] or "awq" in config["model_path"].lower()
    if is_awq:
        llm_kwargs["quantization"], llm_kwargs["load_format"] = "awq", "auto"
    elif config["bits_and_bytes"]:
        llm_kwargs["quantization"], llm_kwargs["load_format"] = "bitsandbytes", "bitsandbytes"
    else:
        llm_kwargs["quantization"], llm_kwargs["load_format"] = None, "auto"

    llm = LLM(**llm_kwargs)

    # LoRA 어댑터(경량 파인튜닝 결과물)를 실제로 어떻게 붙이는지에 대해: 참조
    # 문서(reference_notes)에 발췌되어 있는 범위 안에는 `_worker_loop_fn`이 이
    # 어댑터를 정확히 어떤 방식으로 적용하는지(요청 하나하나마다 LoRARequest를
    # 붙이는 방식인지, 아니면 엔진 전체의 기본값으로 미리 설정해두는 방식인지)가
    # 나와 있지 않다. 그래서 아래 코드는 vLLM 이 공식적으로 권장하는 일반적인
    # 패턴(요청마다 LoRARequest 객체를 만들어서 generate/chat 호출 시 함께
    # 넘기는 방식)을 그대로 따랐다. 이는 **원본 참조 repo 의 실제 동작과 일치하는지
    # 확인되지 않은, 우리가 추측으로 채운 부분이다** (최종 보고서의 "추측할 수밖에
    # 없었던 지점" 항목 참조).
    lora_request = None
    if config["use_lora"] and config.get("lora_path"):
        from vllm.lora.request import LoRARequest  # noqa: PLC0415

        lora_request = LoRARequest("tutor_adapter", 1, config["lora_path"])

    # ---- RELOAD HOOK (강화학습(RL) 단계에서만 쓰는 자리, 지금은 비워둠) -----
    # 원본 참조 repo 는 이 위치에서 `_get_latest_checkpoint_id()` /
    # `_reload_workers()` 를 호출해, 학습 중에 새로 저장된 모델 체크포인트가
    # 있는지 주기적으로 확인(폴링)하고 있으면 워커 프로세스를 재시작해서 최신
    # 모델로 갱신했다 (reference_notes §1.4 절의 세 번째 항목 참고). 지금은
    # 시뮬레이션 단계(마일스톤 M0~M5)라 이 기능이 필요 없어서 뺐다. 나중에
    # 강화학습 단계(M6 이후)에서 학습된 정책(policy) 체크포인트를 롤아웃(시뮬레이션
    # 진행)에 반영해야 할 때가 오면, 이 위치에 폴링 로직을 다시 넣으면 된다.
    # -------------------------------------------------------------------------

    result_queue.put("READY")

    while True:
        task = task_queue.get()
        kind = task[0]

        # sleep/wake_up(2026-08-16 도입, student_simulator_plan.md §8.1
        # 재검토): task_queue는 지금까지 "생성 요청" 하나만 실어 날랐는데,
        # GPU 메모리를 아끼려고 안 쓰는 엔진을 재웠다 깨웠다 하려면(
        # Classroom.rollout()이 라운드마다 그 순간 필요한 엔진만 깨우고
        # 나머지는 재우는 오케스트레이션, classroom.py 참고) 워커 쪽에도
        # 그 명령을 전달할 경로가 있어야 한다. 그래서 task를
        # (kind, ...) 튜플로 태그해서 "생성"과 "재우기"/"깨우기"를
        # 구분한다. 응답은 항상 별도 문자열 센티널("SLEPT"/"AWAKE")로
        # 보낸다 -- run_batch()의 결과 수집 루프가 리스트만 진짜 결과로
        # 취급하도록 방어적으로 바꿔뒀으므로(아래 run_batch 참고) 서로
        # 섞여도 안전하다.
        if kind == "sleep":
            level = task[1]
            llm.sleep(level=level)
            result_queue.put("SLEPT")
            continue

        if kind == "wake_up":
            tags = task[1]
            llm.wake_up(tags=tags)
            result_queue.put("AWAKE")
            continue

        # kind == "generate"
        _, chunk, sampling_params, meta = task
        prompts = [p for _, p in chunk]

        gen_kwargs: dict[str, Any] = {}
        if lora_request is not None:
            gen_kwargs["lora_request"] = lora_request
        if config.get("enable_thinking"):
            # Qwen3.5 같은 reasoning 모델에게 vLLM의 채팅 템플릿 자체가
            # <think> 스캐폴딩을 열어주도록 하는 네이티브 토글이다
            # (persona.build_tutor_prompt의 enable_thinking -- tutor.j2가
            # 프롬프트 텍스트로 <think> 사용법을 명시적으로 가르치는 것과는
            # 별개의, 상호 보강하는 메커니즘). 참조 repo(Dynamic-Peda-RL)는
            # 이 값을 생성자에 저장만 하고 실제로 llm.chat() 까지 흘려보내지
            # 않는 죽은 코드였다(src/vllm/data_parallel_vllm.py) -- 여기서는
            # 실제로 연결한다.
            gen_kwargs["chat_template_kwargs"] = {"enable_thinking": True}

        if _is_chat_prompt(prompts):
            outs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=True, **gen_kwargs)
        else:
            outs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True, **gen_kwargs)

        result_queue.put([(idx, out) for (idx, _), out in zip(chunk, outs)])


class ParallelVLLMEngine:
    """vLLM 엔진 1개를 백그라운드 프로세스(들)로 띄우고, 배치(여러 프롬프트를
    묶은) 추론 요청을 그 프로세스들에 전달·중계하는 클래스다.

    ``registry.py`` 가 이 클래스의 인스턴스를 튜터용/학생용/judge용으로
    최대 3개 만든다. 우리가 실제로 배포하는 환경(물리 GPU 1장짜리)에서는
    ``gpus_per_instance=1`` (인스턴스 하나당 GPU 1개 사용)로 두면
    ``gpu_groups=[[0]]``(0번 GPU 하나로 이뤄진 그룹 하나) 만 산출되어, 결과적으로
    워커 프로세스도 인스턴스당 1개만 뜬다 (reference_notes §1.2 참고). GPU가
    여러 장인 환경으로 나중에 확장할 가능성을 열어두기 위해, 여러 인스턴스를
    산출하는 로직 자체는 원본 참조 repo 와 동일하게 그대로 남겨뒀다.

    **``gpu_ids`` 로 물리 GPU를 역할별로 명시 지정할 수 있다**(2026-08-17
    도입, ``_compute_gpu_groups`` 참고) -- 예를 들어 튜터/학생은 GPU 0에
    같이 올리고 judge만 GPU 1로 분리하고 싶으면, ``config/models.yaml``
    의 ``tutor.vllm.gpu_ids: [0]``/``student.vllm.gpu_ids: [0]``/
    ``judge.vllm.gpu_ids: [1]`` 처럼 준다. 안 주면(기존 동작) 이 프로세스가
    보는 ``CUDA_VISIBLE_DEVICES`` 환경변수를 읽거나, 그것도 없으면 0번부터
    순서대로 잡는다 -- 이 경우 엔진끼리 서로 어떤 GPU를 이미 쓰고 있는지
    모르므로, 여러 엔진을 만들면 전부 같은 번호(0, 1, ...)로 되돌아가서
    물리 GPU가 우연히 겹칠 수 있다(``gpu_ids`` 를 명시하는 게 안전하다).

    .. warning::
       :meth:`__init__` (생성자)이 실행되면 워커 프로세스를 띄우는 과정에서
       부모 프로세스의 ``CUDA_VISIBLE_DEVICES``(어떤 GPU를 쓸지 지정하는
       환경변수) 값을 **프로세스 전체에 영향을 주는 방식으로 덮어써 버린다**
       (reference_notes §1.3 참고). 그러므로 이 클래스의 인스턴스 2개를
       동시에(비동기 작업이나 별도 스레드로 병렬로) 생성하면, 이 환경변수
       값을 서로 먼저 쓰려고 경합하는 문제가 생긴다. **반드시 인스턴스를
       하나씩 순서대로 생성해야 한다.** 이 순서를 지키도록 강제하는 일은
       ``registry.py`` 에 있는 엔진 생성 팩토리 함수가 담당한다.
    """

    def __init__(
        self,
        model_path: str,
        *,
        gpus_per_instance: int = 1,
        n_instances: int | None = None,
        max_number_of_instances: int = -1,
        from_0: bool = True,
        gpu_ids: Sequence[int] | None = None,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 12000,
        max_num_seqs: int = 512,
        enable_sleep_mode: bool = False,
        enforce_eager: bool = False,
        use_lora: bool = False,
        lora_path: str | None = None,
        max_lora_rank: int = 128,
        enable_prefix_caching: bool = True,
        kv_cache_dtype: str | None = None,
        quantization: str | None = None,
        use_v0: bool = False,
        enable_thinking: bool = False,
        use_awq: bool = False,
        bits_and_bytes: bool = False,
        limit_mm_per_prompt: Mapping[str, int] | None = None,
    ) -> None:
        self.model_path = model_path
        self.gpus_per_instance = gpus_per_instance
        self.n_instances = n_instances if n_instances is not None else 1
        self.from_0 = from_0

        self.gpu_groups = _compute_gpu_groups(self.n_instances, gpus_per_instance, gpu_ids)

        if max_number_of_instances > 0:
            self.n_instances = min(
                max_number_of_instances,
                self.n_instances,
            )
            self.gpu_groups = self.gpu_groups[:self.n_instances]
        # -------------------------------------------------------------------------

        self._worker_config = dict(
            model_path=model_path,
            gpus_per_instance=gpus_per_instance,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enable_sleep_mode=enable_sleep_mode,
            enforce_eager=enforce_eager,
            use_lora=use_lora,
            lora_path=lora_path,
            max_lora_rank=max_lora_rank,
            enable_prefix_caching=enable_prefix_caching,
            kv_cache_dtype=kv_cache_dtype,
            quantization=quantization,
            use_v0=use_v0,
            enable_thinking=enable_thinking,
            use_awq=use_awq,
            bits_and_bytes=bits_and_bytes,
            limit_mm_per_prompt=dict(limit_mm_per_prompt) if limit_mm_per_prompt is not None else None,
        )

        #: 지금 재워둔 상태인가 (2026-08-16 도입). ``LLM(...)`` 생성 직후는
        #: 항상 깨어있는 상태로 시작하므로 ``False`` 로 초기화한다.
        #: ``sleep()``/``wake_up()`` 이 이 값을 보고 이미 목표 상태면 아무
        #: 것도 안 하고 조용히 리턴한다 -- **실제로 vLLM을 이미 잠든 엔진에
        #: 다시 ``sleep()`` 을 호출하면 죽는 것을 실측으로 확인했다**(호출자
        #: 쪽에서 실수로 두 번 재우는 경우를 완전히 막을 수 없으므로, 엔진
        #: 자신이 자기 상태를 기억하고 방어하는 게 유일하게 안전한 방법이다
        #: -- ``Classroom.rollout()`` 이 아무리 조심해서 호출 순서를 짜도,
        #: 이 가드가 없으면 앞으로 생길 다른 호출 경로의 실수까지는 못
        #: 막는다).
        self._asleep = False

        self.ctx: Any = None
        self.task_queues: list[Any] = []
        self.result_queues: list[Any] = []
        self.processes: list[Any] = []
        self._start_workers()

    def _start_workers(self) -> None:
        """워커 프로세스들을 실제로 띄우고, 각 워커가 "준비 완료(READY)"라는
        신호를 보낼 때까지 기다린다(핸드셰이크. reference_notes §1.3절 참고).

        체크포인트 다시 불러오기(리로드) 관련 설정값은 우리 배포 방식에서는
        항상 꺼져 있는 기능이므로, 아예 워커 설정 dict 자체에 넣지 않고 뺐다
        (원본 참조 repo 코드는 이 값도 함께 워커에 넘겼었다). sleep/wake
        (재우기·깨우기)는 반대로 2026-08-16부터 실제로 쓰인다 --
        ``enable_sleep_mode`` 로 엔진을 띄운 뒤 ``sleep()``/``wake_up()``
        메서드로 호출한다(위 클래스 독스트링, ``Classroom.rollout()`` 참고).
        """
        self.ctx = get_context("spawn")
        print(f"[ENGINE] n_instances={self.n_instances}")
        print(f"[ENGINE] gpu_groups={self.gpu_groups}")
        self.task_queues = [self.ctx.Queue() for _ in range(self.n_instances)]
        self.result_queues = [self.ctx.Queue() for _ in range(self.n_instances)]
        self.processes = []

        for idx, gpu_group in enumerate(self.gpu_groups):
            print(f"[ENGINE] starting worker {idx}: GPU {gpu_group}")

            p = self.ctx.Process(
                target=_worker_loop_fn,
                args=(gpu_group, self.task_queues[idx], self.result_queues[idx], self._worker_config),
            )
            p.start()
            self.processes.append(p)

        print(f"[ENGINE] started {len(self.processes)} workers")
        try:
            for i, q in enumerate(self.result_queues):
                while True:
                    try:
                        if q.get(timeout=0.5) != "READY":
                            raise RuntimeError(f"Worker {i} sent an invalid startup response")
                        break
                    except Empty:
                        if not self.processes[i].is_alive():
                            raise RuntimeError(f"vLLM worker {i} exited during startup")
        except BaseException:
            self.shutdown()
            raise

    def run_batch(
        self,
        prompts: Sequence[Any],
        sampling_params: "SamplingParams",
        meta: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        """``prompts``(프롬프트 목록)를 여러 워커 프로세스에 나눠서 배치 생성을
        시키고, 그 결과를 원래 순서를 그대로 유지한 채로 반환한다.

        반환되는 리스트의 각 원소는 **가공되지 않은, vLLM 이 그대로 낸
        ``RequestOutput`` 객체**다. contracts.py 가 요구하는 것처럼 이 객체들을
        문자열 리스트(``list[str]``)로 바꾸는 변환 작업은 이 클래스가 아니라
        ``registry.py`` 에 있는 어댑터의 책임이다. 이 경계는 reference_notes
        §1.6절에서 우리가 의도적으로 정한 역할 분담이다.

        Args:
            prompts: 대화 메시지 목록(chat message list)들을 모아놓은 시퀀스.
            sampling_params: vLLM 이 요구하는 ``SamplingParams`` 객체. 딕셔너리
                (dict)가 아니라는 점에 주의한다(reference_notes §1.6절 — 원본
                참조 repo 는 dict 를 그대로 넘겼지만, 우리는 그와 다르게 객체를
                그대로 넘긴다).
            meta: 워커 프로세스에 그대로 전달되는 부가 정보. 지금은 실제로 쓰이지
                않고, 나중에 기능을 확장할 때 쓸 수 있도록 남겨둔 자리다.
        """
        if self.n_instances <= 0:
            raise RuntimeError(
                f"No vLLM instances available (n_instances={self.n_instances}). "
                "Check GPU allocation and initialization."
            )

        # ---- RELOAD HOOK (강화학습(RL) 단계에서만 쓰는 자리) ----
        # 원본 참조 repo 는 run_batch 가 호출될 때마다 체크포인트가 새로 갱신됐는지
        # 확인(폴링)해서, 갱신됐으면 워커 프로세스를 재시작했다 (reference_notes
        # §1.4절 세 번째 항목 참고). 지금 시뮬레이션 단계에서는 model_save_path
        # (체크포인트 경로) 자체가 없으므로 이 로직을 뺐다. 나중에 강화학습(RL)
        # 단계에서 학습된 정책(policy) 체크포인트를 반영해야 할 때가 오면, 여기에
        # `_get_latest_checkpoint_id()` / `_reload_workers()` 호출을 다시 넣으면 된다.
        # -------------------------------------

        indexed = list(enumerate(prompts))
        total = len(indexed)
        if total == 0:
            return []

        num_workers_to_use = min(self.n_instances, total)
        chunk_size = math.ceil(total / num_workers_to_use)
        chunks = [indexed[i * chunk_size : (i + 1) * chunk_size] for i in range(num_workers_to_use)]

        for i, chunk in enumerate(chunks):
            self.task_queues[i].put(("generate", chunk, sampling_params, meta))

        results: list[tuple[int, Any]] = []
        received = 0
        expected = total
        while received < expected:
            for i in range(num_workers_to_use):
                q = self.result_queues[i]
                try:
                    batch = q.get(block=True, timeout=0.1)
                except Empty:
                    if not self.processes[i].is_alive():
                        raise RuntimeError(f"vLLM worker {i} exited during inference")
                    continue
                # "READY"(기동 핸드셰이크)/"SLEPT"/"AWAKE"(sleep()/wake_up()
                # 응답, 아래 참고) 같은 문자열 센티널은 여기서 걸러낸다 --
                # 이 루프는 run_batch() 호출 중에만 도는데, sleep()/wake_up()
                # 은 절대 run_batch()와 동시에 호출되지 않으므로(이 클래스는
                # 애초에 단일 스레드 순차 호출 전제) 실제로 섞일 일은 없지만,
                # 문자열이 실수로 여기 들어오면 `results.extend("SLEPT")`
                # 처럼 글자 단위로 쪼개져 조용히 데이터가 깨지는 사고를 막기
                # 위해 "리스트일 때만 진짜 결과"라는 조건으로 명시적으로
                # 방어한다(예전엔 `!= "READY"`만 확인했다).
                if isinstance(batch, list):
                    results.extend(batch)
                    received += len(batch)

        # 각 결과에 매겨둔 인덱스(순번) 기준으로 정렬해서 원래 순서를 보존한다.
        # 이는 원본 참조 repo 의 run_batch 에서도 핵심적으로 검증된 동작 방식이다
        # (reference_notes §1.4절 첫 번째 항목 참고).
        return [out for _, out in sorted(results, key=lambda x: x[0])]

    def sleep(self, level: int = 1) -> None:
        """모든 워커의 vLLM 엔진을 재운다(GPU 메모리를 오프로드해서 비운다,
        vLLM의 ``LLM.sleep(level=...)``). 이미 재워둔 상태면(``self._asleep``)
        **아무 것도 하지 않고 조용히 리턴한다** -- 이미 잠든 vLLM 엔진에
        ``sleep()`` 을 다시 호출하면 실제로 죽는 것을 실측으로 확인했다.
        그렇지 않으면 워커 전원에게 명령을 보내고, 전원이 완료 응답을 보낼
        때까지 블록한다.

        ``enable_sleep_mode=True`` 로 엔진을 띄운 것과는 별개다 --
        ``enable_sleep_mode`` 는 "나중에 이 메서드를 호출해도 되게 허용"만
        하는 생성 시점 플래그일 뿐이고, 실제로 재우려면 이 메서드를 호출해야
        한다(``Classroom.rollout()`` 이 라운드마다 그 순간 필요 없는 엔진에
        이 메서드를 호출한다, student_simulator_plan.md §8.1 재검토 --
        2026-08-16).

        ``run_batch()`` 와 동시에(다른 스레드 등에서) 호출하면 안 된다 --
        이 클래스는 애초에 단일 스레드에서 순차 호출되는 걸 전제로 만들어졌다
        (``build_registry()`` 경고 참고).
        """
        if self._asleep:
            return
        for q in self.task_queues:
            q.put(("sleep", level))
        for q in self.result_queues:
            q.get()
        self._asleep = True

    def wake_up(self, tags: list[str] | None = None) -> None:
        """모든 워커의 vLLM 엔진을 깨운다(오프로드했던 가중치를 GPU로
        되돌림, vLLM의 ``LLM.wake_up(tags=...)``). 이미 깨어있는 상태면
        (``not self._asleep``) 아무 것도 하지 않고 조용히 리턴한다 --
        ``sleep()`` 과 대칭인 가드다. 그렇지 않으면 워커 전원에게 명령을
        보내고, 전원이 완료 응답을 보낼 때까지 블록한다. ``sleep()`` 과
        마찬가지로 ``run_batch()`` 와 동시 호출은 안 된다."""
        if not self._asleep:
            return
        for q in self.task_queues:
            q.put(("wake_up", tags))
        for q in self.result_queues:
            q.get()
        self._asleep = False

    def shutdown(self) -> None:
        """워커 프로세스들을 종료한다.

        원본 참조 repo 에서 발췌된 범위 안에는, "정상적으로 종료 신호를 보내서
        마무리 짓는" 절차(예: 작업 큐를 통해 종료하라는 신호를 보내는 방식)가
        나타나 있지 않다. ``_worker_loop_fn`` 코드를 보면 큐에서 받은 작업을
        ``("generate", chunk, sampling_params, meta)``/``("sleep", level)``/
        ``("wake_up", tags)`` 세 가지로만 해석할 뿐, 종료 신호를 별도로 받아
        처리하는 분기가 확인되지 않았다.
        그래서 여기서는 확인되지 않은 종료 절차를 짐작해서 만들어 넣는 대신,
        그냥 프로세스를 직접 강제 종료(``terminate``)하는 방식을 쓴다. 이는
        배치 추론이 정확하게 동작하는지와는 무관한, 운영상의 뒷정리(리소스
        정리) 조치일 뿐이며 reference_notes 문서에서 이미 확인된 동작을
        바꾸는 것은 아니다.
        """
        for p in self.processes:
            if p.is_alive():
                p.terminate()
        for p in self.processes:
            p.join(timeout=5)
