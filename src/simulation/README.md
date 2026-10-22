# 문제 풀이 시뮬레이션

프로젝트 루트에서 실행합니다. 입력은 `annotation.question.text`, 선택지와
`annotation.question.images`의 실제 이미지입니다. `solution_steps`, `answer`,
`visual_information` 등 사람이 작성한 해설 annotation과 제출자 정보는 모델에 전달하지 않습니다.

```powershell
# GPU 없이 전체 dataset의 경로 및 멀티모달 프롬프트 구성 검증
python -m src.simulation --dry-run

# vLLM이 설치된 Linux/CUDA 환경에서 먼저 문제 2건 실행
python -m src.simulation --limit 2

# 전체 실행 (기존 결과 파일을 덮어쓰지 않음)
python -m src.simulation --output outputs/physics-run-01.jsonl

# 특정 문제만 실행; --problem-id를 여러 번 지정할 수 있음
python -m src.simulation --problem-id P_4b2b8e00956d43a1

# JSON 배열이나 개별 annotation 파일도 지원
python -m src.simulation --dataset dataset/dataset.json --dry-run
```

Python 3.11 이상을 사용합니다. 의존성은 프로젝트 루트의 `pyproject.toml`에서 관리합니다.
프로젝트 루트에서 필요한 실행 방식에 맞춰 설치하세요.

```powershell
# 기본 패키지 (GPU 없이 데이터 검증 및 unittest 실행 가능)
python -m pip install -e .

# Linux/CUDA 추론 환경: 기존 vLLM 엔진과 이미지 문제 시뮬레이션
python -m pip install -e ".[vllm]"

# 기존 API 서빙 모듈에 필요한 선택 의존성
python -m pip install -e ".[openrouter]"
python -m pip install -e ".[gemini]"

# 여러 선택 의존성 함께 설치
python -m pip install -e ".[vllm,openrouter,gemini]"
```

vLLM이 호환되는 PyTorch와 Transformers 버전을 설치하므로 두 패키지를 별도로
고정하지 않습니다. 테스트는 표준 라이브러리 `unittest`를 사용해 추가 의존성이 없습니다.
기존 API/registry 모듈이 참조하는 `src.contracts`는 현재 저장소에 없으므로,
API 선택 의존성 설치만으로 해당 모듈의 실행이 완성되지는 않습니다.
현재 로컬 검증 환경에는 vLLM/GPU 의존성이 없습니다.

`config/models.yaml`에서 이미지 입력이 가능한 모델 경로, GPU 배치와 생성 길이를
지정합니다. 기존 tutor/student/judge 역할 설정은 현재 시뮬레이션에 맞춰
단일 모델의 `model_path`, `vllm`, `sampling` 구조로 통합했습니다.
다른 설정은 `--config 경로`로 지정할 수 있으며 YAML과 JSON을 지원합니다.
`vllm` 항목은 `ParallelVLLMEngine` 생성자 인자이며, `sampling` 항목은
vLLM `SamplingParams` 인자입니다. 한 문제당 한 응답(`n=1`)을 사용합니다.
`limit_mm_per_prompt.image`는 문제당 최대 이미지 수 이상이어야 합니다.
GPU 메모리가 부족하면 `--batch-size`, `max_num_seqs`, `max_model_len`을 조정하세요.

출력 JSONL은 한 줄에 한 문제이며 `problem_id`, `model`, `problem`, `choices`,
`images`, `solution_steps`(문자열 배열), `answer`(`text`, `choice_id`),
`raw_output`, `finish_reason`, `status`를 기록합니다.
`status`는 `ok`, `invalid_output`(JSON/스키마 오류), `truncated`(출력 길이 초과),
`error`(추론 실패)입니다. 실패 원문과 오류를 보존하고 완료된 배치마다 저장합니다.
실패가 있으면 종료 코드는 1입니다. `ok`는 응답 형식 검증 통과를 뜻하며,
정답 여부를 채점하지는 않습니다. `--dry-run`은 결과 파일이나 가짜 정답을 생성하지 않습니다.

코드 흐름: `data.py`에서 데이터와 이미지 로딩 → `build_prompt`에서 이미지의
base64 data URL 구성 → `runner.py`에서 기존 vLLM 엔진으로 배치 추론 →
응답 파싱 및 검증 → JSONL 저장. CLI 종료 시 엔진 워커를 정리합니다.
이미지 메시지 형식은 [vLLM 공식 문서](https://docs.vllm.ai/en/v0.31.0/features/multimodal_inputs/)를 따릅니다.

```powershell
python -m unittest discover -s tests -v
```

공통 system 프롬프트는 `prompts/physics_solver.txt`에서 읽습니다. 모든 모델의
기본 입력에 적용되며 `--prompt 파일경로`로 교체할 수 있습니다. 문제별 user
메시지에도 허용된 선택지 ID 또는 주관식의 `null` 규칙을 덧붙입니다.

```bash
python -m src.simulation --config config/qwen2.5-7b.yaml --prompt prompts/physics_solver.txt --output outputs/qwen2.5-7b-run-02.jsonl
```

기존 `qwen2.5-7b-run-01.jsonl`의 101건 중 `invalid_output` 37건은
역슬래시 JSON 오류 23건, 선택지 ID 오류 14건입니다. 공통 프롬프트는
일반 텍스트 또는 JSON 이스케이프를 적용한 LaTeX 수식, 정확한 선택지 ID와 일치하는 정답 내용,
1~8개 핵심 풀이 단계 및 반복 금지를 명시합니다. 기존 결과와 파싱 규칙은
보존하며, 프롬프트 개선 효과는 실제 재추론 결과로 확인해야 합니다.

파서는 JSON 문자열 내부의 이스케이프되지 않은 LaTeX 역슬래시를 보정합니다.
이미 이스케이프된 역슬래시, 큰따옴표, Unicode 및 제어문자 escape는 유지합니다.
특히 `\frac`, `\theta` 등의 명령은 JSON 디코딩 전에 처리하여 제어문자로
변환되는 것을 방지합니다. 단일 역슬래시 뒤 여러 영문자가 오면 LaTeX 명령으로
해석하므로, 실제 줄바꿈이나 탭 바로 뒤에 영문자가 오는 모호한 표현은 피하고
정확한 JSON 이스케이프 또는 실제 Unicode 문자를 사용하는 것이 좋습니다.
미완성 JSON과 잘못된 선택지 ID는 계속 실패로 처리하며 `raw_output`은 보존합니다.

기존 LaTeX 보정 파싱이 JSON 문법 오류로 실패하면, 문자열 안 실제 LF/CR
줄바꿈을 JSON escape로 바꿔 재시도합니다. 완성된 JSON 뒤에 닫는 괄호
하나만 추가된 경우에도 이를 제거하여 재시도합니다. 여러 초과 괄호,
추가 객체, 미완성 응답, 잘못된 선택지 ID는 계속 오류로 처리합니다.
