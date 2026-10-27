# Physics-Bench

그림이 포함된 물리 문제를 모델로 풀이하고, annotation과 비교해 정답률·물리식 설정 정확도·관계 F1을 평가하는 프로젝트입니다. 아래 명령은 모두 **프로젝트 루트**에서 실행합니다.

## 1. 폴더 구조

```text
Physics-Bench/
├── config/                 # 모델별 시뮬레이션 설정
│   └── evaluation/judge.yaml # 평가 모델 및 정답 판정 방식
├── config_temp/            # 추가·실험용 모델 설정
├── dataset/
│   ├── annotations/        # 문제별 원본 annotation
│   ├── annotations.jsonl   # 시뮬레이션 기본 입력
│   ├── images/             # 문제 이미지
│   └── evaluation/         # 주관식 정답·단위·허용오차 YAML
├── src/
│   ├── simulation/         # 모델 풀이 생성 CLI
│   ├── evaluation/         # 평가 CLI 및 정답 기준 HTML 편집기
│   └── serving/            # 모델/API 서빙 모듈
├── prompts/                # 공통 풀이 프롬프트
├── outputs/                # 결과 JSONL 및 HTML viewer
├── scripts/                # viewer 서버 등 실행 도구
├── tests/                  # 테스트
├── docs/                   # 참고 문서
├── .env.example            # API 키 설정 예시
└── pyproject.toml          # Python 버전 및 의존성
```

## 2. 가상환경 및 라이브러리 설치

Python **3.11 이상**을 사용합니다. Linux/macOS 기준:

```bash
python3 -m venv benchvenv
source benchvenv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[openrouter,gemini,evaluation]"
```

Windows PowerShell에서는 `benchvenv\Scripts\Activate.ps1`로 활성화합니다.

위 설치는 OpenRouter 시뮬레이션, Gemini 평가, 수식 동치 검사를 포함합니다. 로컬 GPU 추론을 사용한다면 Linux/CUDA 환경에서 추가 설치합니다.

```bash
python -m pip install -e ".[vllm]"
```

기본 데이터 검증만 필요하면 `python -m pip install -e .`로 충분합니다. 의존성은 `pyproject.toml`에서 관리합니다.

## 3. .env 설정

```bash
cp .env.example .env
```

`.env`에 사용할 서비스의 실제 키를 입력합니다.

```dotenv
OPENROUTER_API_KEY=실제_OpenRouter_키
GEMINI_API_KEY=실제_Gemini_키
```

시뮬레이션은 OpenRouter를, LLM 평가는 Gemini 또는 OpenRouter를 사용할 수 있습니다. 기존 환경변수가 `.env`보다 우선하며, 실제 키는 커밋하지 않습니다. API 호출에는 비용이 발생합니다.

## 4. config로 시뮬레이션 실행

`config/`의 YAML에서 `model_path`, `sampling`을 설정합니다. `provider: openrouter`는 API를 사용하며, provider를 생략하면 로컬 vLLM을 사용합니다. vLLM 설정의 `gpu_ids` 등은 실행 환경에 맞게 수정하세요.

```bash
# 추론 없이 데이터·이미지·프롬프트 구성 확인
python -m src.simulation --dry-run

# 샘플: 앞의 2문제
python -m src.simulation \
  --config config/openrouter-gpt-5-mini.yaml \
  --limit 2 --batch-size 1 \
  --output outputs/gpt-5-mini-sample.jsonl

# 일괄: 전체 데이터셋
python -m src.simulation \
  --config config/openrouter-gpt-5-mini.yaml \
  --batch-size 1 \
  --output outputs/gpt-5-mini-full.jsonl
```

로컬 추론은 `--config config/qwen2.5-7b.yaml`처럼 설정 파일을 교체합니다. 특정 문제만 실행하려면 `--limit 2` 대신 `--problem-id P_0207bb07aba04054`를 사용합니다. `--problem-id`는 반복 지정할 수 있습니다.

기본 입력은 `dataset/annotations.jsonl`, 공통 프롬프트는 `prompts/physics_solver.txt`입니다. `--dataset`, `--prompt`로 교체할 수 있습니다. 결과는 문제당 한 줄의 JSONL로 저장하며 풀이(`solution_steps`), 정답(`answer`), 원문 및 실행 상태를 포함합니다. `status: ok`는 출력 형식 통과를 뜻하며 정답 판정은 평가 단계에서 수행합니다. 기존 결과를 덮어쓰지 않으므로 재실행할 때 새 출력 파일명을 사용하세요.

## 5. HTML에서 정답 허용오차·단위 설정

**주관식 최종 정답 평가 전에** HTML 편집기로 기준을 작성합니다.

```bash
python -m src.evaluation serve --port 8010
```

브라우저에서 <http://127.0.0.1:8010>에 접속합니다. 원격 서버라면 VS Code 등에서 8010 포트를 전달하세요.

1. 왼쪽 목록에서 문제를 선택하고 기존 annotation 정답을 확인합니다.
2. 주관식은 **수치/수식**을 선택합니다.
3. 수치는 **기준값, 기준 단위, 단위 표기 필수 여부, 절대 허용오차**를 입력합니다. 상대오차는 사용하지 않습니다.
4. 추가 허용 단위가 있으면 단위와 배율을 등록합니다. 예: 기준이 `-200 J`, 허용오차가 `1 J`라면 `kJ`의 배율을 `1000`으로 등록합니다. 이는 `-0.2 kJ ±0.001 kJ`에 해당합니다.
5. 수식은 기준 수식과 추가 인정 수식을 입력합니다. 예: `a*b`, `sqrt(x)`, `a^2`. 답이 여러 개라면 추가 항목을 등록합니다.
6. 상태를 **정답 기준 작성 완료**로 바꾸고 **저장**합니다.

배율은 `1 허용 단위 = 배율 × 기준 단위`입니다. 섭씨↔켈빈처럼 상수를 더해야 하는 변환은 지원하지 않습니다. 객관식은 annotation의 `answer.choice_id`로 판정하므로 추가 설정이 필요 없습니다.

설정은 `dataset/evaluation/문제ID.yaml`에 저장하며 원본 annotation은 수정하지 않습니다. 평가 범위에 포함된 **모든 주관식**의 기준이 작성 완료여야 정답 평가를 실행할 수 있습니다.

## 6. judge config로 evaluation 실행

`config/evaluation/judge.yaml`에서 평가 모델과 생성 설정, 최종 정답 판정 방식을 지정합니다.

```yaml
final_answer:
  method: rules  # rules: 규칙 판정 / llm: 선택한 provider의 LLM 판정
```

`rules`는 수치·단위를 환산해 절대오차를 비교하고, 수식은 SymPy로 동치성을 검사합니다. `llm`은 저장한 정답 기준을 선택한 provider에 전달합니다. 객관식은 두 방식 모두 선택지 ID를 정확히 비교합니다.

```bash
# 샘플 평가: 시뮬레이션 결과에 있는 특정 문제 지정
python -m src.evaluation judge --task answer \
  --predictions outputs/gpt-5-mini-sample.jsonl \
  --problem-id P_0207bb07aba04054 \
  --judge-config config/evaluation/judge.yaml \
  --output outputs/eval-answer-sample

# 일괄 평가: 전체 annotation 기준
python -m src.evaluation judge --task answer \
  --predictions outputs/gpt-5-mini-full.jsonl \
  --judge-config config/evaluation/judge.yaml \
  --output outputs/eval-answer-full
```

샘플 명령의 ID는 실제 샘플 결과에 있는 `problem_id`로 바꾸세요. 여러 문제는 `--problem-id`를 반복합니다. 평가에는 `--limit` 옵션이 없으며, **ID를 생략하면 전체 annotation이 평가 대상**입니다. 누락되거나 실패한 모델 응답도 오답으로 포함합니다.

다른 평가 지표는 위 명령의 `--task`와 출력 경로를 바꿔 실행합니다. 샘플·일괄 범위 지정 방식은 동일합니다.

| task | 평가 내용 | 판정 방식 |
| --- | --- | --- |
| `answer` | 최종 정답률 | config의 `rules` 또는 `llm` |
| `physics` | annotation 풀이 대비 물리식 설정 정확도 | Gemini / OpenRouter |
| `relations` | 모델 풀이에 명시된 그림 관계의 precision/recall/F1 | Gemini / OpenRouter |

`physics`, `relations`는 주관식 정답 기준 YAML 없이 원본 annotation과 비교하며, `final_answer.method`와 관계없이 선택한 provider의 LLM을 사용합니다.

결과 디렉터리에는 `summary.json`(점수), `results.jsonl`(문항별 판정), `manifest.json`(설정·입력 기록)이 저장됩니다. 매번 새로운 출력 경로를 사용하세요. 판정 불가나 평가 오류가 있으면 최종 점수는 `null`로 남으므로 결과 상태도 확인합니다.

OpenRouter로 평가하려면 `--judge-config config/evaluation/judge-openrouter.yaml`로 바꾸고 `.env`의 `OPENROUTER_API_KEY`를 설정하세요. Gemini 키는 필요하지 않습니다. 최종 정답도 LLM으로 판정하려면 해당 config의 `final_answer.method`를 `llm`으로 바꿉니다.

```yaml
provider: openrouter
model: openai/gpt-5-mini
api_key_env: OPENROUTER_API_KEY
max_output_tokens: 8192
final_answer:
  method: llm
```

이미지 입력 및 [구조화 출력](https://github.com/OpenRouterTeam/docs/blob/main/guides/features/structured-outputs.mdx)을 지원하는 모델을 사용하세요. `thinking_level`은 Gemini 전용이며, OpenRouter의 추론 옵션은 `extra_body.reasoning`에서 설정합니다.

## 7. 시뮬레이션 결과를 HTML viewer로 확인

```bash
# outputs/ 바로 아래의 JSONL을 모아 viewer 생성·갱신
python outputs/build_html_viewer.py

# viewer 및 문제 이미지 서버 실행
python scripts/serve_html_viewer.py --port 8004
```

브라우저에서 <http://127.0.0.1:8004>에 접속해 모델별 풀이와 정답을 확인합니다. 결과를 추가했다면 viewer를 다시 생성하고 브라우저를 새로고침하세요. 원격 서버에서는 8004 포트를 전달합니다.

다른 폴더의 결과를 보려면 `python outputs/build_html_viewer.py --input-dir temp_outputs`처럼 지정합니다.

상세 설명: [시뮬레이션](src/simulation/README.md) · [평가 및 정답 기준](src/evaluation/README.md) · [결과 viewer](outputs/README.md)
