# 정답 기준 입력과 Gemini / OpenRouter Judge

## 정답 기준 입력

```bash
python -m src.evaluation serve --port 8010
```

브라우저에서 http://127.0.0.1:8010 을 엽니다. 원격 실행이라면 VS Code에서 8010 포트를 전달합니다.
왼쪽 목록에서 문항을 클릭하세요. 원본 annotation은 수정하지 않습니다.

- 객관식: 기존 answer.choice_id를 사용하므로 목록에 자동 완료로 표시합니다. 추가 입력·저장이 필요 없습니다.
- 주관식: 먼저 수치/수식을 선택합니다. 정답 번호를 입력하지 않습니다.
- 수치: 기준값, 기준 단위, 단위 표기 필수 여부, 절대 허용오차를 지정합니다. 상대오차는 사용하지 않습니다.
- 추가 허용 단위: 예를 들어 기준이 -200 J ±1 J이면 kJ와 배율 1000을 등록합니다.
  -0.2 kJ ±0.001 kJ에 해당하는 값이 화면에 표시됩니다. 배율은 `1 허용 단위 = 배율 × 기준 단위`입니다.
  J와 kJ처럼 배율로 변환되는 단위만 지원하며 섭씨/켈빈처럼 상수가 더해지는 변환은 이 필드로 표현하지 않습니다.
- 수식: 기준 수식과 추가 인정 수식만 입력합니다. 변수 가정은 입력하지 않습니다.
- 답이 여러 개인 문항만 ‘정답이 여러 개인 문제만 추가’를 펼칩니다.

추가 기준은 dataset/evaluation/문제ID.yaml에 저장합니다. 형식 버전은 2입니다.
기존 버전 1 파일은 읽을 때 고급 설정·가정을 제거하고 객관식은 자동 완료로 취급합니다.
이전 상대오차가 0이 아니었던 주관식 기준은 절대오차 재확인을 위해 작성 중으로 표시합니다.
파일 자체는 저장할 때 갱신합니다. 내부 parts.id는 저장 호환용이며 학생 입력 항목이 아닙니다.

```bash
python -m src.evaluation init
python -m src.evaluation validate
```

## 최종 정답률 평가: 규칙 또는 LLM 선택

config/evaluation/judge.yaml에서 선택합니다. 이 설정은 최종 정답 평가에만 적용합니다.
물리식 설정 정확도와 관계 F1은 config에서 선택한 Gemini 또는 OpenRouter Judge를 사용합니다.

```yaml
final_answer:
  method: rules  # rules 또는 llm
```

```bash
# 규칙의 수식 동치 검사 의존성 (수치/객관식만 평가하면 불필요)
python -m pip install -e '.[evaluation]'
# LLM 방식을 쓸 때
python -m pip install -e '.[gemini]'

python -m src.evaluation judge --task answer \
  --predictions outputs/qwen2.5-3b.jsonl \
  --problem-id P_0207bb07aba04054 \
  --judge-config config/evaluation/judge.yaml \
  --output outputs/final-answer-01
```

`--problem-id` 생략 시 전체 annotation을 평가하며 누락된 모델 응답도 분모에 포함합니다.
선택한 범위의 모든 주관식 기준이 작성 완료여야 실행됩니다. 객관식은 기준 YAML이 필요 없습니다.

- 객관식은 method와 관계없이 choice_id를 정확히 비교하며 API를 호출하지 않습니다.
- rules는 API 없이 answer.text에서 수치·단위를 분리해 등록 단위로 환산하고 절대오차를 비교합니다.
  필수 단위 누락·미등록 단위는 오답입니다. 단위가 선택이면 단위 없는 값은 기준 단위로 취급합니다.
- 수식은 제한된 문법을 AST로 읽어 SymPy로 동치성을 확인합니다. 입력 문자열을 eval하지 않습니다.
  덧셈·곱셈 항 순서, 전개·인수분해, 단일 ±/∓의 해집합, 추가 인정식을 지원합니다.
  `a*b`, `sqrt(x)`, `sin(x)`, `a^2`, `\frac{a}{b}`, `\sqrt{x}` 등을 사용하세요.
  임의의 LaTeX, 일반 방정식, 연쇄 등식, 암묵적 곱셈 `2R` 등은 지원하지 않습니다.
  각 식 검사는 별도 프로세스에서 최대 5초 실행합니다. 해석·동치 여부가 불명확하면 needs_review입니다.
- llm은 question, annotation answer, 동일한 저장 기준, 제출 answer만 전달합니다.
  모델 풀이·모델명은 전달하지 않습니다. 수치 오차·허용 단위를 완화하지 않도록 안내합니다.
  판정 원문·근거 인용·입력 기준을 보존하지만 Judge의 의미 판정이 항상 정확하다는 보장은 없습니다.
- 규칙 판정이 어려워도 자동으로 LLM을 호출하지 않습니다. method를 바꾸고 새 출력 경로로 재실행하세요.
- 여러 정답은 answer.text에 `(1) ...; (2) ...`처럼 기준 항목과 대응되게 작성합니다.
  선택적으로 answer.parts의 `[{id: "1", text: "..."}, ...]`도 규칙 채점에서 읽습니다.
  모든 정답 항목이 맞아야 문항 정답입니다.

기존 시뮬레이션 출력 형식은 유지합니다. 수치는 answer.text에 값과 단위를 함께 쓰고
수식은 최종식을 씁니다. raw_output의 boxed를 탐색하거나 풀이에서 답을 추측하지 않습니다.
answer.text 안의 단일 `\boxed{...}` 표기는 규칙 채점에서 정리합니다.

summary.json에는 전체 accuracy와 by_question_type의 객관식·주관식 정확도가 저장됩니다.
판정 불가 또는 평가 오류가 있으면 전체 accuracy는 null이며, 해당 항목이 없는 유형의
정확도는 별도로 표시합니다. 모델 응답 실패·누락은 오답으로 계산합니다.
주관식 기준은 manifest에 스냅샷으로 저장하며 선택한 method도 기록합니다.

## OpenRouter 설정

```bash
python -m pip install -e '.[openrouter,evaluation]'
python -m src.evaluation judge --task physics \
  --predictions outputs/qwen2.5-3b.jsonl \
  --problem-id P_0207bb07aba04054 \
  --judge-config config/evaluation/judge-openrouter.yaml \
  --output outputs/judge-openrouter-sample
```

`.env`에 `OPENROUTER_API_KEY`를 설정합니다. Gemini 키는 필요 없습니다.
`judge`, `extract` 모두 이 config를 사용할 수 있으며 `answer`, `physics`, `relations`에 적용됩니다.
최종 정답을 LLM으로 평가하려면 `final_answer.method: llm`으로 설정하세요.
모델은 이미지 입력과 JSON Schema 구조화 출력을 지원해야 합니다.
`max_output_tokens`, `timeout`, `extra_body`를 설정할 수 있습니다.
`thinking_level`은 Gemini 전용이며 OpenRouter 추론 옵션은 `extra_body.reasoning`을 사용합니다.

## Gemini 설정

```bash
python -m pip install -e '.[gemini]'
```

프로젝트 루트 .env에 GEMINI_API_KEY를 설정합니다. 기존 환경변수가 우선합니다.
config/evaluation/judge.yaml의 모델과 설정을 사용하며 실제 호출은 비용이 발생합니다.
Judge 코드와 프롬프트는 judge.py에 있고 기존 serving 모듈에 의존하지 않습니다.

## 물리식 설정 정확도: 시뮬레이션과 annotation 직접 비교

```bash
python -m src.evaluation judge --task physics \
  --predictions outputs/qwen2.5-3b.jsonl \
  --problem-id P_0207bb07aba04054 \
  --output outputs/judge-physics-01
```

추가 기준 YAML, 핵심 단계 선택, 평가 대상 지정은 필요 없습니다.
기존 annotation의 solution_steps 전체와 모델 solution_steps를 비교합니다.
Judge는 각 기준 단계에서 식의 물리적 설정을 평가하고 계산 결과만 있는 단계는 제외합니다.
법칙 이름 언급만으로 정답 처리하지 않고 문제 조건에 적용된 식·구체적 관계를 요구합니다.
산술 오류와 대안 풀이를 구분합니다. 평가할 식 설정이 전혀 없으면 needs_review로 남깁니다.
이는 사전 선정한 핵심식 평가와 다른 규약이므로 기존 결과와 혼합하지 마세요.

## 관계 F1: 시뮬레이션 풀이와 annotation 직접 비교

```bash
python -m src.evaluation judge --task relations \
  --predictions outputs/qwen2.5-3b.jsonl \
  --problem-id P_0207bb07aba04054 \
  --output outputs/judge-relations-01
```

관계 검수 체크나 기준 YAML 없이 실행합니다. 일반 시뮬레이션 결과라면:

1. 모델 solution_steps만 별도 Judge 호출에 제공해 실제로 명시한 그림 관계를 추출합니다.
   기준 annotation은 이 추출 호출에 제공하지 않습니다. 추출 description은 풀이의 실제 인용이어야 합니다.
2. 기존 entities·visual_information과 추출 관계를 Judge로 일대일 매칭합니다.
3. 코드가 TP/FP/FN을 합산하여 전체 precision/recall/F1을 계산합니다.

즉 이 지표는 **풀이에 나타난 관계의 F1**입니다. 모델이 인식했지만 풀이에 적지 않은 관계는 복원하지 않습니다.
원본 그림만 제공하는 별도 실험의 추출 성능과 구분하세요. 일반 시뮬레이션 문항당 최대 두 번 API를 호출합니다.
이미 entities·relations가 있는 별도 추출 파일도 입력할 수 있습니다. 이때 추출 호출은 생략합니다.

별도 그림 전용 추출 실험도 유지합니다:

```bash
python -m src.evaluation extract --problem-id P_0207bb07aba04054 --output outputs/relations-01.jsonl
```

--problem-id는 반복 가능하며 생략하면 전체 annotation을 평가합니다. 모델 결과가 누락된 대상도 분모에 포함합니다.

## 결과

매번 새로운 출력 경로를 사용하세요. manifest.json에는 설정·프롬프트·입력 해시·관계 입력 경로를,
results.jsonl에는 판정·원문·근거·사용량·관계 추출 중간 결과를, summary.json에는 점수를 저장합니다.
모델 응답 실패·누락은 오답/FN입니다. Judge 오류나 기준 오류는 최종 점수를 null로 남깁니다.
분모가 0인 지표도 null입니다. API 오류(429, 주요 5xx)만 최대 3회 시도합니다.
자동 재개·수동 override·교차 Judge 자동화는 제공하지 않습니다.
먼저 사람 판정 표본과 비교하고 전체 실험 동안 모델·프롬프트·기준을 고정하세요.

```bash
python -m unittest discover -s tests -v
# jsdom이 있는 개발 환경에서 UI 회귀 검증
node tests/test_evaluation_editor.cjs
```
