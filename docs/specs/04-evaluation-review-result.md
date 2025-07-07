# Selvage 평가 에이전트 - 평가(4단계)

### 3단계: DeepEval 변환 및 평가

#### 목표
Selvage 결과를 DeepEval 형식으로 변환하고 평가

#### DeepEvalTestCaseConverterTool 인터페이스 명세

**클래스 구조:**
- `Tool` 클래스를 상속하여 구현
- 기존 `FileExistsTool` 등과 동일한 패턴 적용
- `ToolGenerator` 클래스에 등록 필요

**주요 기능:**
- review_logs 디렉토리 구조 탐색
- 리뷰 로그 파일에서 prompt와 review_response 추출
- DeepEval 테스트 케이스 형식으로 변환
- 변환된 테스트 케이스 파일 저장

**파일 저장 경로:**
```
~/Library/selvage-eval/deep_eval_test_case/{session_id}/{model_name}/
```

**DeepEval 테스트 케이스 형식:**
```json
{
  "input": "review_log의 prompt (JSON 문자열)",
  "actual_output": "review_log의 review_response (JSON 문자열)",
  "expected_output": null
}
```

**메타데이터 파일 생성:**
- 경로: `~/Library/selvage-eval/deep_eval_test_case/{session_id}/metadata.json`
- 구조:
  ```json
  {
    "selvage_version": "현재 selvage 버전",
    "execution_date": "평가 실행 날짜"
  }
  ```

#### 평가 메트릭

DeepEval을 사용한 4개 핵심 메트릭으로 Selvage 리뷰 품질을 정량화합니다.

**1. Correctness (정확성) - 임계값: 0.7**
```python
correctness = GEval(
    name="Correctness",
    model="gemini-2.5-pro",
    evaluation_steps=[
        "Verify that all pertinent issues (e.g., bugs, security vulnerabilities, performance issues, significant style/design flaws) found in the input code are reported in the 'issues' array.",
        "If the 'issues' array in the output is empty, critically assess the input code to confirm this emptiness is justified by an actual absence of pertinent issues, not a failure of detection.",
        "If issues are reported, check if their specified filenames and line numbers are accurate.",
        "If issues are reported, evaluate if their identified types (bug, security, performance, style, design) are appropriate for the code.",
        "If issues are reported, confirm if their severity levels (info, warning, error) are assigned according to the actual impact of each issue.",
        "If issues are reported, review if their descriptions accurately and factually reflect the impact of the code changes.",
        "If the 'issues' array is legitimately empty (because no pertinent issues exist in the input code), verify that the 'summary' appropriately states this, aligning with system prompt guidelines.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    threshold=0.7,
)
```

**2. Clarity (명확성) - 임계값: 0.7**
```python
clarity = GEval(
    name="Clarity",
    model="gemini-2.5-pro",
    evaluation_steps=[
        "Evaluate whether the overall code review output (including summary, and if present, issue descriptions, suggestions, and recommendations) uses concise and direct language.",
        "Assess whether issue descriptions and suggestions and recommendations are specific and clear.",
        "Review if the purpose and intent of code changes are clearly understandable.",
        "Verify that improved code examples are provided and easy to understand.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)
```

**3. Actionability (실행가능성) - 임계값: 0.7**
```python
actionability = GEval(
    name="Actionability",
    model="gemini-2.5-pro",
    evaluation_steps=[
        "Check if specific solutions are provided for each issue.",
        "Evaluate whether the suggested improvements are practically implementable.",
        "Review if code improvement examples are specific enough to be integrated into the actual codebase.",
        "Assess whether the suggestions can bring substantial improvements in terms of code quality, performance, security, etc.",
        "Confirm that overall recommendations are actionable within the project context.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    threshold=0.7,
)
```

**4. JsonCorrectnessMetric (JSON 스키마 검증)**
```python
jsoncorrectness = JsonCorrectnessMetric(
    expected_schema=StructuredReviewResponse(
        issues=[
            StructuredReviewIssue(
                type="",           # 이슈 유형 (버그, 보안, 성능, 스타일, 설계)
                line_number=0,     # 라인 번호
                file="",           # 파일 경로
                description="",    # 이슈 설명
                suggestion="",     # 개선 제안
                severity=IssueSeverityEnum.INFO,  # 심각도 (info, warning, error)
                target_code="",    # 문제가 있는 원본 코드
                suggested_code="", # 개선된 코드
            )
        ],
        summary="",           # 전체 리뷰 요약
        score=0,             # 0-10 품질 점수
        recommendations=[],   # 전반적인 권장사항 목록
    ),
    model="gemini-2.5-pro-preview-05-06",
    include_reason=True,
)

# Structured Outputs용 스키마 클래스 (기본값 없음)
class IssueSeverityEnum(str, Enum):
    """이슈 심각도 열거형"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class StructuredReviewIssue(BaseModel):
    """Structured Outputs용 코드 리뷰 이슈 모델"""

    type: str
    line_number: int | None
    file: str | None
    description: str
    suggestion: str | None
    severity: IssueSeverityEnum
    target_code: str | None  # 리뷰 대상 코드
    suggested_code: str | None  # 개선된 코드


class StructuredReviewResponse(BaseModel):
    """Structured Outputs용 코드 리뷰 응답 모델"""

    issues: list[StructuredReviewIssue]
    summary: str
    score: float | None
    recommendations: list[str]
```

#### DeepEvalExecutorTool 인터페이스 명세

**클래스 구조:**
- `Tool` 클래스를 상속하여 구현
- 기존 `ExecuteSafeCommandTool` 등과 연동하여 DeepEval CLI 실행
- `ToolGenerator` 클래스에 등록 필요

**주요 기능:**
- DeepEval 테스트 케이스 데이터셋 로딩
- 4개 핵심 메트릭으로 평가 실행 (Correctness, Clarity, Actionability, JsonCorrectness)
- 병렬 처리 지원 및 필터링 옵션 제공
- 평가 결과 수집 및 저장

**파일 저장 경로:**
```
~/Library/selvage-eval/deepeval_results/{session_id}/{repo_name}/{model_name}/
```

**환경 설정 요구사항:**
```bash
export DEEPEVAL_RESULTS_FOLDER="~/Library/selvage-eval/deepeval_results"
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

**메타데이터 파일 생성:**
- 경로: `~/Library/selvage-eval/deepeval_results/{session_id}/metadata.json`
- 구조:
  ```json
  {
    "selvage_version": "현재 selvage 버전",
    "execution_date": "평가 실행 날짜",
    "deep_eval_test_case_path": "테스트 케이스 파일 경로"
  }
  ```

**실행 옵션:**
- 기본 실행: `deepeval test run {test_file_path}`
- 병렬 실행: `-n {workers}` 옵션 사용
- 필터링: `-d {filter}` 옵션으로 실패/성공 테스트만 표시

#### ToolGenerator 통합 요구사항

**새로운 Tool 클래스 등록:**
- `DeepEvalTestCaseConverterTool`을 `ToolGenerator` 클래스에 추가
- `DeepEvalExecutorTool`을 `ToolGenerator` 클래스에 추가
- 기존 Tool 패턴과 일관성 유지

#### 파일 저장 구조 및 메타데이터 관리

**디렉토리 구조:**
```
~/Library/selvage-eval/
├── deep_eval_test_case/
│   └── {session_id}/
│       ├── metadata.json
│       └── {repo_name}/
│           └── {model_name}/
│               └── test_cases.json
└── deepeval_results/
    └── {session_id}/
        ├── metadata.json
        └── {repo_name}/
            └── {model_name}/
                └── evaluation_results.json
```

**메타데이터 스키마:**
- `selvage_version`: 현재 실행 중인 Selvage 바이너리 버전
- `execution_date`: ISO 8601 형식의 평가 실행 날짜/시간
- `deep_eval_test_case_path`: (DeepEvalExecutor용) 참조하는 테스트 케이스 파일 경로

#### 결과 확인 및 해석

**결과 저장 위치:**
- 로컬 결과: `~/Library/selvage-eval/deepeval_results/` 디렉토리
- 콘솔 출력: 각 메트릭별 점수 및 통과/실패 상태
- 실행 로그: subprocess 출력을 통한 상세 결과

**메트릭 점수 해석:**
- **0.7 이상**: 통과 (양질의 리뷰)
- **0.5-0.7**: 보통 (개선 필요)
- **0.5 미만**: 실패 (심각한 문제)