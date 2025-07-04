
## 문서 작성 요구사항
아래 요구사항에 맞게 현재 문서를 수정해주세요.

1. DeepEvalTestCaseConverterTool
  - 아래 '#### 변환 스키마'의 구조를 따름
  - 파일 저장 경로: ~/Library/selvage-eval/deep_eval_test_case/{session_id}/{repo_name}/{model_name}
  - Tool을 확장한 클래스여야하며 기존의 'FileExistsTool' 등을 참고할 것.
  - ToolGenerator 클래스에도 추가할 것
  - {session_id} 디렉토리에는 metadata.json 파일을 생성할 것.
    - 파일 저장 경로: ~/Library/selvage-eval/deepeval_results/{session_id}/metadata.json
    - 파일 내용:
      - selvage_version: 현재 selvage 버전
      - execution_date: 평가 실행 날짜
2. DeepEvalExecutorTool
  - 아래 '#### 평가 실행 코드'의 구조를 따름
  - 파일 저장 경로: ~/Library/selvage-eval/deepeval_results/{session_id}/{repo_name}/{model_name}
  - Tool을 확장한 클래스여야하며 기존의 'FileExistsTool' 등을 참고할 것.
  - ToolGenerator 클래스에도 추가할 것
  - {session_id} 디렉토리에는 metadata.json 파일을 생성할 것.
    - 파일 저장 경로: ~/Library/selvage-eval/deepeval_results/{session_id}/metadata.json
    - 파일 내용:
      - selvage_version: 현재 selvage 버전
      - execution_date: 평가 실행 날짜
      - deep_eval_test_case_path: 테스트 케이스 파일 경로

---------
# Selvage 평가 에이전트 - 평가(4단계)

### 3단계: DeepEval 변환 및 평가

#### 목표
Selvage 결과를 DeepEval 형식으로 변환하고 평가

#### 변환 스키마
# DeepEval Test case 구조
**주요 필드 설명:**
-  input: review_log의 prompt와 대응됨
-  actual_output: review_log의 review_response와 대응됨
-  expected_output: None(현재 사용하지 않음)

```python
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


def scan_review_logs(base_path: str = "~/Library/selvage-eval-agent/review_logs") -> List[Dict[str, Any]]:
    """
    review_logs 디렉토리 구조를 탐색하여 모든 리뷰 로그 파일을 찾는다.
    
    Args:
        base_path: 리뷰 로그 기본 경로
    
    Returns:
        리뷰 로그 파일 정보 리스트 (repo_name, commit_id, model_name, file_path 포함)
    """
    review_logs = []
    base_path = Path(base_path).expanduser()
    
    if not base_path.exists():
        print(f"경로가 존재하지 않습니다: {base_path}")
        return review_logs
    
    # repo_name 폴더 순회
    for repo_dir in base_path.iterdir():
        if not repo_dir.is_dir():
            continue
            
        repo_name = repo_dir.name
        
        # commit_id 폴더 순회
        for commit_dir in repo_dir.iterdir():
            if not commit_dir.is_dir():
                continue
                
            commit_id = commit_dir.name
            
            # model_name 폴더 순회
            for model_dir in commit_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                model_name = model_dir.name
                
                # 리뷰 로그 파일 찾기 (.json 파일)
                for log_file in model_dir.glob("*.json"):
                    review_logs.append({
                        "repo_name": repo_name,
                        "commit_id": commit_id,
                        "model_name": model_name,
                        "file_path": log_file,
                        "file_name": log_file.name
                    })
    
    return review_logs


def extract_prompt_and_response(log_file_path: Path) -> Optional[Dict[str, Any]]:
    """
    리뷰 로그 파일에서 prompt와 review_response를 추출한다.
    
    Args:
        log_file_path: 리뷰 로그 파일 경로
    
    Returns:
        추출된 prompt와 review_response 데이터
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        # prompt와 review_response 필드 추출
        prompt = log_data.get("prompt", [])
        review_response = log_data.get("review_response", {})
        
        if not prompt or not review_response:
            print(f"필수 필드가 없습니다: {log_file_path}")
            return None
        
        return {
            "prompt": prompt,
            "review_response": review_response,
            "original_data": log_data  # 원본 데이터도 포함
        }
        
    except Exception as e:
        print(f"파일 읽기 오류 {log_file_path}: {e}")
        return None


def convert_to_deepeval_format(repo_name: str, model_name: str, 
                             extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    추출된 데이터를 DeepEval 테스트 케이스 형식으로 변환한다.
    
    Args:
        repo_name: 리포지토리 이름
        model_name: 모델 이름
        extracted_data: 추출된 prompt와 response 데이터
    
    Returns:
        DeepEval 형식의 테스트 케이스
    """
    return {
        "input": json.dumps(extracted_data["prompt"]),
        "actual_output": json.dumps(extracted_data["review_response"])
    }


def save_deepeval_test_cases(repo_name: str, model_name: str, 
                           test_cases: List[Dict[str, Any]], 
                           output_dir: str = "~/Library/selvage-eval-agent/deep_eval_test_case") -> str:
    """
    DeepEval 테스트 케이스를 파일로 저장한다.
    
    Args:
        repo_name: 리포지토리 이름
        model_name: 모델 이름
        test_cases: 테스트 케이스 리스트
        output_dir: 출력 디렉토리
    
    Returns:
        저장된 파일 경로
    """
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 현재 시간을 기반으로 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_data_{timestamp}_{repo_name}_{model_name}.json"
    
    file_path = output_path / filename
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    
    print(f"테스트 케이스 저장 완료: {file_path}")
    return str(file_path)


def process_review_logs_to_deepeval():
    """
    전체 프로세스를 실행하는 메인 함수
    """
    print("리뷰 로그 스캔 시작...")
    review_logs = scan_review_logs()
    
    if not review_logs:
        print("리뷰 로그 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(review_logs)}개의 리뷰 로그 파일을 발견했습니다.")
    
    # repo_name과 model_name별로 그룹화
    grouped_logs = {}
    for log_info in review_logs:
        key = (log_info["repo_name"], log_info["model_name"])
        if key not in grouped_logs:
            grouped_logs[key] = []
        grouped_logs[key].append(log_info)
    
    # 각 그룹별로 테스트 케이스 생성
    for (repo_name, model_name), logs in grouped_logs.items():
        print(f"\n처리 중: {repo_name} - {model_name} ({len(logs)}개 파일)")
        
        test_cases = []
        for log_info in logs:
            extracted_data = extract_prompt_and_response(log_info["file_path"])
            if extracted_data:
                test_case = convert_to_deepeval_format(repo_name, model_name, extracted_data)
                test_cases.append(test_case)
        
        if test_cases:
            saved_file = save_deepeval_test_cases(repo_name, model_name, test_cases)
            print(f"  → {len(test_cases)}개 테스트 케이스 저장: {saved_file}")
        else:
            print(f"  → 유효한 테스트 케이스가 없습니다.")


# 실행 예시
if __name__ == "__main__":
    process_review_logs_to_deepeval()
```

#### 평가 메트릭

DeepEval을 사용한 4개 핵심 메트릭으로 Selvage 리뷰 품질을 정량화합니다.

**1. Correctness (정확성) - 임계값: 0.7**
```python
correctness = GEval(
    name="Correctness",
    model="gemini-2.5-pro",
    evaluation_steps=[
        "입력 코드에서 발견된 모든 관련 주요 이슈(버그, 보안 취약점, 성능 문제, 중대한 스타일/설계 결함)가 'issues' 배열에 보고되었는지 확인",
        "'issues' 배열이 비어있는 경우, 입력 코드를 비판적으로 평가하여 탐지 실패가 아닌 실제 이슈 부재인지 확인",
        "이슈가 보고된 경우, 파일명과 라인 번호의 정확성 확인",
        "이슈 유형(버그, 보안, 성능, 스타일, 설계)이 해당 코드에 적절한지 평가",
        "심각도 수준(info, warning, error)이 각 이슈의 실제 영향에 따라 적절히 할당되었는지 확인",
        "이슈 설명이 코드 변경의 영향을 정확하고 사실적으로 반영하는지 검토",
        "'issues' 배열이 정당하게 비어있는 경우, 'summary'가 시스템 프롬프트 가이드라인에 따라 적절히 명시하는지 확인"
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
        "전체 코드 리뷰 출력(요약, 이슈 설명, 제안, 권장사항)이 간결하고 직접적인 언어를 사용하는지 평가",
        "이슈 설명과 제안, 권장사항이 구체적이고 명확한지 평가",
        "코드 변경의 목적과 의도가 명확하게 이해 가능한지 검토",
        "개선된 코드 예시가 제공되고 이해하기 쉬운지 확인"
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
        "각 이슈에 대해 구체적인 해결책이 제시되었는지 확인",
        "제안된 개선 사항이 실제로 구현 가능한지 평가",
        "코드 개선 예시가 실제 코드베이스에 통합될 수 있을 만큼 구체적인지 검토",
        "제안이 코드 품질, 성능, 보안 등의 측면에서 실질적인 개선을 가져올 수 있는지 평가",
        "전반적인 권장사항이 프로젝트 맥락에서 실행 가능한지 확인"
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
```

**평가 실행 코드**

**1. 환경 설정**
```bash
# 결과 저장 경로 설정
export DEEPEVAL_RESULTS_FOLDER="~/Library/selvage-eval/deepeval_results"

# API 키 설정
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

**2. 완전한 테스트 파일 (test_code_review_evaluation.py)**
```python
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import pytest
from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.json_correctness import JsonCorrectnessMetric
from deepeval.dataset import EvaluationDataset

# 메트릭 정의 (문서의 위 섹션 참조)
correctness = GEval(
    name="Correctness",
    model="gemini-2.5-pro",
    evaluation_steps=[
        "입력 코드에서 발견된 모든 관련 주요 이슈(버그, 보안 취약점, 성능 문제, 중대한 스타일/설계 결함)가 'issues' 배열에 보고되었는지 확인",
        "'issues' 배열이 비어있는 경우, 입력 코드를 비판적으로 평가하여 탐지 실패가 아닌 실제 이슈 부재인지 확인",
        "이슈가 보고된 경우, 파일명과 라인 번호의 정확성 확인",
        "이슈 유형(버그, 보안, 성능, 스타일, 설계)이 해당 코드에 적절한지 평가",
        "심각도 수준(info, warning, error)이 각 이슈의 실제 영향에 따라 적절히 할당되었는지 확인",
        "이슈 설명이 코드 변경의 영향을 정확하고 사실적으로 반영하는지 검토",
        "'issues' 배열이 정당하게 비어있는 경우, 'summary'가 시스템 프롬프트 가이드라인에 따라 적절히 명시하는지 확인"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    threshold=0.7,
)

clarity = GEval(
    name="Clarity",
    model="gemini-2.5-pro",
    evaluation_steps=[
        "전체 코드 리뷰 출력(요약, 이슈 설명, 제안, 권장사항)이 간결하고 직접적인 언어를 사용하는지 평가",
        "이슈 설명과 제안, 권장사항이 구체적이고 명확한지 평가",
        "코드 변경의 목적과 의도가 명확하게 이해 가능한지 검토",
        "개선된 코드 예시가 제공되고 이해하기 쉬운지 확인"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

actionability = GEval(
    name="Actionability",
    model="gemini-2.5-pro",
    evaluation_steps=[
        "각 이슈에 대해 구체적인 해결책이 제시되었는지 확인",
        "제안된 개선 사항이 실제로 구현 가능한지 평가",
        "코드 개선 예시가 실제 코드베이스에 통합될 수 있을 만큼 구체적인지 검토",
        "제안이 코드 품질, 성능, 보안 등의 측면에서 실질적인 개선을 가져올 수 있는지 평가",
        "전반적인 권장사항이 프로젝트 맥락에서 실행 가능한지 확인"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    threshold=0.7,
)

jsoncorrectness = JsonCorrectnessMetric(
    model="gemini-2.5-pro-preview-05-06",
    include_reason=True,
)

def load_test_dataset(dataset_path: str) -> List[LLMTestCase]:
    """DeepEval 테스트 케이스 데이터셋 로딩"""
    dataset_path = Path(dataset_path).expanduser()
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    test_cases = []
    for item in test_data:
        test_case = LLMTestCase(
            input=item["input"],
            actual_output=item["actual_output"],
            expected_output=item.get("expected_output")
        )
        test_cases.append(test_case)
    
    return test_cases

def run_deepeval_test(test_file_path: str, parallel_workers: int = 1, display_filter: str = "all") -> bool:
    """DeepEval CLI 명령어를 ExecuteSafeCommandTool을 통해 실행하여 평가 수행"""
    from selvage_eval.tools.execute_safe_command_tool import ExecuteSafeCommandTool
    
    # ExecuteSafeCommandTool 인스턴스 생성
    # 주의: ExecuteSafeCommandTool의 allowed_commands에 'deepeval'이 추가되어야 함
    command_tool = ExecuteSafeCommandTool()
    
    # 기본 명령어 구성
    cmd_parts = ["deepeval", "test", "run", test_file_path]
    
    # 병렬 실행 옵션 추가
    if parallel_workers > 1:
        cmd_parts.extend(["-n", str(parallel_workers)])
    
    # 표시 필터 옵션 추가
    if display_filter != "all":
        cmd_parts.extend(["-d", display_filter])
    
    # 명령어 문자열로 결합
    command = " ".join(cmd_parts)
    
    try:
        print(f"DeepEval 실행 중: {command}")
        
        # ExecuteSafeCommandTool을 통해 CLI 명령어 실행
        result = command_tool.execute(
            command=command,
            timeout=300,  # 5분 타임아웃 (평가는 시간이 오래 걸릴 수 있음)
            capture_output=True
        )
        
        # 실행 결과 출력
        if result.success and result.data:
            stdout = result.data.get("stdout", "")
            stderr = result.data.get("stderr", "")
            returncode = result.data.get("returncode", -1)
            
            if stdout:
                print("=== DeepEval 실행 결과 ===")
                print(stdout)
            
            if stderr:
                print("=== 에러 출력 ===")
                print(stderr)
            
            print(f"평가 완료: {'성공' if returncode == 0 else '실패'} (exit code: {returncode})")
            return returncode == 0
        else:
            print(f"DeepEval 실행 실패: {result.error_message}")
            return False
        
    except Exception as e:
        print(f"평가 실행 중 오류 발생: {e}")
        return False

def run_code_review_evaluation():
    """코드 리뷰 평가 실행 메인 함수"""
    # 테스트 파일 경로 (pytest 기반 테스트 파일)
    test_file = "test_code_review_evaluation.py"
    
    print("=== Selvage 코드 리뷰 평가 시작 ===")
    
    # 1. 기본 실행
    print("\n1. 기본 평가 실행...")
    success = run_deepeval_test(test_file)
    
    if not success:
        print("기본 평가 실행 실패")
        return False
    
    # 2. 병렬 실행 (4개 프로세스)
    print("\n2. 병렬 평가 실행 (4개 프로세스)...")
    success = run_deepeval_test(test_file, parallel_workers=4)
    
    if not success:
        print("병렬 평가 실행 실패")
        return False
    
    # 3. 실패한 테스트만 표시
    print("\n3. 실패한 테스트 확인...")
    run_deepeval_test(test_file, display_filter="failing")
    
    print("\n=== 코드 리뷰 평가 완료 ===")
    return True

if __name__ == "__main__":
    # 직접 실행 시 코드 리뷰 평가 수행
    run_code_review_evaluation()
```

**4. 결과 확인**
- 로컬 결과: `~/Library/selvage-eval/deepeval_results/` 디렉토리
- 콘솔 출력: 각 메트릭별 점수 및 통과/실패 상태
- 실행 로그: subprocess 출력을 통한 상세 결과

**메트릭 점수 해석**
- **0.7 이상**: 통과 (양질의 리뷰)
- **0.5-0.7**: 보통 (개선 필요)
- **0.5 미만**: 실패 (심각한 문제)