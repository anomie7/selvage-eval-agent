# Selvage 평가 에이전트 - 평가(3단계)

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
    model="gemini-2.5-pro-preview-05-06",
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
    model="gemini-2.5-pro-preview-05-06",
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
    model="gemini-2.5-pro-preview-05-06",
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
```python
@pytest.mark.parametrize("test_case", dataset)
def test_code_review_evaluation(test_case: LLMTestCase):
    """코드 리뷰 평가 테스트."""
    assert_test(
        test_case,
        metrics=[correctness, clarity, actionability, jsoncorrectness],
    )
```

**메트릭 점수 해석**
- **0.7 이상**: 통과 (양질의 리뷰)
- **0.5-0.7**: 보통 (개선 필요)
- **0.5 미만**: 실패 (심각한 문제)

**결과 분석을 위한 프롬프트**
DeepEval 결과의 영어 실패 사유를 한국어로 번역하고 가독성을 개선하기 위해 다음 프롬프트를 사용:

```markdown
# ROLE
당신은 문제 해결 중심의 정확하고 충실한 테크니컬 라이터입니다.

# PROBLEM
deepeval의 metric을 통해 평가한 결과에서 fail reason이 영어로 적혀있어 평가가 어렵습니다.

## INSTRUCTIONS
1. 각 testCase의 metricsData만 추출
2. metricsData.reason들을 한국어로 번역
3. reason 의견을 토대로 input(프롬프트), actualOutput에서 문제 부분만 첨부
4. reason 결과를 종합한 의견 첨부

가독성을 고려해서 편집해서 반환해주세요.
```

## Phase 3 Tool 구현

### Phase 3 Tools: DeepEval Conversion

#### ReviewLogScannerTool - 리뷰 로그 스캔

**LLM Tool Definition (OpenAI Function Calling)**
```javascript
{
  "type": "function",
  "function": {
    "name": "review_log_scanner",
    "description": "리뷰 로그 디렉토리를 스캔하여 모든 리뷰 로그 파일을 찾고 메타데이터를 추출합니다",
    "parameters": {
      "type": "object",
      "properties": {
        "base_path": {
          "type": "string",
          "description": "리뷰 로그 기본 경로 (기본값: ~/Library/selvage-eval-agent/review_logs)"
        }
      }
    }
  }
}
```

**LLM Tool Definition (Anthropic Claude)**
```javascript
{
  "name": "review_log_scanner",
  "description": "리뷰 로그 디렉토리를 스캔하여 모든 리뷰 로그 파일을 찾고 메타데이터를 추출합니다",
  "input_schema": {
    "type": "object",
    "properties": {
      "base_path": {
        "type": "string",
        "description": "리뷰 로그 기본 경로 (기본값: ~/Library/selvage-eval-agent/review_logs)"
      }
    }
  }
}
```

**Python Implementation**
```python
class ReviewLogScannerTool(Tool):
    """리뷰 로그 파일 스캔 및 메타데이터 추출"""
    
    @property
    def name(self) -> str:
        return "review_log_scanner"
    
    @property
    def description(self) -> str:
        return "리뷰 로그 디렉토리를 스캔하여 모든 리뷰 로그 파일을 찾고 메타데이터를 추출합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "base_path": {
                    "type": "string", 
                    "description": "리뷰 로그 기본 경로",
                    "default": "~/Library/selvage-eval-agent/review_logs"
                }
            }
        }
    
    def execute(self, **kwargs) -> ToolResult:
        base_path = kwargs.get("base_path", "~/Library/selvage-eval-agent/review_logs")
        
        review_logs = []
        base_path = Path(base_path).expanduser()
        
        try:
            if not base_path.exists():
                return ToolResult(
                    success=False,
                    error_message=f"경로가 존재하지 않습니다: {base_path}"
                )
            
            # 디렉토리 구조 탐색: repo_name/commit_id/model_name/*.json
            for repo_dir in base_path.iterdir():
                if not repo_dir.is_dir():
                    continue
                    
                for commit_dir in repo_dir.iterdir():
                    if not commit_dir.is_dir():
                        continue
                        
                    for model_dir in commit_dir.iterdir():
                        if not model_dir.is_dir():
                            continue
                            
                        for log_file in model_dir.glob("*.json"):
                            metadata = self._extract_log_metadata(log_file)
                            review_logs.append({
                                "repo_name": repo_dir.name,
                                "commit_id": commit_dir.name,
                                "model_name": model_dir.name,
                                "file_path": str(log_file),
                                "file_name": log_file.name,
                                "metadata": metadata
                            })
            
            return ToolResult(
                success=True, 
                data={
                    "review_logs": review_logs,
                    "total_count": len(review_logs),
                    "scan_path": str(base_path)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to scan review logs: {str(e)}"
            )
    
    def _extract_log_metadata(self, log_file: Path) -> Dict[str, Any]:
        """리뷰 로그 파일에서 메타데이터 추출"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            return {
                "log_id": log_data.get("id"),
                "model": log_data.get("model", {}),
                "created_at": log_data.get("created_at"),
                "status": log_data.get("status", "UNKNOWN"),
                "prompt_version": log_data.get("prompt_version"),
                "file_size": log_file.stat().st_size,
                "has_prompt": bool(log_data.get("prompt")),
                "has_response": bool(log_data.get("review_response"))
            }
        except Exception as e:
            return {"error": str(e)}
```

#### DeepEvalConverterTool - DeepEval 형식 변환

**LLM Tool Definition (OpenAI Function Calling)**
```javascript
{
  "type": "function",
  "function": {
    "name": "deepeval_converter",
    "description": "리뷰 로그 데이터를 DeepEval 테스트 케이스 형식으로 변환합니다",
    "parameters": {
      "type": "object",
      "properties": {
        "review_logs": {
          "type": "array",
          "description": "변환할 리뷰 로그 정보 리스트",
          "items": {
            "type": "object",
            "properties": {
              "repo_name": {"type": "string"},
              "commit_id": {"type": "string"},
              "model_name": {"type": "string"},
              "file_path": {"type": "string"}
            }
          }
        },
        "output_dir": {
          "type": "string",
          "description": "출력 디렉토리 (기본값: ~/Library/selvage-eval-agent/deep_eval_test_case)"
        },
        "group_by": {
          "type": "string",
          "description": "그룹화 기준",
          "enum": ["repo_model", "repo", "model"],
          "default": "repo_model"
        }
      },
      "required": ["review_logs"]
    }
  }
}
```

**LLM Tool Definition (Anthropic Claude)**
```javascript
{
  "name": "deepeval_converter",
  "description": "리뷰 로그 데이터를 DeepEval 테스트 케이스 형식으로 변환합니다",
  "input_schema": {
    "type": "object",
    "properties": {
      "review_logs": {
        "type": "array",
        "description": "변환할 리뷰 로그 정보 리스트",
        "items": {
          "type": "object",
          "properties": {
            "repo_name": {"type": "string"},
            "commit_id": {"type": "string"},
            "model_name": {"type": "string"},
            "file_path": {"type": "string"}
          }
        }
      },
      "output_dir": {
        "type": "string",
        "description": "출력 디렉토리 (기본값: ~/Library/selvage-eval-agent/deep_eval_test_case)"
      },
      "group_by": {
        "type": "string",
        "description": "그룹화 기준",
        "enum": ["repo_model", "repo", "model"]
      }
    },
    "required": ["review_logs"]
  }
}
```

**Python Implementation**
```python
class DeepEvalConverterTool(Tool):
    """리뷰 로그를 DeepEval 테스트 케이스로 변환"""
    
    @property
    def name(self) -> str:
        return "deepeval_converter"
    
    @property
    def description(self) -> str:
        return "리뷰 로그 데이터를 DeepEval 테스트 케이스 형식으로 변환합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "review_logs": {
                    "type": "array",
                    "description": "변환할 리뷰 로그 정보 리스트"
                },
                "output_dir": {
                    "type": "string",
                    "description": "출력 디렉토리",
                    "default": "~/Library/selvage-eval-agent/deep_eval_test_case"
                },
                "group_by": {
                    "type": "string",
                    "description": "그룹화 기준 (repo_model, repo, model)",
                    "default": "repo_model"
                }
            },
            "required": ["review_logs"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        review_logs = kwargs["review_logs"]
        output_dir = kwargs.get("output_dir", "~/Library/selvage-eval-agent/deep_eval_test_case")
        group_by = kwargs.get("group_by", "repo_model")
        
        try:
            output_path = Path(output_dir).expanduser()
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 그룹별로 테스트 케이스 생성
            grouped_logs = self._group_logs(review_logs, group_by)
            converted_files = []
            
            for group_key, logs in grouped_logs.items():
                test_cases = []
                for log_info in logs:
                    test_case = self._convert_single_log(log_info)
                    if test_case:
                        test_cases.append(test_case)
                
                if test_cases:
                    file_path = self._save_test_cases(
                        group_key, test_cases, output_path
                    )
                    converted_files.append({
                        "group": group_key,
                        "file_path": file_path,
                        "test_case_count": len(test_cases)
                    })
            
            return ToolResult(
                success=True,
                data={
                    "converted_files": converted_files,
                    "total_test_cases": sum(f["test_case_count"] for f in converted_files),
                    "output_directory": str(output_path)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to convert logs: {str(e)}"
            )
    
    def _group_logs(self, review_logs: List[Dict], group_by: str) -> Dict[str, List[Dict]]:
        """로그를 그룹별로 분류"""
        grouped = {}
        
        for log in review_logs:
            if group_by == "repo_model":
                key = f"{log['repo_name']}_{log['model_name']}"
            elif group_by == "repo":
                key = log['repo_name']
            elif group_by == "model":
                key = log['model_name']
            else:
                key = "all"
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(log)
        
        return grouped
    
    def _convert_single_log(self, log_info: Dict) -> Optional[Dict[str, Any]]:
        """단일 로그를 DeepEval 테스트 케이스로 변환"""
        try:
            with open(log_info["file_path"], 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            prompt = log_data.get("prompt", [])
            review_response = log_data.get("review_response", {})
            
            if not prompt or not review_response:
                return None
            
            return {
                "input": json.dumps(prompt, ensure_ascii=False),
                "actual_output": json.dumps(review_response, ensure_ascii=False),
                "expected_output": None,  # 현재 사용하지 않음
                "metadata": {
                    "repo_name": log_info["repo_name"],
                    "commit_id": log_info["commit_id"],
                    "model_name": log_info["model_name"],
                    "log_id": log_data.get("id"),
                    "created_at": log_data.get("created_at")
                }
            }
        except Exception as e:
            print(f"Failed to convert log {log_info['file_path']}: {e}")
            return None
    
    def _save_test_cases(self, group_key: str, test_cases: List[Dict], 
                              output_path: Path) -> str:
        """테스트 케이스를 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_data_{timestamp}_{group_key}.json"
        file_path = output_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, ensure_ascii=False, indent=2)
        
        return str(file_path)
```

#### MetricEvaluatorTool - 메트릭 평가 실행

**LLM Tool Definition (OpenAI Function Calling)**
```javascript
{
  "type": "function",
  "function": {
    "name": "metric_evaluator",
    "description": "DeepEval 메트릭을 사용하여 테스트 케이스를 평가합니다",
    "parameters": {
      "type": "object",
      "properties": {
        "test_case_files": {
          "type": "array",
          "description": "평가할 테스트 케이스 파일 목록",
          "items": {
            "type": "object",
            "properties": {
              "file_path": {"type": "string", "description": "테스트 케이스 파일 경로"},
              "group": {"type": "string", "description": "그룹 이름"}
            }
          }
        },
        "metrics": {
          "type": "array",
          "description": "사용할 메트릭 목록",
          "items": {
            "type": "string",
            "enum": ["correctness", "clarity", "actionability", "json_correctness"]
          },
          "default": ["correctness", "clarity", "actionability", "json_correctness"]
        },
        "judge_model": {
          "type": "string",
          "description": "평가에 사용할 judge 모델",
          "enum": ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"],
          "default": "gpt-4"
        },
        "output_dir": {
          "type": "string",
          "description": "결과 저장 디렉토리 (기본값: ~/Library/selvage-eval-agent/evaluation_results)"
        }
      },
      "required": ["test_case_files"]
    }
  }
}
```

**LLM Tool Definition (Anthropic Claude)**
```javascript
{
  "name": "metric_evaluator",
  "description": "DeepEval 메트릭을 사용하여 테스트 케이스를 평가합니다",
  "input_schema": {
    "type": "object",
    "properties": {
      "test_case_files": {
        "type": "array",
        "description": "평가할 테스트 케이스 파일 목록",
        "items": {
          "type": "object",
          "properties": {
            "file_path": {"type": "string", "description": "테스트 케이스 파일 경로"},
            "group": {"type": "string", "description": "그룹 이름"}
          }
        }
      },
      "metrics": {
        "type": "array",
        "description": "사용할 메트릭 목록",
        "items": {
          "type": "string",
          "enum": ["correctness", "clarity", "actionability", "json_correctness"]
        }
      },
      "judge_model": {
        "type": "string",
        "description": "평가에 사용할 judge 모델",
        "enum": ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]
      },
      "output_dir": {
        "type": "string",
        "description": "결과 저장 디렉토리 (기본값: ~/Library/selvage-eval-agent/evaluation_results)"
      }
    },
    "required": ["test_case_files"]
  }
}
```

**Python Implementation**
```python
class MetricEvaluatorTool(Tool):
    """DeepEval 메트릭을 사용한 평가 실행"""
    
    @property
    def name(self) -> str:
        return "metric_evaluator"
    
    @property
    def description(self) -> str:
        return "DeepEval 메트릭을 사용하여 테스트 케이스를 평가합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "test_case_files": {
                    "type": "array",
                    "description": "평가할 테스트 케이스 파일 목록"
                },
                "metrics": {
                    "type": "array",
                    "description": "사용할 메트릭 목록",
                    "default": ["correctness", "clarity", "actionability", "json_correctness"]
                },
                "judge_model": {
                    "type": "string",
                    "description": "평가에 사용할 judge 모델",
                    "default": "gpt-4"
                },
                "output_dir": {
                    "type": "string",
                    "description": "결과 저장 디렉토리",
                    "default": "~/Library/selvage-eval-agent/evaluation_results"
                }
            },
            "required": ["test_case_files"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        test_case_files = kwargs["test_case_files"]
        metrics = kwargs.get("metrics", ["correctness", "clarity", "actionability", "json_correctness"])
        judge_model = kwargs.get("judge_model", "gpt-4")
        output_dir = kwargs.get("output_dir", "~/Library/selvage-eval-agent/evaluation_results")
        
        try:
            output_path = Path(output_dir).expanduser()
            output_path.mkdir(parents=True, exist_ok=True)
            
            evaluation_results = []
            
            for file_info in test_case_files:
                file_path = file_info["file_path"]
                group = file_info["group"]
                
                # 테스트 케이스 로드
                with open(file_path, 'r', encoding='utf-8') as f:
                    test_cases = json.load(f)
                
                # DeepEval 평가 실행
                results = self._run_deepeval_evaluation(
                    test_cases, metrics, judge_model
                )
                
                # 결과 저장
                result_file = self._save_evaluation_results(
                    group, results, output_path
                )
                
                evaluation_results.append({
                    "group": group,
                    "test_case_file": file_path,
                    "result_file": result_file,
                    "test_case_count": len(test_cases),
                    "evaluation_count": len(results)
                })
            
            return ToolResult(
                success=True,
                data={
                    "evaluation_results": evaluation_results,
                    "total_evaluations": sum(r["evaluation_count"] for r in evaluation_results),
                    "output_directory": str(output_path)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to evaluate metrics: {str(e)}"
            )
    
    def _run_deepeval_evaluation(self, test_cases: List[Dict], 
                                     metrics: List[str], judge_model: str) -> List[Dict]:
        """DeepEval을 사용한 실제 평가 실행"""
        from deepeval.metrics import (
            AnswerRelevancyMetric, 
            FaithfulnessMetric,
            HallucinationMetric,
            G_Eval
        )
        from deepeval.test_case import LLMTestCase
        
        # 메트릭 인스턴스 생성
        metric_instances = {}
        
        if "correctness" in metrics:
            metric_instances["correctness"] = G_Eval(
                name="Correctness",
                criteria="코드 리뷰의 정확성을 평가합니다",
                evaluation_params=[
                    "이슈 식별의 정확성",
                    "제안 사항의 적절성",
                    "코드 이해도"
                ],
                model=judge_model
            )
        
        if "clarity" in metrics:
            metric_instances["clarity"] = G_Eval(
                name="Clarity",
                criteria="리뷰 내용의 명확성을 평가합니다",
                evaluation_params=[
                    "설명의 이해하기 쉬움",
                    "구체적인 예시 제공",
                    "전문 용어 사용의 적절성"
                ],
                model=judge_model
            )
        
        if "actionability" in metrics:
            metric_instances["actionability"] = G_Eval(
                name="Actionability",
                criteria="리뷰의 실행 가능성을 평가합니다",
                evaluation_params=[
                    "구체적인 해결 방안 제시",
                    "실제 적용 가능성",
                    "우선순위의 명확성"
                ],
                model=judge_model
            )
        
        if "json_correctness" in metrics:
            metric_instances["json_correctness"] = G_Eval(
                name="JsonCorrectness",
                criteria="JSON 형식의 정확성을 평가합니다",
                evaluation_params=[
                    "JSON 구조의 유효성",
                    "필수 필드 포함 여부",
                    "데이터 타입의 일관성"
                ],
                model=judge_model
            )
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                # DeepEval 테스트 케이스 생성
                llm_test_case = LLMTestCase(
                    input=test_case["input"],
                    actual_output=test_case["actual_output"],
                    expected_output=test_case.get("expected_output")
                )
                
                # 각 메트릭별 평가
                case_results = {
                    "test_case_index": i,
                    "metadata": test_case.get("metadata", {}),
                    "scores": {}
                }
                
                for metric_name, metric_instance in metric_instances.items():
                    try:
                        metric_instance.measure(llm_test_case)
                        case_results["scores"][metric_name] = {
                            "score": metric_instance.score,
                            "reason": getattr(metric_instance, 'reason', None),
                            "success": metric_instance.success
                        }
                    except Exception as e:
                        case_results["scores"][metric_name] = {
                            "score": 0.0,
                            "reason": f"Evaluation failed: {str(e)}",
                            "success": False
                        }
                
                results.append(case_results)
                
            except Exception as e:
                results.append({
                    "test_case_index": i,
                    "metadata": test_case.get("metadata", {}),
                    "scores": {},
                    "error": str(e)
                })
        
        return results
    
    def _save_evaluation_results(self, group: str, results: List[Dict], 
                                     output_path: Path) -> str:
        """평가 결과를 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}_{group}.json"
        file_path = output_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return str(file_path)
```
