# DeepEval 평가 시스템 구현 가이드

본 문서는 [04-evaluation-review-result.md](../specs/04-evaluation-review-result.md) 명세서를 바탕으로 DeepEval 관련 Tool 클래스들의 구현 가이드를 제공합니다.

## 목차
1. [Tool 클래스 기본 구조](#tool-클래스-기본-구조)
2. [DeepEvalTestCaseConverterTool 구현](#deepevaltestcaseconvertertool-구현)
3. [DeepEvalExecutorTool 구현](#deepevalexecutortool-구현)
4. [ToolGenerator 통합](#toolgenerator-통합)
5. [테스트 작성 가이드](#테스트-작성-가이드)
6. [메타데이터 및 파일 관리](#메타데이터-및-파일-관리)

## Tool 클래스 기본 구조

Selvage 평가 시스템의 모든 Tool은 다음 패턴을 따릅니다:

### 필수 구현 요소

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

from .tool import Tool, generate_parameters_schema_from_hints
from .tool_result import ToolResult

class YourTool(Tool):
    """도구 설명"""
    
    @property
    def name(self) -> str:
        """도구 이름 (snake_case)"""
        return "your_tool_name"
    
    @property
    def description(self) -> str:
        """도구 기능 설명 (한국어)"""
        return "도구의 기능을 설명합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        """파라미터 스키마 자동 생성"""
        return generate_parameters_schema_from_hints(self.execute)
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """파라미터 유효성 검증"""
        # 필수 파라미터 확인
        # 타입 검증
        # 값 범위 검증
        return True
    
    def execute(self, **kwargs) -> ToolResult:
        """도구 실행 메인 로직"""
        try:
            # 실행 로직
            return ToolResult(
                success=True,
                data=result_data,
                metadata={"execution_info": "추가 정보"}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"실행 실패: {str(e)}"
            )
```

### 코딩 컨벤션

- **타입 힌팅**: 모든 함수와 메서드에 타입 힌트 필수
- **Google 스타일 독스트링**: 한국어로 작성
- **데이터 클래스 사용**: Dict보다 @dataclass 우선 사용
- **에러 처리**: 명확한 에러 메시지와 ToolResult 반환
- **파일 경로**: Path 객체 사용, expanduser() 적용

## DeepEvalTestCaseConverterTool 구현

### 클래스 정의

```python
"""DeepEval 테스트 케이스 변환 도구

리뷰 로그를 DeepEval 형식으로 변환하는 도구입니다.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .tool import Tool, generate_parameters_schema_from_hints
from .tool_result import ToolResult


@dataclass
class ReviewLogInfo:
    """리뷰 로그 파일 정보"""
    repo_name: str
    commit_id: str
    model_name: str
    file_path: Path
    file_name: str


@dataclass  
class DeepEvalTestCase:
    """DeepEval 테스트 케이스"""
    input: str
    actual_output: str
    expected_output: Optional[str] = None


class DeepEvalTestCaseConverterTool(Tool):
    """DeepEval 테스트 케이스 변환 도구"""
    
    def __init__(self):
        self.review_logs_base_path = "~/Library/selvage-eval-agent/review_logs"
        self.output_base_path = "~/Library/selvage-eval/deep_eval_test_case"
    
    @property
    def name(self) -> str:
        return "deepeval_test_case_converter"
    
    @property
    def description(self) -> str:
        return "리뷰 로그를 DeepEval 테스트 케이스 형식으로 변환합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return generate_parameters_schema_from_hints(self.execute)
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """파라미터 유효성 검증
        
        Args:
            params: 검증할 파라미터 딕셔너리
            
        Returns:
            bool: 유효성 검증 결과
        """
        # session_id 필수 확인
        if 'session_id' not in params:
            return False
            
        if not isinstance(params['session_id'], str):
            return False
            
        if not params['session_id'].strip():
            return False
        
        # 선택적 파라미터 타입 확인
        if 'review_logs_path' in params and params['review_logs_path'] is not None:
            if not isinstance(params['review_logs_path'], str):
                return False
        
        if 'output_path' in params and params['output_path'] is not None:
            if not isinstance(params['output_path'], str):
                return False
            
        return True
    
    def execute(self, session_id: str, 
                review_logs_path: Optional[str] = None,
                output_path: Optional[str] = None) -> ToolResult:
        """리뷰 로그를 DeepEval 테스트 케이스로 변환합니다
        
        Args:
            session_id: 세션 ID (UUID 형식 권장)
            review_logs_path: 리뷰 로그 경로 (기본값: ~/Library/selvage-eval-agent/review_logs)
            output_path: 출력 경로 (기본값: ~/Library/selvage-eval/deep_eval_test_case)
            
        Returns:
            ToolResult: 변환 결과
        """
        try:
            # 경로 설정
            logs_path = Path(review_logs_path or self.review_logs_base_path).expanduser()
            output_base = Path(output_path or self.output_base_path).expanduser()
            
            # 리뷰 로그 스캔
            review_logs = self._scan_review_logs(logs_path)
            if not review_logs:
                return ToolResult(
                    success=False,
                    data=None,
                    error_message="리뷰 로그 파일을 찾을 수 없습니다"
                )
            
            # session_id 디렉토리 생성
            session_dir = output_base / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # 메타데이터 파일 생성
            metadata = self._create_metadata()
            metadata_path = session_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 모델별 테스트 케이스 변환
            converted_files = {}
            grouped_logs = self._group_logs_by_model(review_logs)
            
            for model_name, logs in grouped_logs.items():
                test_cases = []
                
                for log_info in logs:
                    extracted_data = self._extract_prompt_and_response(log_info.file_path)
                    if extracted_data:
                        test_case = self._convert_to_deepeval_format(extracted_data)
                        # repo 정보를 테스트 케이스에 추가
                        test_case['repo_name'] = log_info.repo_name
                        test_cases.append(test_case)
                
                if test_cases:
                    # 저장 경로: session_id/model_name/test_cases.json
                    output_dir = session_dir / model_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_file = output_dir / "test_cases.json"
                    test_cases_data = [self._test_case_to_dict(tc) for tc in test_cases]
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(test_cases_data, f, ensure_ascii=False, indent=2)
                    
                    converted_files[model_name] = {
                        "file_path": str(output_file),
                        "test_case_count": len(test_cases)
                    }
            
            return ToolResult(
                success=True,
                data={
                    "session_id": session_id,
                    "converted_files": converted_files,
                    "total_files": len(converted_files),
                    "metadata_path": str(metadata_path)
                },
                metadata={
                    "session_dir": str(session_dir),
                    "review_logs_processed": len(review_logs)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"DeepEval 테스트 케이스 변환 실패: {str(e)}"
            )
    
    def _scan_review_logs(self, base_path: Path) -> List[ReviewLogInfo]:
        """리뷰 로그 디렉토리 스캔"""
        review_logs = []
        
        if not base_path.exists():
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
                    
                    # 리뷰 로그 파일 찾기
                    for log_file in model_dir.glob("*.json"):
                        review_logs.append(ReviewLogInfo(
                            repo_name=repo_name,
                            commit_id=commit_id,
                            model_name=model_name,
                            file_path=log_file,
                            file_name=log_file.name
                        ))
        
        return review_logs
    
    def _group_logs_by_model(self, review_logs: List[ReviewLogInfo]) -> Dict[str, List[ReviewLogInfo]]:
        """리뷰 로그를 model_name별로 그룹화"""
        grouped = {}
        for log_info in review_logs:
            key = log_info.model_name
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(log_info)
        return grouped
    
    def _extract_prompt_and_response(self, log_file_path: Path) -> Optional[Dict[str, Any]]:
        """리뷰 로그 파일에서 prompt와 review_response 추출"""
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            prompt = log_data.get("prompt", [])
            review_response = log_data.get("review_response", {})
            
            if not prompt or not review_response:
                return None
            
            return {
                "prompt": prompt,
                "review_response": review_response,
                "original_data": log_data
            }
            
        except Exception:
            return None
    
    def _convert_to_deepeval_format(self, extracted_data: Dict[str, Any]) -> DeepEvalTestCase:
        """DeepEval 테스트 케이스 형식으로 변환"""
        return DeepEvalTestCase(
            input=json.dumps(extracted_data["prompt"], ensure_ascii=False),
            actual_output=json.dumps(extracted_data["review_response"], ensure_ascii=False)
        )
    
    def _test_case_to_dict(self, test_case: DeepEvalTestCase) -> Dict[str, Any]:
        """테스트 케이스를 딕셔너리로 변환"""
        result = {
            "input": test_case.input,
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output
        }
        # repo_name이 추가된 경우 포함
        if hasattr(test_case, 'repo_name'):
            result["repo_name"] = test_case.repo_name
        return result
    
    def _create_metadata(self) -> Dict[str, Any]:
        """메타데이터 파일 생성"""
        # Selvage 버전 확인
        selvage_version = get_selvage_version()
        
        return {
            "selvage_version": selvage_version,
            "execution_date": datetime.now().isoformat(),
            "tool_name": self.name,
            "created_by": "DeepEvalTestCaseConverterTool",
            "evaluation_framework": "DeepEval"
        }
```

## DeepEvalExecutorTool 구현

### 클래스 정의

```python
# JsonCorrectnessMetric를 위해 선언하는 파일

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


```python
"""DeepEval 평가 실행 도구

DeepEval 평가를 실행하고 결과를 수집하는 도구입니다.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .tool import Tool, generate_parameters_schema_from_hints
from .tool_result import ToolResult
from .execute_safe_command_tool import ExecuteSafeCommandTool


class DeepEvalExecutorTool(Tool):
    """DeepEval 평가 실행 도구"""
    
    def __init__(self):
        self.test_case_base_path = "~/Library/selvage-eval/deep_eval_test_case"
        self.results_base_path = "~/Library/selvage-eval/deepeval_results"
        self.command_executor = ExecuteSafeCommandTool()
    
    @property
    def name(self) -> str:
        return "deepeval_executor"
    
    @property
    def description(self) -> str:
        return "DeepEval을 사용하여 코드 리뷰 품질을 평가합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return generate_parameters_schema_from_hints(self.execute)
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """파라미터 유효성 검증"""
        # session_id 필수 확인
        if 'session_id' not in params:
            return False
            
        if not isinstance(params['session_id'], str):
            return False
            
        if not params['session_id'].strip():
            return False
        
        # 선택적 파라미터 타입 확인
        if 'parallel_workers' in params:
            if not isinstance(params['parallel_workers'], int) or params['parallel_workers'] < 1:
                return False
                
        if 'test_case_path' in params and params['test_case_path'] is not None:
            if not isinstance(params['test_case_path'], str):
                return False
                
        return True
    
    def execute(self, session_id: str,
                test_case_path: Optional[str] = None,
                parallel_workers: int = 1,
                display_filter: str = "all") -> ToolResult:
        """DeepEval 평가를 실행합니다
        
        Args:
            session_id: 세션 ID
            test_case_path: 테스트 케이스 경로 (지정하지 않으면 session_id로 자동 검색)
            parallel_workers: 병렬 워커 수 (기본값: 1)
            display_filter: 표시 필터 ("all", "failing", "passing")
            
        Returns:
            ToolResult: 평가 결과
        """
        try:
            # 환경 변수 확인
            env_check = self._check_environment()
            if not env_check["valid"]:
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"환경 설정 오류: {env_check['message']}"
                )
            
            # 테스트 케이스 경로 확인
            if test_case_path:
                test_cases_dir = Path(test_case_path).expanduser()
            else:
                test_cases_dir = Path(self.test_case_base_path).expanduser() / session_id
            
            if not test_cases_dir.exists():
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"테스트 케이스 디렉토리를 찾을 수 없습니다: {test_cases_dir}"
                )
            
            # 결과 저장 디렉토리 생성
            results_base = Path(self.results_base_path).expanduser()
            session_results_dir = results_base / session_id
            session_results_dir.mkdir(parents=True, exist_ok=True)
            
            # 메타데이터 파일 생성
            metadata = self._create_metadata(session_id, str(test_cases_dir))
            metadata_path = session_results_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 각 모델별 평가 실행
            evaluation_results = {}
            
            for model_dir in test_cases_dir.iterdir():
                if not model_dir.is_dir() or model_dir.name == "metadata.json":
                    continue
                    
                model_name = model_dir.name
                test_cases_file = model_dir / "test_cases.json"
                
                if not test_cases_file.exists():
                    continue
                
                # 평가 실행
                output_path = session_results_dir / model_name
                output_path.mkdir(parents=True, exist_ok=True)
                result = self._run_evaluation(
                    test_cases_file=test_cases_file,
                    output_path=output_path,
                    parallel_workers=parallel_workers,
                    display_filter=display_filter
                )
                
                # 평가 결과 저장
                evaluation_results[model_name] = {
                    "success": result["success"],
                    "error": result.get("error") if not result["success"] else None
                }
            
            return ToolResult(
                success=True,
                data={
                    "session_id": session_id,
                    "evaluation_results": evaluation_results,
                    "total_evaluations": len(evaluation_results),
                    "metadata_path": str(metadata_path)
                },
                metadata={
                    "session_results_dir": str(session_results_dir),
                    "parallel_workers": parallel_workers
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"DeepEval 평가 실행 실패: {str(e)}"
            )
    
    def _check_environment(self) -> Dict[str, Any]:
        """필수 환경 변수 확인"""
        required_vars = ["OPENAI_API_KEY", "GEMINI_API_KEY"]
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            return {
                "valid": False,
                "message": f"필수 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}"
            }
        
        return {"valid": True, "message": "환경 설정 완료"}
    
    def _run_evaluation(self, test_cases_file: Path, 
                       output_path: Path,
                       parallel_workers: int, 
                       display_filter: str) -> Dict[str, Any]:
        """실제 DeepEval 평가 실행"""
        try:
            # 필요한 import 문
            from deepeval import evaluate
            from deepeval.evaluate import DisplayConfig, AsyncConfig
            
            # 테스트 케이스 로드
            test_cases = self._load_test_cases(test_cases_file)
            if not test_cases:
                return {
                    "success": False,
                    "error": f"테스트 케이스를 로드할 수 없습니다: {test_cases_file}"
                }
            
            # 메트릭 생성
            metrics = self._create_metrics()
            
            # DisplayConfig 설정
            display_config = DisplayConfig(
                display=display_filter,
                output_file_dr=str(output_path)
            )
            
            # AsyncConfig 설정
            async_config = AsyncConfig(
                max_concurrent=parallel_workers
            )
            
            # DeepEval 평가 실행
            evaluate(
                test_cases=test_cases,
                metrics=metrics,
                display_config=display_config,
                async_config=async_config
            )
            
            return {
                "success": True,
                "data": {"executed": True, "test_cases_count": len(test_cases)}
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"평가 실행 중 오류: {str(e)}"
            }
    
    def _create_metrics(self) -> list:
        """평가 메트릭 생성"""
        from deepeval.metrics.g_eval.g_eval import GEval
        from deepeval.metrics.json_correctness.json_correctness import JsonCorrectnessMetric
        from deepeval.test_case.llm_test_case import LLMTestCaseParams
        from selvage.src.utils.token.models import (
            IssueSeverityEnum,
            StructuredReviewIssue,
            StructuredReviewResponse,
        )
        
        model = "gemini-2.0-flash-exp"
        
        # Correctness GEval - 코드 리뷰 정확성 평가
        correctness = GEval(
            name="Correctness",
            model=model,
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
        
        # Clarity GEval - 코드 리뷰 명확성 평가
        clarity = GEval(
            name="Clarity",
            model=model,
            evaluation_steps=[
                "Evaluate whether the overall code review output (including summary, and if present, issue descriptions, suggestions, and recommendations) uses concise and direct language.",
                "Assess whether issue descriptions and suggestions and recommendations are specific and clear.",
                "Review if the purpose and intent of code changes are clearly understandable.",
                "Verify that improved code examples are provided and easy to understand.",
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.7,
        )
        
        # Actionability GEval - 코드 리뷰 실행 가능성 평가
        actionability = GEval(
            name="Actionability",
            model=model,
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
        
        # JsonCorrectnessMetric - JSON 형식 정확성 평가
        jsoncorrectness = JsonCorrectnessMetric(
            expected_schema=StructuredReviewResponse(
                issues=[
                    StructuredReviewIssue(
                        type="",
                        line_number=0,
                        file="",
                        description="",
                        suggestion="",
                        severity=IssueSeverityEnum.INFO,
                        target_code="",
                        suggested_code="",
                    )
                ],
                summary="",
                score=0,
                recommendations=[],
            ),
            model=model,
            include_reason=True,
        )
        
        return [correctness, clarity, actionability, jsoncorrectness]
    
    def _load_test_cases(self, test_cases_file: Path) -> list:
        """테스트 케이스 JSON 파일 로드"""
        import json
        from deepeval.test_case.llm_test_case import LLMTestCase
        
        try:
            with open(test_cases_file, 'r', encoding='utf-8') as f:
                test_cases_data = json.load(f)
            
            test_cases = []
            for case_data in test_cases_data:
                test_case = LLMTestCase(
                    input=case_data.get("input", ""),
                    actual_output=case_data.get("actual_output", "")
                )
                test_cases.append(test_case)
            
            return test_cases
            
        except Exception as e:
            print(f"테스트 케이스 로드 실패: {str(e)}")
            return []
    
    def _create_metadata(self, session_id: str, test_case_path: str) -> Dict[str, Any]:
        """메타데이터 파일 생성"""
        # Selvage 버전 확인
        selvage_version = get_selvage_version()
        
        return {
            "selvage_version": selvage_version,
            "execution_date": datetime.now().isoformat(),
            "session_id": session_id,
            "deep_eval_test_case_path": test_case_path,
            "tool_name": self.name,
            "created_by": "DeepEvalExecutorTool"
        }
```

## ToolGenerator 통합

ToolGenerator에 새로운 Tool들을 등록합니다:

```python
# src/selvage_eval/tools/tool_generator.py 수정

from .deepeval_test_case_converter_tool import DeepEvalTestCaseConverterTool
from .deepeval_executor_tool import DeepEvalExecutorTool

class ToolGenerator:
    """ToolGenerator class"""

    def generate_tool(self, tool_name: str, params: Dict[str, Any]) -> Tool:
        """Generate a tool"""
        if tool_name == "read_file":
            return ReadFileTool()
        elif tool_name == "write_file":
            return WriteFileTool()
        elif tool_name == "file_exists":
            return FileExistsTool()
        elif tool_name == "execute_safe_command":
            return ExecuteSafeCommandTool()
        elif tool_name == "list_directory":
            return ListDirectoryTool()
        elif tool_name == "execute_reviews":
            return ReviewExecutorTool()
        elif tool_name == "deepeval_test_case_converter":  # 새로 추가
            return DeepEvalTestCaseConverterTool()
        elif tool_name == "deepeval_executor":  # 새로 추가
            return DeepEvalExecutorTool()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
```
## 메타데이터 및 파일 관리

### 디렉토리 구조

```
~/Library/selvage-eval/
├── deep_eval_test_case/
│   └── {session_id}/
│       ├── metadata.json
│       └── {model_name}/
│           └── test_cases.json
└── deepeval_results/
    └── {session_id}/
        ├── metadata.json
        └── {model_name}/
            └── [DeepEval 라이브러리가 자동 생성하는 결과 파일들]
```

**주의사항**: `deepeval_results` 디렉토리의 각 모델 폴더에는 DeepEval 라이브러리가 자동으로 생성하는 결과 파일들이 저장됩니다. 파일명과 형식은 DeepEval 버전과 설정에 따라 달라질 수 있습니다.

### 메타데이터 스키마

#### DeepEvalTestCaseConverterTool 메타데이터

```json
{
  "selvage_version": "0.1.2",
  "execution_date": "2024-01-15T10:30:00.123456",
  "tool_name": "deepeval_test_case_converter",
  "created_by": "DeepEvalTestCaseConverterTool",
  "evaluation_framework": "DeepEval"
}
```

#### DeepEvalExecutorTool 메타데이터

```json
{
  "selvage_version": "0.1.2",
  "execution_date": "2024-01-15T10:30:00.123456",
  "session_id": "uuid-string",
  "deep_eval_test_case_path": "/path/to/test/cases",
  "tool_name": "deepeval_executor",
  "created_by": "DeepEvalExecutorTool"
}
```

### 유틸리티 함수

#### get_selvage_version()

메타데이터 생성 시 Selvage 바이너리의 실제 버전을 동적으로 가져오는 함수입니다:

```python
import subprocess

def get_selvage_version() -> str:
    """Selvage 바이너리 버전 확인
    
    Returns:
        str: Selvage 바이너리 버전 (예: "0.1.2") 또는 "unknown"
    """
    try:
        result = subprocess.run(
            ["/Users/demin_coder/.local/bin/selvage", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"
```

#### 메타데이터 생성 시 사용 방법

각 Tool의 `_create_metadata()` 메서드에서 다음과 같이 사용합니다:

```python
# 기존 하드코딩 방식
selvage_version = "0.1.2"  # 기본값

# 개선된 동적 방식
selvage_version = get_selvage_version()

return {
    "selvage_version": selvage_version,
    "execution_date": datetime.now().isoformat(),
    # ... 기타 메타데이터 필드들
}
```

## 실행 예시

```python
# DeepEval 테스트 케이스 변환
converter = DeepEvalTestCaseConverterTool()
result = converter.execute(session_id="eval-2024-01-15")

if result.success:
    print(f"변환 완료: {result.data['total_files']}개 파일")
    print(f"메타데이터 저장 위치: {result.data['metadata_path']}")
    
    # 변환된 파일 정보 출력
    for model_name, info in result.data['converted_files'].items():
        print(f"모델 {model_name}: {info['test_case_count']}개 테스트 케이스")
    
    # DeepEval 평가 실행
    executor = DeepEvalExecutorTool()
    eval_result = executor.execute(
        session_id="eval-2024-01-15",
        parallel_workers=4
    )
    
    if eval_result.success:
        print(f"평가 완료: {eval_result.data['total_evaluations']}개 평가")
        print(f"결과 메타데이터: {eval_result.data['metadata_path']}")
        
        # 평가 결과 요약
        for model_name, result_info in eval_result.data['evaluation_results'].items():
            if result_info['success']:
                print(f"모델 {model_name}: 평가 성공")
            else:
                print(f"모델 {model_name}: 평가 실패 - {result_info['error']}")
    else:
        print(f"평가 실패: {eval_result.error_message}")
else:
    print(f"변환 실패: {result.error_message}")
```

이 구현 가이드는 Selvage 평가 시스템의 기존 패턴과 완전히 일치하며, 명세서의 모든 요구사항을 충족합니다.