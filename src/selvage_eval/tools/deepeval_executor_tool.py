"""DeepEval 평가 실행 도구

DeepEval을 사용하여 코드 리뷰 품질을 평가하는 도구입니다.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel

from .tool import Tool, generate_parameters_schema_from_hints
from .tool_result import ToolResult
from .execute_safe_command_tool import ExecuteSafeCommandTool
from .deepeval_test_case_converter_tool import get_selvage_version


# Structured Outputs용 스키마 클래스들
class IssueSeverityEnum(str, Enum):
    """이슈 심각도 열거형"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

class StructuredReviewIssue(BaseModel):
    """Structured Outputs용 코드 리뷰 이슈 모델"""
    type: str
    line_number: Optional[int]
    file: Optional[str]
    description: str
    suggestion: Optional[str]
    severity: IssueSeverityEnum
    target_code: Optional[str]
    suggested_code: Optional[str]

class StructuredReviewResponse(BaseModel):
    """Structured Outputs용 코드 리뷰 응답 모델"""
    issues: List[StructuredReviewIssue]
    summary: str
    score: Optional[float]
    recommendations: List[str]


class DeepEvalExecutorTool(Tool):
    """DeepEval 평가 실행 도구"""
    
    # 토큰 제한 설정 (Gemini 2.5 Pro 기준)
    MAX_TOKENS_PER_REQUEST = 100_000  # 안전 마진 포함
    MAX_BATCH_SIZE = 5  # 배치당 최대 테스트 케이스 수
    
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
                
                print(f"모델 {model_name} 평가 시작...")
                try:
                    result = self._run_evaluation(
                        test_cases_file=test_cases_file,
                        output_path=output_path,
                        parallel_workers=parallel_workers,
                        display_filter=display_filter
                    )
                    
                    # 평가 결과 저장
                    evaluation_results[model_name] = {
                        "success": result["success"],
                        "error": result.get("error") if not result["success"] else None,
                        "data": result.get("data", {})
                    }
                    
                    if result["success"]:
                        print(f"모델 {model_name} 평가 완료")
                    else:
                        print(f"모델 {model_name} 평가 실패: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"모델 {model_name} 평가 중 예외 발생: {str(e)}")
                    evaluation_results[model_name] = {
                        "success": False,
                        "error": f"평가 실행 중 예외: {str(e)}",
                        "data": {}
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
    
    def _estimate_token_count(self, text: str) -> int:
        """텍스트의 토큰 수를 추정합니다 (대략적인 계산)"""
        # 대략적인 토큰 수 계산 (영어 기준 4글자당 1토큰, 한국어 기준 2글자당 1토큰)
        english_chars = sum(1 for c in text if ord(c) < 128)
        korean_chars = len(text) - english_chars
        return (english_chars // 4) + (korean_chars // 2)
    
    def _split_test_cases_by_token_limit(self, test_cases: List[Any], max_tokens: int) -> List[List[Any]]:
        """테스트 케이스를 토큰 제한에 따라 배치로 분할"""
        batches = []
        current_batch = []
        current_tokens = 0
        
        for test_case in test_cases:
            # 테스트 케이스의 토큰 수 계산
            test_case_text = str(test_case.input) + str(test_case.actual_output)
            test_case_tokens = self._estimate_token_count(test_case_text)
            
            # 단일 테스트 케이스가 토큰 제한을 초과하는 경우 별도 배치로 처리
            if test_case_tokens > max_tokens:
                print(f"⚠️  테스트 케이스가 토큰 제한({max_tokens})을 초과합니다 ({test_case_tokens} 토큰). 별도 배치로 처리합니다.")
                # 현재 배치가 있으면 먼저 완료
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                # 큰 테스트 케이스를 혼자서 배치로 생성
                batches.append([test_case])
                continue
            
            # 현재 배치에 추가할 수 있는지 확인
            if current_tokens + test_case_tokens > max_tokens and current_batch:
                # 현재 배치를 완료하고 새 배치 시작
                batches.append(current_batch)
                current_batch = [test_case]
                current_tokens = test_case_tokens
            else:
                current_batch.append(test_case)
                current_tokens += test_case_tokens
                
            # 배치 크기 제한 확인
            if len(current_batch) >= self.MAX_BATCH_SIZE:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
        
        # 마지막 배치 추가
        if current_batch:
            batches.append(current_batch)
            
        return batches
    
    def _retry_with_backoff(self, func, max_retries: int = 3, initial_delay: float = 1.0):
        """지수 백오프를 사용한 재시도 로직"""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                # API 할당량 오류인 경우 더 긴 대기
                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    delay = initial_delay * (2 ** attempt) * 3  # 할당량 오류 시 더 긴 대기 (3분, 6분, 12분)
                else:
                    delay = initial_delay * (2 ** attempt)
                
                print(f"재시도 {attempt + 1}/{max_retries} - {delay:.1f}초 후 재시도: {str(e)}")
                time.sleep(delay)
    
    def _run_evaluation(self, test_cases_file: Path, 
                       output_path: Path,
                       parallel_workers: int, 
                       display_filter: str) -> Dict[str, Any]:
        """실제 DeepEval 평가 실행"""
        try:
            # 테스트 케이스 로드
            test_cases = self._load_test_cases(test_cases_file)
            if not test_cases:
                return {
                    "success": False,
                    "error": f"테스트 케이스를 로드할 수 없습니다: {test_cases_file}"
                }
            
            # DeepEval import 확인
            try:
                from deepeval import evaluate
                from deepeval.evaluate.configs import DisplayConfig, AsyncConfig
                from deepeval.test_run.test_run import TestRunResultDisplay
            except ImportError as e:
                return {
                    "success": False,
                    "error": f"DeepEval 라이브러리를 가져올 수 없습니다: {str(e)}"
                }
            
            # 메트릭 생성
            metrics = self._create_metrics()
            
            # 테스트 케이스 분할 처리
            batches = self._split_test_cases_by_token_limit(test_cases, self.MAX_TOKENS_PER_REQUEST)
            print(f"총 {len(test_cases)}개 테스트 케이스를 {len(batches)}개 배치로 분할")
            
            # 병렬 처리 수 조정 (토큰 제한 고려)
            adjusted_parallel_workers = min(parallel_workers, 1)  # API 안정성을 위해 1로 제한
            
            # DisplayConfig 설정
            display_config = DisplayConfig(
                display_option=self._convert_display_filter_to_enum(display_filter, TestRunResultDisplay),
                file_output_dir=str(output_path),
                verbose_mode=True,
            )
            
            # AsyncConfig 설정
            async_config = AsyncConfig(
                max_concurrent=adjusted_parallel_workers
            )
            
            # 각 배치별로 평가 실행
            total_processed = 0
            for i, batch in enumerate(batches):
                print(f"배치 {i+1}/{len(batches)} 처리 중 ({len(batch)}개 테스트 케이스)")
                
                # 배치 평가 실행 (재시도 로직 적용)
                def run_batch_evaluation():
                    return evaluate(
                        test_cases=batch,
                        metrics=metrics,
                        display_config=display_config,
                        async_config=async_config
                    )
                
                try:
                    self._retry_with_backoff(run_batch_evaluation, max_retries=3, initial_delay=60.0)
                    total_processed += len(batch)
                    print(f"배치 {i+1} 완료")
                    
                    # 배치 간 딜레이 (API 안정성)
                    if i < len(batches) - 1:
                        time.sleep(180.0)  # 3분 대기
                        
                except Exception as e:
                    print(f"배치 {i+1} 실패: {str(e)}")
                    # 부분 성공이라도 계속 진행
                    continue
            
            return {
                "success": True,
                "data": {
                    "executed": True, 
                    "test_cases_count": len(test_cases),
                    "processed_count": total_processed,
                    "batches_count": len(batches)
                }
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"평가 실행 중 오류: {str(e)}"
            }
    
    def _create_metrics(self) -> List[Any]:
        """평가 메트릭 생성"""
        try:
            from deepeval.metrics.g_eval.g_eval import GEval
            from deepeval.metrics.json_correctness.json_correctness import JsonCorrectnessMetric
            from deepeval.test_case.llm_test_case import LLMTestCaseParams
        except ImportError:
            # DeepEval을 가져올 수 없는 경우 빈 리스트 반환
            return []
        
        model = "gemini-2.5-pro"
        
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
        try:
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
        except Exception:
            # JsonCorrectnessMetric 생성 실패 시 다른 메트릭만 반환
            return [correctness, clarity, actionability]
    
    def _create_improved_metrics(self) -> List[Any]:
        """개선된 평가 메트릭 생성 (실험용)
        
        학술 연구 기반으로 설계된 새로운 메트릭 4개를 생성합니다.
        기존 _create_metrics()와는 독립적으로 작동하며, 실험 목적으로 사용됩니다.
        
        Returns:
            List[Any]: 개선된 DeepEval 메트릭 리스트
        """
        try:
            from deepeval.metrics.g_eval.g_eval import GEval
            from deepeval.test_case.llm_test_case import LLMTestCaseParams
        except ImportError:
            # DeepEval을 가져올 수 없는 경우 빈 리스트 반환
            return []
        
        model = "gemini-2.5-pro"
        
        # 1. Defect Detection Rate GEval (25%)
        defect_detection = GEval(
            name="DefectDetectionRate",
            model=model,
            evaluation_steps=[
                "코드에서 실제 버그, 보안 취약점, 성능 문제를 정확히 식별했는지 평가",
                "False Positive (잘못된 지적) 비율을 확인하고 정확한 지적만 있는지 검증",
                "Critical한 이슈를 놓치지 않았는지 검증 - 심각한 문제의 누락 여부 평가",
                "이슈의 우선순위가 실제 영향도와 일치하는지 평가 - 심각도 분류의 적절성",
                "제기된 문제가 실제 프로덕션 환경에서 발생할 수 있는 현실적인 이슈인지 확인"
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
            threshold=0.7,
        )
        
        # 2. Review Efficiency GEval (30%)
        review_efficiency = GEval(
            name="ReviewEfficiency",
            model=model,
            evaluation_steps=[
                "개발자가 리뷰를 이해하고 적용하는 데 필요한 노력과 시간을 평가",
                "불필요하거나 중복된 제안이 있는지 확인 - 효율성 저해 요소 식별",
                "제안의 구체성과 실용성 평가 - 실제 구현 가능한 구체적 가이드 제공 여부",
                "코드 품질 개선에 실질적으로 기여하는지 검증 - 의미있는 개선사항 여부",
                "개발 워크플로우를 방해하지 않고 자연스럽게 적용할 수 있는지 평가"
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
            threshold=0.7,
        )
        
        # 3. Context Awareness GEval (25%)
        context_awareness = GEval(
            name="ContextAwareness",
            model=model,
            evaluation_steps=[
                "전체 코드베이스와의 일관성을 고려한 리뷰인지 평가 - 시스템 전반의 이해도",
                "아키텍처 패턴과 설계 원칙 준수 여부 확인 - 기존 설계와의 호환성",
                "프로젝트 특성과 요구사항을 이해한 제안인지 검증 - 도메인 지식 반영",
                "기존 코드 스타일과 컨벤션과의 일치성 평가 - 팀 표준 준수",
                "다른 모듈이나 컴포넌트와의 상호작용을 고려한 제안인지 확인"
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
            threshold=0.7,
        )
        
        # 4. Impact Assessment GEval (20%)
        impact_assessment = GEval(
            name="ImpactAssessment",
            model=model,
            evaluation_steps=[
                "제안된 변경사항의 실제 영향도를 정확히 평가했는지 확인",
                "성능, 보안, 유지보수성에 대한 영향 분석의 정확성과 완전성 검증",
                "변경으로 인한 부작용이나 리스크를 식별하고 대안을 제시했는지 확인",
                "비용 대비 효과를 적절히 고려했는지 평가 - ROI 관점의 분석",
                "장기적인 기술 부채나 확장성에 미치는 영향을 고려했는지 검증"
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
            threshold=0.7,
        )
        
        return [defect_detection, review_efficiency, context_awareness, impact_assessment]
    
    def _load_test_cases(self, test_cases_file: Path) -> List[Any]:
        """테스트 케이스 JSON 파일 로드"""
        try:
            from deepeval.test_case.llm_test_case import LLMTestCase
        except ImportError:
            return []
        
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
    
    def _convert_display_filter_to_enum(self, display_filter: str, TestRunResultDisplay) -> Any:
        """문자열 display_filter를 TestRunResultDisplay enum으로 변환"""
        try:
            filter_mapping = {
                "all": TestRunResultDisplay.ALL,
                "failing": TestRunResultDisplay.FAILING,
                "passing": TestRunResultDisplay.PASSING
            }
            return filter_mapping.get(display_filter.lower(), TestRunResultDisplay.ALL)
        except Exception:
            # 변환 실패 시 기본값 반환
            return TestRunResultDisplay.ALL
    
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