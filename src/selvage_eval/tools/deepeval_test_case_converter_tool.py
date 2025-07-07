"""DeepEval 테스트 케이스 변환 도구

리뷰 로그를 DeepEval 형식으로 변환하는 도구입니다.
"""

import json
import subprocess
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
                        test_case_dict = self._test_case_to_dict(test_case)
                        test_case_dict['repo_name'] = log_info.repo_name
                        test_cases.append(test_case_dict)
                
                if test_cases:
                    # 저장 경로: session_id/model_name/test_cases.json
                    output_dir = session_dir / model_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_file = output_dir / "test_cases.json"
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(test_cases, f, ensure_ascii=False, indent=2)
                    
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
        return {
            "input": test_case.input,
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output
        }
    
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