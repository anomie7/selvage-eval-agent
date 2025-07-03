import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from ..tools.tool import Tool
from ..tools.tool_result import ToolResult
from ..tools.tool_executor import ToolExecutor
from ..commit_collection import MeaningfulCommitsData
from ..review_execution_summary import ReviewExecutionSummary
# from ..utils.parameter_schema import generate_parameters_schema_from_hints


class ReviewExecutorTool(Tool):
    """Selvage 리뷰 실행 도구"""
    
    def __init__(self):
        super().__init__()
        self.tool_executor = ToolExecutor()
    
    @property
    def name(self) -> str:
        return "execute_reviews"
    
    @property  
    def description(self) -> str:
        return "선별된 커밋들에 대해 다중 모델로 Selvage 리뷰를 실행하고 결과를 수집합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        # return generate_parameters_schema_from_hints(self.execute)
        return {
            "type": "object",
            "properties": {
                "meaningful_commits_path": {"type": "string"},
                "model": {"type": "string"},
                "output_dir": {"type": "string", "default": "~/Library/selvage-eval-agent/review_logs"}
            },
            "required": ["meaningful_commits_path", "model"]
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """파라미터 유효성 검증"""
        required_params = ['meaningful_commits_path', 'output_dir', 'model']
        for param in required_params:
            if param not in params:
                return False
            if not isinstance(params[param], str) or not params[param].strip():
                return False
                
        return True
    
    def execute(self, meaningful_commits_path: str, 
                model: str,
                output_dir: str = '~/Library/selvage-eval-agent/review_logs') -> ToolResult:
        """리뷰 실행
        
        Args:
            meaningful_commits_path: meaningful_commits.json 파일 경로
            output_dir: 리뷰 결과 저장 디렉토리 경로
            model: 사용할 리뷰 모델
            
        Returns:
            ToolResult: 리뷰 실행 결과 (ReviewExecutionSummary 포함)
        """
        start_time = time.time()
        
        try:
            # 1. meaningful_commits.json 파일 로드
            meaningful_commits = self._load_meaningful_commits(meaningful_commits_path)
            
            # 2. 출력 디렉토리 생성
            output_path = Path(output_dir).expanduser()
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 3. 각 저장소별 커밋 리뷰 실행
            total_commits = sum(len(repo.commits) for repo in meaningful_commits.repositories)
            total_successes = 0
            total_failures = 0
            
            for repo in meaningful_commits.repositories:
                repo_successes, repo_failures = self._execute_repo_reviews(
                    repo, output_path, model
                )
                total_successes += repo_successes
                total_failures += repo_failures
            
            # 4. 실행 요약 생성
            execution_time = time.time() - start_time
            success_rate = total_successes / total_commits if total_commits > 0 else 0.0
            
            summary = ReviewExecutionSummary(
                total_commits_reviewed=total_commits,
                total_reviews_executed=total_successes + total_failures,
                total_successes=total_successes,
                total_failures=total_failures,
                execution_time_seconds=execution_time,
                output_directory=str(output_path),
                success_rate=success_rate
            )
            
            return ToolResult(
                success=True,
                data=summary,
                error_message=None
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"리뷰 실행 중 오류 발생: {str(e)}"
            )
    
    def _load_meaningful_commits(self, file_path: str) -> MeaningfulCommitsData:
        """meaningful_commits.json 파일 로드"""
        try:
            return MeaningfulCommitsData.from_json(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"meaningful_commits.json 파일을 찾을 수 없습니다: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 파일 파싱 오류: {str(e)}")
    
    def _execute_repo_reviews(self, repo, output_path: Path, model: str) -> tuple[int, int]:
        """저장소별 리뷰 실행"""
        successes = 0
        failures = 0
        
        # 현재 브랜치 저장
        current_branch = self._get_current_branch(repo.repo_path)
        
        try:
            for commit in repo.commits:
                try:
                    success = self._execute_single_review(
                        repo.repo_path, repo.repo_name, commit.id, 
                        output_path, model
                    )
                    if success:
                        successes += 1
                    else:
                        failures += 1
                        
                except Exception as e:
                    print(f"커밋 {commit.id} 리뷰 실행 중 오류: {str(e)}")
                    failures += 1
                    
        finally:
            # 원래 브랜치로 복원
            self._restore_branch(repo.repo_path, current_branch)
            
        return successes, failures
    
    def _execute_single_review(self, repo_path: str, repo_name: str, 
                               commit_id: str, output_path: Path, model: str) -> bool:
        """단일 커밋 리뷰 실행"""
        try:
            # 1. 커밋 체크아웃
            checkout_result = self.tool_executor.execute_tool_call(
                "execute_safe_command",
                {
                    "command": f"git checkout {commit_id}",
                    "cwd": repo_path,
                    "capture_output": True,
                    "timeout": 60
                }
            )
            
            if not checkout_result.success:
                print(f"커밋 체크아웃 실패: {commit_id}")
                return False
            
            # 2. 부모 커밋 ID 조회
            parent_result = self.tool_executor.execute_tool_call(
                "execute_safe_command",
                {
                    "command": "git rev-parse HEAD^",
                    "cwd": repo_path,
                    "capture_output": True,
                    "timeout": 60
                }
            )
            
            if not parent_result.success:
                print(f"부모 커밋 조회 실패: {commit_id}")
                return False
            
            parent_commit_id = parent_result.data["stdout"].strip()
            
            # 3. 리뷰 로그 디렉토리 생성
            review_log_dir = output_path / repo_name / commit_id / model
            review_log_dir.mkdir(parents=True, exist_ok=True)
            
            # 4. Selvage 리뷰 실행
            review_result = self.tool_executor.execute_tool_call(
                "execute_safe_command",
                {
                    "command": (f"selvage review --no-print --skip-cache --target-commit {parent_commit_id} "
                               f"--model {model} --log-dir {review_log_dir}"),
                    "cwd": repo_path,
                    "capture_output": True,
                    "timeout": 300
                }
            )
            
            if not review_result.success:
                print(f"Selvage 리뷰 실행 실패: {commit_id} (모델: {model})")
                return False
            
            print(f"리뷰 완료: {repo_name}/{commit_id} (모델: {model})")
            return True
            
        except Exception as e:
            print(f"리뷰 실행 중 예외 발생: {commit_id} - {str(e)}")
            return False
    
    def _get_current_branch(self, repo_path: str) -> str:
        """현재 브랜치 조회"""
        result = self.tool_executor.execute_tool_call(
            "execute_safe_command",
            {
                "command": "git branch --show-current",
                "cwd": repo_path,
                "capture_output": True,
                "timeout": 30
            }
        )
        
        if result.success:
            return result.data["stdout"].strip()
        else:
            return "main"  # 기본값
    
    def _restore_branch(self, repo_path: str, branch: str):
        """원래 브랜치로 복원"""
        self.tool_executor.execute_tool_call(
            "execute_safe_command",
            {
                "command": f"git checkout {branch}",
                "cwd": repo_path,
                "capture_output": True,
                "timeout": 60
            }
        )