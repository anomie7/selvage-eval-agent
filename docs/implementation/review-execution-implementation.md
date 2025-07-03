# 리뷰 실행 구현 가이드

## 개요

이 문서는 Selvage 평가 에이전트의 2단계인 리뷰 실행 기능의 구현을 다룹니다. 선별된 커밋들에 대해 다중 모델로 Selvage 리뷰를 실행하고 결과를 수집하는 기능을 구현합니다.

## 구현 목표

- 선별된 커밋들에 대해 Selvage 리뷰 실행
- 다중 모델 지원 (gemini-2.5-pro, claude-sonnet-4, claude-sonnet-4-thinking)
- 리뷰 결과 구조화된 저장
- 에러 처리 및 복구 메커니즘

## 핵심 구현 파일

### 1. 데이터 클래스 (src/selvage_eval/review_execution.py)

```python
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class ReviewExecutionSummary:
    """리뷰 실행 요약 (ToolResult.data 용)"""
    total_commits_reviewed: int
    total_reviews_executed: int
    total_successes: int
    total_failures: int
    execution_time_seconds: float
    output_directory: str
    success_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewExecutionSummary':
        return cls(**data)
    
    @property
    def summary_message(self) -> str:
        """실행 요약 메시지"""
        return (f"리뷰 완료: {self.total_commits_reviewed}개 커밋, "
                f"{self.total_reviews_executed}개 리뷰 ({self.success_rate:.1%} 성공)")
```

### 2. ReviewExecutor Tool (src/selvage_eval/tools/review_executor_tool.py)

```python
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
from ..review_execution import ReviewExecutionSummary
from ..utils.parameter_schema import generate_parameters_schema_from_hints


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
        return generate_parameters_schema_from_hints(self.execute)
    
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
                output_dir: str = '~/Library/selvage-eval-agent/review_logs', 
                model: str) -> ToolResult:
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
                        repo.repo_path, repo.repo_name, commit.commit_id, 
                        output_path, model
                    )
                    if success:
                        successes += 1
                    else:
                        failures += 1
                        
                except Exception as e:
                    print(f"커밋 {commit.commit_id} 리뷰 실행 중 오류: {str(e)}")
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
                    "command": (f"selvage review --no-print --target-commit {parent_commit_id} "
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
```

### 3. Tool 등록 (src/selvage_eval/tools/tool_generator.py)

```python
# 기존 코드에 다음 import 추가
from .review_executor_tool import ReviewExecutorTool

class ToolGenerator:
    def generate_tool(self, tool_name: str, params: Dict[str, Any]) -> Tool:
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
        elif tool_name == "execute_reviews":  # 새로 추가
            return ReviewExecutorTool()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
```

## 리뷰 결과 저장 구조

### 디렉토리 구조

```
~/Library/selvage-eval-agent/review_logs/
├── {repo_name}/
│   ├── {commit_id}/
│   │   ├── {model_name}/
│   │   │   └── {%Y%m%d_%H%M%S}_{model_name}_review_log.json
│   │   └── metadata.json
│   └── ...
```

### 리뷰 로그 JSON 구조

리뷰 로그 파일은 Selvage가 생성하는 JSON 형식을 따릅니다:

```json
{
  "id": "google-gemini-2.5-flash-preview-05-20-1750493437",
  "model": {
    "provider": "google",
    "name": "gemini-2.5-flash-preview-05-20"
  },
  "created_at": "2025-06-21T17:10:58.947588",
  "review_response": {
    "issues": [...],
    "summary": "리뷰 요약",
    "score": 9.0,
    "recommendations": [...]
  },
  "status": "SUCCESS",
  "error": null
}
```

## 에러 처리 및 복구

### 주요 에러 처리 항목

1. **파일 관련 에러**
   - meaningful_commits.json 파일 누락
   - 출력 디렉토리 생성 실패
   - 권한 부족

2. **Git 관련 에러**
   - 저장소 경로 무효
   - 커밋 체크아웃 실패
   - 부모 커밋 조회 실패

3. **Selvage 실행 에러**
   - 바이너리 누락
   - 명령어 실행 실패
   - 타임아웃 발생

### 복구 메커니즘

- Git 작업 후 항상 원래 브랜치로 복원
- 실패한 커밋은 건너뛰고 다음 커밋 계속 처리
- 에러 발생 시 상세한 로그 출력

## 사용 예시

### 기본 사용법

```python
from selvage_eval.tools.tool_executor import ToolExecutor

# ToolExecutor 인스턴스 생성
tool_executor = ToolExecutor()

# 리뷰 실행
result = tool_executor.execute_tool_call(
    "execute_reviews",
    {
        "meaningful_commits_path": "/path/to/meaningful_commits.json",
        "output_dir": "~/Library/selvage-eval-agent/review_logs",
        "model": "gemini-2.5-pro"
    }
)

if result.success:
    summary = result.data  # ReviewExecutionSummary 객체
    print(summary.summary_message)
    print(f"성공률: {summary.success_rate:.1%}")
else:
    print(f"리뷰 실행 실패: {result.error_message}")
```

### 다중 모델 병렬 실행

```python
import asyncio
from selvage_eval.tools.tool_executor import ToolExecutor

async def execute_multi_model_reviews():
    tool_executor = ToolExecutor()
    
    # 설정에서 모델 목록 로드
    review_models = ["gemini-2.5-pro", "claude-sonnet-4", "claude-sonnet-4-thinking"]
    
    # 각 모델별 병렬 실행
    tasks = [
        tool_executor.execute_tool_call(
            "execute_reviews",
            {
                "meaningful_commits_path": "/path/to/meaningful_commits.json",
                "output_dir": "~/Library/selvage-eval-agent/review_logs",
                "model": model
            }
        )
        for model in review_models
    ]
    
    results = await asyncio.gather(*tasks)
    
    # 결과 처리
    for i, result in enumerate(results):
        model = review_models[i]
        if result.success:
            print(f"{model}: {result.data.summary_message}")
        else:
            print(f"{model}: 실패 - {result.error_message}")

# 실행
asyncio.run(execute_multi_model_reviews())
```

## 테스트 방법

### 단위 테스트

```python
import pytest
from unittest.mock import Mock, patch
from selvage_eval.tools.review_executor_tool import ReviewExecutorTool

def test_review_executor_tool_validation():
    tool = ReviewExecutorTool()
    
    # 유효한 파라미터
    valid_params = {
        "meaningful_commits_path": "/path/to/file.json",
        "output_dir": "~/output",
        "model": "gemini-2.5-pro"
    }
    assert tool.validate_parameters(valid_params) == True
    
    # 필수 파라미터 누락
    invalid_params = {
        "meaningful_commits_path": "/path/to/file.json",
        "output_dir": "~/output"
    }
    assert tool.validate_parameters(invalid_params) == False

@patch('selvage_eval.commit_collection.MeaningfulCommitsData.from_json')
def test_review_executor_tool_execution(mock_from_json):
    # 테스트 데이터 설정
    mock_commits_data = Mock()
    mock_commits_data.repositories = []
    mock_from_json.return_value = mock_commits_data
    
    tool = ReviewExecutorTool()
    result = tool.execute(
        meaningful_commits_path="/path/to/test.json",
        output_dir="/tmp/test_output",
        model="test-model"
    )
    
    assert result.success == True
    assert result.data is not None
    assert isinstance(result.data, ReviewExecutionSummary)
```

### 통합 테스트

```python
def test_full_review_execution():
    """전체 리뷰 실행 프로세스 테스트"""
    # 1. 테스트용 meaningful_commits.json 생성
    # 2. ReviewExecutorTool 실행
    # 3. 결과 파일 생성 확인
    # 4. 정리
    pass
```

## 성능 최적화

### 메모리 사용량 최적화

- 대용량 diff 처리 시 스트리밍 방식 사용
- 커밋별 독립적 처리로 메모리 누수 방지

### 실행 시간 최적화

- 병렬 처리로 다중 모델 동시 실행
- Git 작업 최소화
- 타임아웃 설정으로 무한 대기 방지

## 모니터링 및 로깅

### 진행률 표시

```python
def _execute_repo_reviews(self, repo, output_path: Path, model: str) -> tuple[int, int]:
    total_commits = len(repo.commits)
    
    for i, commit in enumerate(repo.commits, 1):
        print(f"[{i}/{total_commits}] 리뷰 진행 중: {repo.repo_name}/{commit.commit_id}")
        # ... 리뷰 실행 코드
```

### 상세 로깅

```python
import logging

logger = logging.getLogger(__name__)

def _execute_single_review(self, ...):
    logger.info(f"리뷰 시작: {repo_name}/{commit_id} (모델: {model})")
    # ... 리뷰 실행 코드
    logger.info(f"리뷰 완료: {repo_name}/{commit_id}")
```

이 구현 가이드를 따라 리뷰 실행 기능을 구현하면 Selvage 평가 에이전트의 2단계가 완성됩니다.