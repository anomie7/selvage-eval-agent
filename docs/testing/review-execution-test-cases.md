# 리뷰 실행 단계 - 테스트 케이스

본 문서는 [`review-execution-implementation.md`](../implementation/review-execution-implementation.md) 구현을 위한 포괄적인 테스트 케이스입니다.

## 1. 테스트 환경 설정

```python
"""테스트 의존성 및 픽스처"""
import pytest
import json
import tempfile
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path

# 테스트 대상 클래스들
from selvage_eval.tools.review_executor_tool import ReviewExecutorTool
from selvage_eval.review_execution import ReviewExecutionSummary
from selvage_eval.commit_collection import (
    CommitStats, CommitScore, CommitData, RepositoryMetadata,
    RepositoryResult, MeaningfulCommitsData
)
from selvage_eval.tools.tool_executor import ToolExecutor
from selvage_eval.tools.tool_result import ToolResult


@pytest.fixture
def temp_output_dir():
    """임시 출력 디렉토리"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_commit_data():
    """샘플 커밋 데이터"""
    return CommitData(
        id="abc123",
        message="fix: resolve memory leak in parser",
        author="developer@example.com",
        date=datetime(2024, 1, 15, 10, 30, 0),
        stats=CommitStats(files_changed=3, lines_added=45, lines_deleted=12),
        score=CommitScore(85, -2, 15, 10, 10, 15, 0),
        file_paths=["src/parser.py", "src/memory.py", "tests/test_parser.py"]
    )


@pytest.fixture
def sample_repo_metadata():
    """샘플 저장소 메타데이터"""
    return RepositoryMetadata(
        total_commits=150,
        filtered_commits=25,
        selected_commits=10,
        filter_timestamp=datetime(2024, 1, 20, 15, 0, 0),
        processing_time_seconds=5.2
    )


@pytest.fixture
def sample_repository_result(sample_commit_data, sample_repo_metadata):
    """샘플 저장소 결과"""
    return RepositoryResult(
        repo_name="test-repo",
        repo_path="/tmp/test-repo",
        commits=[sample_commit_data],
        metadata=sample_repo_metadata
    )


@pytest.fixture
def sample_meaningful_commits(sample_repository_result):
    """샘플 MeaningfulCommitsData"""
    return MeaningfulCommitsData(repositories=[sample_repository_result])


@pytest.fixture
def meaningful_commits_json_file(sample_meaningful_commits, temp_output_dir):
    """테스트용 meaningful_commits.json 파일"""
    json_file = temp_output_dir / "meaningful_commits.json"
    sample_meaningful_commits.save_to_json(str(json_file))
    return str(json_file)


@pytest.fixture
def mock_tool_executor():
    """Mock ToolExecutor"""
    mock = Mock(spec=ToolExecutor)
    return mock


@pytest.fixture
def sample_execution_summary():
    """샘플 실행 요약"""
    return ReviewExecutionSummary(
        total_commits_reviewed=10,
        total_reviews_executed=10,
        total_successes=8,
        total_failures=2,
        execution_time_seconds=120.5,
        output_directory="/tmp/test_output",
        success_rate=0.8
    )
```

## 2. 데이터 클래스 단위 테스트

### 2.1 ReviewExecutionSummary 테스트

```python
class TestReviewExecutionSummary:
    """ReviewExecutionSummary 클래스 테스트"""
    
    def test_creation(self, sample_execution_summary):
        """정상적인 객체 생성 테스트"""
        assert sample_execution_summary.total_commits_reviewed == 10
        assert sample_execution_summary.total_reviews_executed == 10
        assert sample_execution_summary.total_successes == 8
        assert sample_execution_summary.total_failures == 2
        assert sample_execution_summary.execution_time_seconds == 120.5
        assert sample_execution_summary.output_directory == "/tmp/test_output"
        assert sample_execution_summary.success_rate == 0.8
    
    def test_to_dict_conversion(self, sample_execution_summary):
        """딕셔너리 변환 테스트"""
        data_dict = sample_execution_summary.to_dict()
        
        assert data_dict['total_commits_reviewed'] == 10
        assert data_dict['total_reviews_executed'] == 10
        assert data_dict['total_successes'] == 8
        assert data_dict['total_failures'] == 2
        assert data_dict['execution_time_seconds'] == 120.5
        assert data_dict['output_directory'] == "/tmp/test_output"
        assert data_dict['success_rate'] == 0.8
    
    def test_from_dict_conversion(self, sample_execution_summary):
        """딕셔너리에서 생성 테스트"""
        data_dict = sample_execution_summary.to_dict()
        recreated = ReviewExecutionSummary.from_dict(data_dict)
        
        assert recreated.total_commits_reviewed == sample_execution_summary.total_commits_reviewed
        assert recreated.total_reviews_executed == sample_execution_summary.total_reviews_executed
        assert recreated.total_successes == sample_execution_summary.total_successes
        assert recreated.total_failures == sample_execution_summary.total_failures
        assert recreated.execution_time_seconds == sample_execution_summary.execution_time_seconds
        assert recreated.output_directory == sample_execution_summary.output_directory
        assert recreated.success_rate == sample_execution_summary.success_rate
    
    def test_summary_message(self, sample_execution_summary):
        """요약 메시지 테스트"""
        message = sample_execution_summary.summary_message
        
        assert "10개 커밋" in message
        assert "10개 리뷰" in message
        assert "80.0% 성공" in message
    
    def test_perfect_success_rate(self):
        """100% 성공률 테스트"""
        summary = ReviewExecutionSummary(
            total_commits_reviewed=5,
            total_reviews_executed=5,
            total_successes=5,
            total_failures=0,
            execution_time_seconds=60.0,
            output_directory="/tmp/output",
            success_rate=1.0
        )
        
        assert summary.success_rate == 1.0
        assert "100.0% 성공" in summary.summary_message
    
    def test_zero_success_rate(self):
        """0% 성공률 테스트"""
        summary = ReviewExecutionSummary(
            total_commits_reviewed=3,
            total_reviews_executed=3,
            total_successes=0,
            total_failures=3,
            execution_time_seconds=30.0,
            output_directory="/tmp/output",
            success_rate=0.0
        )
        
        assert summary.success_rate == 0.0
        assert "0.0% 성공" in summary.summary_message
```

## 3. ReviewExecutorTool 단위 테스트

### 3.1 초기화 및 기본 기능 테스트

```python
class TestReviewExecutorTool:
    """ReviewExecutorTool 클래스 테스트"""
    
    def test_initialization(self):
        """정상적인 초기화 테스트"""
        tool = ReviewExecutorTool()
        
        assert tool.name == "execute_reviews"
        assert "다중 모델로 Selvage 리뷰를 실행" in tool.description
        assert tool.tool_executor is not None
        assert tool.parameters_schema is not None
    
    def test_validate_parameters_success(self):
        """파라미터 유효성 검증 성공 테스트"""
        tool = ReviewExecutorTool()
        
        valid_params = {
            "meaningful_commits_path": "/path/to/file.json",
            "output_dir": "~/output",
            "model": "gemini-2.5-pro"
        }
        
        assert tool.validate_parameters(valid_params) is True
    
    def test_validate_parameters_missing_required(self):
        """필수 파라미터 누락 테스트"""
        tool = ReviewExecutorTool()
        
        # model 파라미터 누락
        invalid_params = {
            "meaningful_commits_path": "/path/to/file.json",
            "output_dir": "~/output"
        }
        
        assert tool.validate_parameters(invalid_params) is False
    
    def test_validate_parameters_empty_string(self):
        """빈 문자열 파라미터 테스트"""
        tool = ReviewExecutorTool()
        
        invalid_params = {
            "meaningful_commits_path": "",
            "output_dir": "~/output",
            "model": "gemini-2.5-pro"
        }
        
        assert tool.validate_parameters(invalid_params) is False
    
    def test_validate_parameters_wrong_type(self):
        """잘못된 타입 파라미터 테스트"""
        tool = ReviewExecutorTool()
        
        invalid_params = {
            "meaningful_commits_path": 123,  # int 타입
            "output_dir": "~/output",
            "model": "gemini-2.5-pro"
        }
        
        assert tool.validate_parameters(invalid_params) is False


class TestMeaningfulCommitsLoading:
    """MeaningfulCommits 로딩 테스트"""
    
    def test_load_meaningful_commits_success(self, meaningful_commits_json_file):
        """JSON 파일 로딩 성공 테스트"""
        tool = ReviewExecutorTool()
        
        result = tool._load_meaningful_commits(meaningful_commits_json_file)
        
        assert isinstance(result, MeaningfulCommitsData)
        assert len(result.repositories) == 1
        assert result.repositories[0].repo_name == "test-repo"
    
    def test_load_meaningful_commits_file_not_found(self):
        """존재하지 않는 파일 로딩 테스트"""
        tool = ReviewExecutorTool()
        
        with pytest.raises(FileNotFoundError):
            tool._load_meaningful_commits("/nonexistent/file.json")
    
    def test_load_meaningful_commits_invalid_json(self, temp_output_dir):
        """잘못된 JSON 파일 로딩 테스트"""
        tool = ReviewExecutorTool()
        
        # 잘못된 JSON 파일 생성
        invalid_json_file = temp_output_dir / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write("{ invalid json content")
        
        with pytest.raises(ValueError):
            tool._load_meaningful_commits(str(invalid_json_file))


class TestGitOperations:
    """Git 작업 관련 테스트"""
    
    def test_get_current_branch_success(self, mock_tool_executor):
        """현재 브랜치 조회 성공 테스트"""
        tool = ReviewExecutorTool()
        tool.tool_executor = mock_tool_executor
        
        # Mock 설정
        mock_tool_executor.execute_tool_call.return_value = ToolResult(
            success=True,
            data={'stdout': 'main', 'stderr': '', 'returncode': 0},
            error_message=None
        )
        
        branch = tool._get_current_branch("/tmp/repo")
        
        assert branch == "main"
        mock_tool_executor.execute_tool_call.assert_called_once_with(
            "execute_safe_command",
            {
                "command": "git branch --show-current",
                "cwd": "/tmp/repo",
                "capture_output": True,
                "timeout": 30
            }
        )
    
    def test_get_current_branch_failure(self, mock_tool_executor):
        """현재 브랜치 조회 실패 테스트"""
        tool = ReviewExecutorTool()
        tool.tool_executor = mock_tool_executor
        
        # Mock 설정
        mock_tool_executor.execute_tool_call.return_value = ToolResult(
            success=False,
            data=None,
            error_message="Git command failed"
        )
        
        branch = tool._get_current_branch("/tmp/repo")
        
        assert branch == "main"  # 기본값 반환
    
    def test_restore_branch(self, mock_tool_executor):
        """브랜치 복원 테스트"""
        tool = ReviewExecutorTool()
        tool.tool_executor = mock_tool_executor
        
        tool._restore_branch("/tmp/repo", "feature-branch")
        
        mock_tool_executor.execute_tool_call.assert_called_once_with(
            "execute_safe_command",
            {
                "command": "git checkout feature-branch",
                "cwd": "/tmp/repo",
                "capture_output": True,
                "timeout": 60
            }
        )


class TestSingleReviewExecution:
    """단일 리뷰 실행 테스트"""
    
    def test_execute_single_review_success(self, mock_tool_executor, temp_output_dir):
        """단일 커밋 리뷰 성공 테스트"""
        tool = ReviewExecutorTool()
        tool.tool_executor = mock_tool_executor
        
        # Mock 설정 - 모든 Git 명령어 성공
        def mock_execute_side_effect(tool_name, params):
            if "git checkout" in params["command"]:
                return ToolResult(True, {'stdout': '', 'returncode': 0}, None)
            elif "git rev-parse" in params["command"]:
                return ToolResult(True, {'stdout': 'parent123', 'returncode': 0}, None)
            elif "selvage review" in params["command"]:
                return ToolResult(True, {'stdout': 'Review completed', 'returncode': 0}, None)
            else:
                return ToolResult(False, None, "Unknown command")
        
        mock_tool_executor.execute_tool_call.side_effect = mock_execute_side_effect
        
        result = tool._execute_single_review(
            "/tmp/repo", "test-repo", "abc123", temp_output_dir, "gemini-2.5-pro"
        )
        
        assert result is True
        
        # Git 명령어 호출 확인
        calls = mock_tool_executor.execute_tool_call.call_args_list
        assert len(calls) == 3  # checkout, rev-parse, selvage review
    
    def test_execute_single_review_checkout_failure(self, mock_tool_executor, temp_output_dir):
        """커밋 체크아웃 실패 테스트"""
        tool = ReviewExecutorTool()
        tool.tool_executor = mock_tool_executor
        
        # Git checkout 실패 Mock
        mock_tool_executor.execute_tool_call.return_value = ToolResult(
            success=False,
            data=None,
            error_message="Commit not found"
        )
        
        result = tool._execute_single_review(
            "/tmp/repo", "test-repo", "nonexistent", temp_output_dir, "gemini-2.5-pro"
        )
        
        assert result is False
    
    def test_execute_single_review_parent_commit_failure(self, mock_tool_executor, temp_output_dir):
        """부모 커밋 조회 실패 테스트"""
        tool = ReviewExecutorTool()
        tool.tool_executor = mock_tool_executor
        
        # Mock 설정 - checkout 성공, rev-parse 실패
        def mock_execute_side_effect(tool_name, params):
            if "git checkout" in params["command"]:
                return ToolResult(True, {'stdout': '', 'returncode': 0}, None)
            elif "git rev-parse" in params["command"]:
                return ToolResult(False, None, "No parent commit")
            else:
                return ToolResult(False, None, "Unknown command")
        
        mock_tool_executor.execute_tool_call.side_effect = mock_execute_side_effect
        
        result = tool._execute_single_review(
            "/tmp/repo", "test-repo", "abc123", temp_output_dir, "gemini-2.5-pro"
        )
        
        assert result is False
    
    def test_execute_single_review_selvage_failure(self, mock_tool_executor, temp_output_dir):
        """Selvage 실행 실패 테스트"""
        tool = ReviewExecutorTool()
        tool.tool_executor = mock_tool_executor
        
        # Mock 설정 - Git 성공, Selvage 실패
        def mock_execute_side_effect(tool_name, params):
            if "git checkout" in params["command"]:
                return ToolResult(True, {'stdout': '', 'returncode': 0}, None)
            elif "git rev-parse" in params["command"]:
                return ToolResult(True, {'stdout': 'parent123', 'returncode': 0}, None)
            elif "selvage review" in params["command"]:
                return ToolResult(False, None, "Selvage execution failed")
            else:
                return ToolResult(False, None, "Unknown command")
        
        mock_tool_executor.execute_tool_call.side_effect = mock_execute_side_effect
        
        result = tool._execute_single_review(
            "/tmp/repo", "test-repo", "abc123", temp_output_dir, "gemini-2.5-pro"
        )
        
        assert result is False
    
    def test_execute_single_review_exception_handling(self, mock_tool_executor, temp_output_dir):
        """예외 발생 처리 테스트"""
        tool = ReviewExecutorTool()
        tool.tool_executor = mock_tool_executor
        
        # Mock에서 예외 발생
        mock_tool_executor.execute_tool_call.side_effect = Exception("Unexpected error")
        
        result = tool._execute_single_review(
            "/tmp/repo", "test-repo", "abc123", temp_output_dir, "gemini-2.5-pro"
        )
        
        assert result is False
```

## 4. 리뷰 실행 프로세스 테스트

```python
class TestRepositoryReviewExecution:
    """저장소별 리뷰 실행 테스트"""
    
    def test_execute_repo_reviews_success(self, mock_tool_executor, sample_repository_result, temp_output_dir):
        """저장소 리뷰 실행 성공 테스트"""
        tool = ReviewExecutorTool()
        tool.tool_executor = mock_tool_executor
        
        # Mock 설정
        with patch.object(tool, '_get_current_branch', return_value='main'), \
             patch.object(tool, '_execute_single_review', return_value=True), \
             patch.object(tool, '_restore_branch'):
            
            successes, failures = tool._execute_repo_reviews(
                sample_repository_result, temp_output_dir, "gemini-2.5-pro"
            )
        
        assert successes == 1
        assert failures == 0
    
    def test_execute_repo_reviews_with_failures(self, mock_tool_executor, sample_repository_result, temp_output_dir):
        """저장소 리뷰 실행 중 실패 포함 테스트"""
        tool = ReviewExecutorTool()
        tool.tool_executor = mock_tool_executor
        
        # Mock 설정 - 리뷰 실행 실패
        with patch.object(tool, '_get_current_branch', return_value='main'), \
             patch.object(tool, '_execute_single_review', return_value=False), \
             patch.object(tool, '_restore_branch'):
            
            successes, failures = tool._execute_repo_reviews(
                sample_repository_result, temp_output_dir, "gemini-2.5-pro"
            )
        
        assert successes == 0
        assert failures == 1
    
    def test_execute_repo_reviews_exception_handling(self, mock_tool_executor, sample_repository_result, temp_output_dir):
        """저장소 리뷰 실행 중 예외 처리 테스트"""
        tool = ReviewExecutorTool()
        tool.tool_executor = mock_tool_executor
        
        # Mock 설정 - 예외 발생
        with patch.object(tool, '_get_current_branch', return_value='main'), \
             patch.object(tool, '_execute_single_review', side_effect=Exception("Unexpected error")), \
             patch.object(tool, '_restore_branch'):
            
            successes, failures = tool._execute_repo_reviews(
                sample_repository_result, temp_output_dir, "gemini-2.5-pro"
            )
        
        assert successes == 0
        assert failures == 1
    
    def test_execute_repo_reviews_branch_restoration(self, mock_tool_executor, sample_repository_result, temp_output_dir):
        """브랜치 복원 보장 테스트"""
        tool = ReviewExecutorTool()
        tool.tool_executor = mock_tool_executor
        
        original_branch = 'feature-branch'
        
        with patch.object(tool, '_get_current_branch', return_value=original_branch), \
             patch.object(tool, '_execute_single_review', side_effect=Exception("Error during review")), \
             patch.object(tool, '_restore_branch') as mock_restore:
            
            tool._execute_repo_reviews(sample_repository_result, temp_output_dir, "gemini-2.5-pro")
            
            # 예외 발생해도 브랜치 복원이 호출되었는지 확인
            mock_restore.assert_called_once_with(sample_repository_result.repo_path, original_branch)


class TestFullExecution:
    """전체 실행 프로세스 테스트"""
    
    def test_execute_success(self, meaningful_commits_json_file, temp_output_dir):
        """전체 실행 성공 테스트"""
        tool = ReviewExecutorTool()
        
        with patch.object(tool, '_execute_repo_reviews', return_value=(1, 0)):
            result = tool.execute(
                meaningful_commits_path=meaningful_commits_json_file,
                output_dir=str(temp_output_dir),
                model="gemini-2.5-pro"
            )
        
        assert result.success is True
        assert isinstance(result.data, ReviewExecutionSummary)
        assert result.data.total_commits_reviewed == 1
        assert result.data.total_successes == 1
        assert result.data.total_failures == 0
        assert result.data.success_rate == 1.0
        assert result.error_message is None
    
    def test_execute_with_failures(self, meaningful_commits_json_file, temp_output_dir):
        """실패 포함 전체 실행 테스트"""
        tool = ReviewExecutorTool()
        
        with patch.object(tool, '_execute_repo_reviews', return_value=(0, 1)):
            result = tool.execute(
                meaningful_commits_path=meaningful_commits_json_file,
                output_dir=str(temp_output_dir),
                model="gemini-2.5-pro"
            )
        
        assert result.success is True
        assert result.data.total_successes == 0
        assert result.data.total_failures == 1
        assert result.data.success_rate == 0.0
    
    def test_execute_file_not_found(self, temp_output_dir):
        """JSON 파일 없음 오류 테스트"""
        tool = ReviewExecutorTool()
        
        result = tool.execute(
            meaningful_commits_path="/nonexistent/file.json",
            output_dir=str(temp_output_dir),
            model="gemini-2.5-pro"
        )
        
        assert result.success is False
        assert result.data is None
        assert "meaningful_commits.json 파일을 찾을 수 없습니다" in result.error_message
    
    def test_execute_output_directory_creation(self, meaningful_commits_json_file, temp_output_dir):
        """출력 디렉토리 생성 테스트"""
        tool = ReviewExecutorTool()
        
        # 존재하지 않는 하위 디렉토리 지정
        nested_output_dir = temp_output_dir / "nested" / "output"
        
        with patch.object(tool, '_execute_repo_reviews', return_value=(1, 0)):
            result = tool.execute(
                meaningful_commits_path=meaningful_commits_json_file,
                output_dir=str(nested_output_dir),
                model="gemini-2.5-pro"
            )
        
        assert result.success is True
        assert nested_output_dir.exists()
        assert nested_output_dir.is_dir()
```

## 5. 에러 처리 및 복구 테스트

```python
class TestErrorHandling:
    """에러 처리 테스트"""
    
    def test_invalid_json_content(self, temp_output_dir):
        """잘못된 JSON 내용 처리 테스트"""
        tool = ReviewExecutorTool()
        
        # 잘못된 JSON 파일 생성
        invalid_json = temp_output_dir / "invalid.json"
        with open(invalid_json, 'w') as f:
            f.write('{"invalid": json content}')  # 잘못된 구조
        
        result = tool.execute(
            meaningful_commits_path=str(invalid_json),
            output_dir=str(temp_output_dir),
            model="gemini-2.5-pro"
        )
        
        assert result.success is False
        assert result.data is None
        assert "JSON 파일 파싱 오류" in result.error_message or "리뷰 실행 중 오류 발생" in result.error_message
    
    def test_permission_denied_output_directory(self, meaningful_commits_json_file):
        """출력 디렉토리 권한 부족 테스트"""
        tool = ReviewExecutorTool()
        
        # 권한이 없는 디렉토리 (일반적으로 /root)
        restricted_dir = "/root/restricted_output"
        
        result = tool.execute(
            meaningful_commits_path=meaningful_commits_json_file,
            output_dir=restricted_dir,
            model="gemini-2.5-pro"
        )
        
        # 권한 부족으로 실패할 수 있음 (환경에 따라 다름)
        if not result.success:
            assert result.data is None
            assert "리뷰 실행 중 오류 발생" in result.error_message
    
    def test_empty_repositories_list(self, temp_output_dir):
        """빈 저장소 목록 처리 테스트"""
        tool = ReviewExecutorTool()
        
        # 빈 저장소 목록 JSON 생성
        empty_commits = MeaningfulCommitsData(repositories=[])
        empty_json = temp_output_dir / "empty.json"
        empty_commits.save_to_json(str(empty_json))
        
        result = tool.execute(
            meaningful_commits_path=str(empty_json),
            output_dir=str(temp_output_dir),
            model="gemini-2.5-pro"
        )
        
        assert result.success is True
        assert result.data.total_commits_reviewed == 0
        assert result.data.total_successes == 0
        assert result.data.total_failures == 0
        assert result.data.success_rate == 0.0
    
    def test_unexpected_exception_during_execution(self, meaningful_commits_json_file, temp_output_dir):
        """실행 중 예상치 못한 예외 처리 테스트"""
        tool = ReviewExecutorTool()
        
        # _execute_repo_reviews에서 예외 발생 시뮬레이션
        with patch.object(tool, '_execute_repo_reviews', side_effect=Exception("Unexpected system error")):
            result = tool.execute(
                meaningful_commits_path=meaningful_commits_json_file,
                output_dir=str(temp_output_dir),
                model="gemini-2.5-pro"
            )
        
        assert result.success is False
        assert result.data is None
        assert "리뷰 실행 중 오류 발생" in result.error_message
        assert "Unexpected system error" in result.error_message
```

## 6. 통합 테스트

```python
class TestIntegration:
    """통합 테스트"""
    
    def test_full_workflow_with_temp_directory(self, sample_commit_data, sample_repo_metadata):
        """임시 디렉토리를 사용한 전체 워크플로우 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. 테스트 데이터 준비
            repo_result = RepositoryResult(
                repo_name="integration-test-repo",
                repo_path="/tmp/integration-test",
                commits=[sample_commit_data],
                metadata=sample_repo_metadata
            )
            
            meaningful_commits = MeaningfulCommitsData(repositories=[repo_result])
            
            # 2. JSON 파일 저장
            json_file = temp_path / "test_commits.json"
            meaningful_commits.save_to_json(str(json_file))
            
            # 3. 출력 디렉토리 설정
            output_dir = temp_path / "review_output"
            
            # 4. ReviewExecutorTool 실행
            tool = ReviewExecutorTool()
            
            with patch.object(tool, '_execute_repo_reviews', return_value=(1, 0)):
                result = tool.execute(
                    meaningful_commits_path=str(json_file),
                    output_dir=str(output_dir),
                    model="test-model"
                )
            
            # 5. 결과 검증
            assert result.success is True
            assert isinstance(result.data, ReviewExecutionSummary)
            assert result.data.total_commits_reviewed == 1
            assert result.data.success_rate == 1.0
            
            # 6. 출력 디렉토리 생성 확인
            assert output_dir.exists()
            assert output_dir.is_dir()
    
    def test_multiple_repositories_workflow(self, sample_commit_data, sample_repo_metadata):
        """다중 저장소 워크플로우 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 다중 저장소 데이터 생성
            repos = []
            for i in range(3):
                repo = RepositoryResult(
                    repo_name=f"repo-{i}",
                    repo_path=f"/tmp/repo-{i}",
                    commits=[sample_commit_data],  # 각 저장소에 1개 커밋
                    metadata=sample_repo_metadata
                )
                repos.append(repo)
            
            meaningful_commits = MeaningfulCommitsData(repositories=repos)
            
            # JSON 파일 저장
            json_file = temp_path / "multi_repo_commits.json"
            meaningful_commits.save_to_json(str(json_file))
            
            # ReviewExecutorTool 실행
            tool = ReviewExecutorTool()
            
            with patch.object(tool, '_execute_repo_reviews', return_value=(1, 0)):
                result = tool.execute(
                    meaningful_commits_path=str(json_file),
                    output_dir=str(temp_path / "output"),
                    model="test-model"
                )
            
            # 총 3개 커밋이 처리되었는지 확인
            assert result.success is True
            assert result.data.total_commits_reviewed == 3
            assert result.data.total_successes == 3
            assert result.data.total_failures == 0
    
    def test_review_log_files_creation(self, sample_commit_data, sample_repo_metadata):
        """리뷰 로그 파일 생성 확인 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 테스트 데이터 준비
            repo_result = RepositoryResult(
                repo_name="log-test-repo",
                repo_path="/tmp/log-test",
                commits=[sample_commit_data],
                metadata=sample_repo_metadata
            )
            
            meaningful_commits = MeaningfulCommitsData(repositories=[repo_result])
            json_file = temp_path / "commits.json"
            meaningful_commits.save_to_json(str(json_file))
            
            output_dir = temp_path / "logs"
            
            # Mock으로 실제 파일 생성 시뮬레이션
            def mock_execute_single_review(repo_path, repo_name, commit_id, output_path, model):
                # 예상되는 디렉토리 구조 생성
                log_dir = output_path / repo_name / commit_id / model
                log_dir.mkdir(parents=True, exist_ok=True)
                
                # 더미 로그 파일 생성
                log_file = log_dir / f"test_review_log.json"
                log_file.write_text('{"test": "log"}')
                
                return True
            
            tool = ReviewExecutorTool()
            
            with patch.object(tool, '_execute_single_review', side_effect=mock_execute_single_review), \
                 patch.object(tool, '_get_current_branch', return_value='main'), \
                 patch.object(tool, '_restore_branch'):
                
                result = tool.execute(
                    meaningful_commits_path=str(json_file),
                    output_dir=str(output_dir),
                    model="test-model"
                )
            
            # 결과 확인
            assert result.success is True
            
            # 디렉토리 구조 확인
            expected_log_dir = output_dir / "log-test-repo" / "abc123" / "test-model"
            assert expected_log_dir.exists()
            assert expected_log_dir.is_dir()
            
            # 로그 파일 존재 확인 (내용은 검증하지 않음)
            log_files = list(expected_log_dir.glob("*.json"))
            assert len(log_files) > 0


class TestMultiModelParallelExecution:
    """다중 모델 병렬 실행 테스트"""
    
    def test_single_commit_multiple_models(self, sample_commit_data, sample_repo_metadata):
        """1개 커밋에 대한 다중 모델 실행 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 단일 커밋 데이터 준비
            repo_result = RepositoryResult(
                repo_name="multi-model-repo",
                repo_path="/tmp/multi-model",
                commits=[sample_commit_data],  # 1개 커밋만
                metadata=sample_repo_metadata
            )
            
            meaningful_commits = MeaningfulCommitsData(repositories=[repo_result])
            json_file = temp_path / "single_commit.json"
            meaningful_commits.save_to_json(str(json_file))
            
            # 3개 모델로 테스트
            models = ["gemini-2.5-pro", "claude-sonnet-4", "claude-sonnet-4-thinking"]
            results = []
            
            for model in models:
                tool = ReviewExecutorTool()
                
                with patch.object(tool, '_execute_repo_reviews', return_value=(1, 0)):
                    result = tool.execute(
                        meaningful_commits_path=str(json_file),
                        output_dir=str(temp_path / f"output_{model}"),
                        model=model
                    )
                    results.append(result)
            
            # 모든 모델 실행 결과 확인
            for i, result in enumerate(results):
                assert result.success is True
                assert result.data.total_commits_reviewed == 1
                assert result.data.total_successes == 1
                assert result.data.total_failures == 0
                
                # 각 모델별 출력 디렉토리 확인
                model_output_dir = temp_path / f"output_{models[i]}"
                assert model_output_dir.exists()
    
    @pytest.mark.asyncio
    async def test_async_parallel_execution_simulation(self, sample_commit_data, sample_repo_metadata):
        """비동기 병렬 실행 시뮬레이션 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 테스트 데이터 준비
            repo_result = RepositoryResult(
                repo_name="async-test-repo",
                repo_path="/tmp/async-test",
                commits=[sample_commit_data],
                metadata=sample_repo_metadata
            )
            
            meaningful_commits = MeaningfulCommitsData(repositories=[repo_result])
            json_file = temp_path / "async_commits.json"
            meaningful_commits.save_to_json(str(json_file))
            
            # 비동기 실행 함수
            async def execute_review_async(model):
                tool = ReviewExecutorTool()
                
                with patch.object(tool, '_execute_repo_reviews', return_value=(1, 0)):
                    return tool.execute(
                        meaningful_commits_path=str(json_file),
                        output_dir=str(temp_path / f"async_output_{model}"),
                        model=model
                    )
            
            # 3개 모델 병렬 실행
            models = ["model-1", "model-2", "model-3"]
            tasks = [execute_review_async(model) for model in models]
            results = await asyncio.gather(*tasks)
            
            # 모든 결과 확인
            assert len(results) == 3
            for result in results:
                assert result.success is True
                assert result.data.total_commits_reviewed == 1
```

## 7. 성능 및 에지 케이스 테스트

```python
class TestPerformance:
    """성능 테스트"""
    
    def test_large_commit_list_performance(self, sample_repo_metadata):
        """대량 커밋 처리 성능 테스트"""
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 100개 커밋 생성
            large_commit_list = []
            for i in range(100):
                commit = CommitData(
                    id=f"commit{i:03d}",
                    message=f"fix: commit number {i}",
                    author="perf@test.com",
                    date=datetime.now() - timedelta(days=i),
                    stats=CommitStats(files_changed=3, lines_added=50, lines_deleted=10),
                    score=CommitScore(85 - (i % 20), 0, 0, 0, 0, 0, 0),
                    file_paths=["src/main.py", "tests/test.py"]
                )
                large_commit_list.append(commit)
            
            # 대량 커밋 저장소 생성
            large_repo = RepositoryResult(
                repo_name="large-repo",
                repo_path="/tmp/large-repo",
                commits=large_commit_list,
                metadata=sample_repo_metadata
            )
            
            meaningful_commits = MeaningfulCommitsData(repositories=[large_repo])
            json_file = temp_path / "large_commits.json"
            meaningful_commits.save_to_json(str(json_file))
            
            # 성능 측정
            start_time = time.time()
            
            tool = ReviewExecutorTool()
            
            with patch.object(tool, '_execute_repo_reviews', return_value=(100, 0)):
                result = tool.execute(
                    meaningful_commits_path=str(json_file),
                    output_dir=str(temp_path / "large_output"),
                    model="perf-test-model"
                )
            
            execution_time = time.time() - start_time
            
            # 성능 기준 (5초 이내)
            assert execution_time < 5.0, f"Execution took {execution_time:.2f} seconds"
            assert result.success is True
            assert result.data.total_commits_reviewed == 100
    
    def test_memory_usage_monitoring(self, sample_commit_data, sample_repo_metadata):
        """메모리 사용량 모니터링 테스트"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 여러 저장소 생성
            repos = []
            for i in range(10):
                repo = RepositoryResult(
                    repo_name=f"memory-test-repo-{i}",
                    repo_path=f"/tmp/memory-test-{i}",
                    commits=[sample_commit_data] * 5,  # 각 저장소에 5개 커밋
                    metadata=sample_repo_metadata
                )
                repos.append(repo)
            
            meaningful_commits = MeaningfulCommitsData(repositories=repos)
            json_file = temp_path / "memory_test.json"
            meaningful_commits.save_to_json(str(json_file))
            
            # 실행
            tool = ReviewExecutorTool()
            
            with patch.object(tool, '_execute_repo_reviews', return_value=(5, 0)):
                result = tool.execute(
                    meaningful_commits_path=str(json_file),
                    output_dir=str(temp_path / "memory_output"),
                    model="memory-test-model"
                )
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # 메모리 증가량이 50MB 이하인지 확인
            assert memory_increase < 50, f"Memory increased by {memory_increase:.2f} MB"
            assert result.success is True


class TestEdgeCases:
    """에지 케이스 테스트"""
    
    def test_empty_commit_message(self, sample_repo_metadata):
        """빈 커밋 메시지 처리 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 빈 메시지 커밋 생성
            empty_message_commit = CommitData(
                id="empty123",
                message="",  # 빈 메시지
                author="test@example.com",
                date=datetime.now(),
                stats=CommitStats(files_changed=1, lines_added=10, lines_deleted=0),
                score=CommitScore(75, 0, 0, 0, 0, 0, 0),
                file_paths=["test.py"]
            )
            
            repo_result = RepositoryResult(
                repo_name="empty-message-repo",
                repo_path="/tmp/empty-message",
                commits=[empty_message_commit],
                metadata=sample_repo_metadata
            )
            
            meaningful_commits = MeaningfulCommitsData(repositories=[repo_result])
            json_file = temp_path / "empty_message.json"
            meaningful_commits.save_to_json(str(json_file))
            
            tool = ReviewExecutorTool()
            
            with patch.object(tool, '_execute_repo_reviews', return_value=(1, 0)):
                result = tool.execute(
                    meaningful_commits_path=str(json_file),
                    output_dir=str(temp_path / "output"),
                    model="test-model"
                )
            
            # 빈 메시지여도 정상 처리되어야 함
            assert result.success is True
            assert result.data.total_commits_reviewed == 1
    
    def test_very_long_file_paths(self, sample_repo_metadata):
        """매우 긴 파일 경로 처리 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 매우 긴 파일 경로 생성
            long_path = "src/" + "/".join([f"very_long_directory_name_{i}" for i in range(10)]) + "/file.py"
            
            long_path_commit = CommitData(
                id="longpath123",
                message="fix: update file with very long path",
                author="test@example.com",
                date=datetime.now(),
                stats=CommitStats(files_changed=1, lines_added=20, lines_deleted=5),
                score=CommitScore(80, 0, 0, 0, 0, 0, 0),
                file_paths=[long_path]
            )
            
            repo_result = RepositoryResult(
                repo_name="long-path-repo",
                repo_path="/tmp/long-path",
                commits=[long_path_commit],
                metadata=sample_repo_metadata
            )
            
            meaningful_commits = MeaningfulCommitsData(repositories=[repo_result])
            json_file = temp_path / "long_path.json"
            meaningful_commits.save_to_json(str(json_file))
            
            tool = ReviewExecutorTool()
            
            with patch.object(tool, '_execute_repo_reviews', return_value=(1, 0)):
                result = tool.execute(
                    meaningful_commits_path=str(json_file),
                    output_dir=str(temp_path / "output"),
                    model="test-model"
                )
            
            # 긴 경로여도 정상 처리되어야 함
            assert result.success is True
            assert result.data.total_commits_reviewed == 1
    
    def test_unicode_commit_data(self, sample_repo_metadata):
        """유니코드 커밋 데이터 처리 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 유니코드 커밋 데이터 생성
            unicode_commit = CommitData(
                id="unicode123",
                message="fix: 한글 커밋 메시지 テスト пример",
                author="개발자@테스트.com",
                date=datetime.now(),
                stats=CommitStats(files_changed=2, lines_added=15, lines_deleted=3),
                score=CommitScore(85, 0, 0, 0, 0, 0, 0),
                file_paths=["src/한글파일.py", "tests/テスト.py"]
            )
            
            repo_result = RepositoryResult(
                repo_name="unicode-repo",
                repo_path="/tmp/unicode",
                commits=[unicode_commit],
                metadata=sample_repo_metadata
            )
            
            meaningful_commits = MeaningfulCommitsData(repositories=[repo_result])
            json_file = temp_path / "unicode.json"
            meaningful_commits.save_to_json(str(json_file))
            
            tool = ReviewExecutorTool()
            
            with patch.object(tool, '_execute_repo_reviews', return_value=(1, 0)):
                result = tool.execute(
                    meaningful_commits_path=str(json_file),
                    output_dir=str(temp_path / "output"),
                    model="test-model"
                )
            
            # 유니코드 데이터도 정상 처리되어야 함
            assert result.success is True
            assert result.data.total_commits_reviewed == 1


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v", "--tb=short"])
```

## 8. 테스트 실행 가이드

### 8.1 테스트 실행 명령어

```bash
# 전체 테스트 실행
pytest docs/testing/review-execution-test-cases.md -v

# 특정 테스트 클래스 실행
pytest docs/testing/review-execution-test-cases.md::TestReviewExecutorTool -v

# 커버리지 포함 실행
pytest docs/testing/review-execution-test-cases.md --cov=selvage_eval.tools.review_executor_tool --cov-report=html

# 통합 테스트만 실행
pytest docs/testing/review-execution-test-cases.md -k "integration" -v

# 성능 테스트만 실행
pytest docs/testing/review-execution-test-cases.md -k "performance" -v

# 비동기 테스트 포함 실행
pytest docs/testing/review-execution-test-cases.md --asyncio-mode=auto -v
```

### 8.2 테스트 커버리지 목표

- **ReviewExecutorTool 클래스**: 95% 이상 커버리지
- **ReviewExecutionSummary 클래스**: 100% 커버리지
- **에러 처리 로직**: 90% 이상 커버리지
- **통합 테스트**: 전체 워크플로우 커버리지

### 8.3 테스트 의존성

```bash
# 테스트 실행을 위한 추가 패키지 설치
pip install pytest-asyncio pytest-mock psutil
```

### 8.4 CI/CD 통합

```yaml
# .github/workflows/test-review-execution.yml 예제
name: Test Review Execution

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-mock pytest-asyncio psutil
    
    - name: Run review execution tests
      run: |
        pytest docs/testing/review-execution-test-cases.md \
          --cov=selvage_eval.tools.review_executor_tool \
          --cov=selvage_eval.review_execution \
          --cov-report=xml \
          --cov-fail-under=90 \
          --asyncio-mode=auto
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

이 테스트 케이스 문서는 리뷰 실행 기능의 모든 측면을 검증하며, 임시 디렉토리를 활용한 안전한 테스트 환경과 다중 모델 지원을 포함합니다.