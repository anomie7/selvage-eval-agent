import pytest
import json
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# 테스트 대상 클래스들
from selvage_eval.tools.review_executor_tool import ReviewExecutorTool
from selvage_eval.review_execution_summary import ReviewExecutionSummary
from selvage_eval.commit_collection.commit_stats import CommitStats
from selvage_eval.commit_collection.commit_score import CommitScore
from selvage_eval.commit_collection.commit_data import CommitData
from selvage_eval.commit_collection.repository_metadata import RepositoryMetadata
from selvage_eval.commit_collection.repository_result import RepositoryResult
from selvage_eval.commit_collection.meaningful_commits_data import MeaningfulCommitsData
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
        score=CommitScore(85, -2, 15, 10, 15, 0),
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
                model="gemini-2.5-pro",
                output_dir=str(temp_output_dir)
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
                model="gemini-2.5-pro",
                output_dir=str(temp_output_dir)
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
            model="gemini-2.5-pro",
            output_dir=str(temp_output_dir)
        )
        
        assert result.success is False
        assert result.data is None
        assert result.error_message is not None and "meaningful_commits.json 파일을 찾을 수 없습니다" in result.error_message
    
    def test_execute_output_directory_creation(self, meaningful_commits_json_file, temp_output_dir):
        """출력 디렉토리 생성 테스트"""
        tool = ReviewExecutorTool()
        
        # 존재하지 않는 하위 디렉토리 지정
        nested_output_dir = temp_output_dir / "nested" / "output"
        
        with patch.object(tool, '_execute_repo_reviews', return_value=(1, 0)):
            result = tool.execute(
                meaningful_commits_path=meaningful_commits_json_file,
                model="gemini-2.5-pro",
                output_dir=str(nested_output_dir)
            )
        
        assert result.success is True
        assert nested_output_dir.exists()
        assert nested_output_dir.is_dir()


class TestErrorHandling:
    """에러 처리 및 엣지 케이스 테스트"""
    
    def test_execute_empty_repositories(self, temp_output_dir):
        """빈 저장소 리스트 처리 테스트"""
        tool = ReviewExecutorTool()
        
        # 빈 저장소 리스트를 가진 JSON 파일 생성
        empty_commits_file = temp_output_dir / "empty_commits.json"
        empty_data = MeaningfulCommitsData(repositories=[])
        empty_data.save_to_json(str(empty_commits_file))
        
        result = tool.execute(
            meaningful_commits_path=str(empty_commits_file),
            model="gemini-2.5-pro",
            output_dir=str(temp_output_dir)
        )
        
        assert result.success is True
        assert result.data.total_commits_reviewed == 0
        assert result.data.total_successes == 0
        assert result.data.total_failures == 0
        assert result.data.success_rate == 0.0
    
    def test_load_meaningful_commits_corrupted_json(self, temp_output_dir):
        """부분적으로 손상된 JSON 파일 처리 테스트"""
        tool = ReviewExecutorTool()
        
        # 부분적으로 손상된 JSON 파일 생성 (구조는 맞지만 데이터가 잘못됨)
        corrupted_json_file = temp_output_dir / "corrupted.json"
        with open(corrupted_json_file, 'w') as f:
            f.write('{"repositories": [{"repo_name": null, "invalid_field": true}]}')
        
        with pytest.raises(Exception):  # 구체적인 예외 타입은 구현에 따라 달라질 수 있음
            tool._load_meaningful_commits(str(corrupted_json_file))
    
    def test_validate_parameters_edge_cases(self):
        """파라미터 검증 엣지 케이스 테스트"""
        tool = ReviewExecutorTool()
        
        # None 값
        assert tool.validate_parameters({
            "meaningful_commits_path": None,
            "output_dir": "~/output",
            "model": "gemini-2.5-pro"
        }) is False
        
        # 공백만 있는 문자열
        assert tool.validate_parameters({
            "meaningful_commits_path": "   ",
            "output_dir": "~/output", 
            "model": "gemini-2.5-pro"
        }) is False
        
        # 매우 긴 경로
        very_long_path = "a" * 1000
        assert tool.validate_parameters({
            "meaningful_commits_path": very_long_path,
            "output_dir": "~/output",
            "model": "gemini-2.5-pro"
        }) is True  # 길이 자체는 유효함
    
    def test_execute_with_special_characters_in_paths(self, temp_output_dir):
        """특수 문자가 포함된 경로 처리 테스트"""
        tool = ReviewExecutorTool()
        
        # 특수 문자가 포함된 출력 디렉토리
        special_output_dir = temp_output_dir / "output with spaces & special-chars"
        
        # 빈 저장소로 테스트 (실제 파일 시스템 작업에 집중)
        empty_commits_file = temp_output_dir / "empty_commits.json"
        empty_data = MeaningfulCommitsData(repositories=[])
        empty_data.save_to_json(str(empty_commits_file))
        
        result = tool.execute(
            meaningful_commits_path=str(empty_commits_file),
            model="gemini-2.5-pro",
            output_dir=str(special_output_dir)
        )
        
        assert result.success is True
        assert special_output_dir.exists()
        assert special_output_dir.is_dir()
    
    def test_execute_concurrent_safety(self, meaningful_commits_json_file, temp_output_dir):
        """동시 실행 안전성 테스트 (기본적인 상태 독립성 확인)"""
        tool1 = ReviewExecutorTool()
        tool2 = ReviewExecutorTool()
        
        # 다른 출력 디렉토리를 사용하여 동시 실행 시뮬레이션
        output_dir_1 = temp_output_dir / "concurrent_1"
        output_dir_2 = temp_output_dir / "concurrent_2"
        
        with patch.object(tool1, '_execute_repo_reviews', return_value=(1, 0)), \
             patch.object(tool2, '_execute_repo_reviews', return_value=(0, 1)):
            
            result1 = tool1.execute(
                meaningful_commits_path=meaningful_commits_json_file,
                model="gemini-2.5-pro",
                output_dir=str(output_dir_1)
            )
            
            result2 = tool2.execute(
                meaningful_commits_path=meaningful_commits_json_file,
                model="claude-sonnet-4",
                output_dir=str(output_dir_2)
            )
        
        # 각 도구의 결과가 독립적이어야 함
        assert result1.success is True
        assert result2.success is True
        assert result1.data.total_successes == 1
        assert result2.data.total_successes == 0
        assert output_dir_1.exists() and output_dir_2.exists()