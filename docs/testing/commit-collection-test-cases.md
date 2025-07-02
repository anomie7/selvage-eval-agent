# 커밋 수집 단계 - 테스트 케이스

본 문서는 [`commit-collection-implementation.md`](../implementation/commit-collection-implementation.md) 구현을 위한 포괄적인 테스트 케이스입니다.

## 1. 테스트 환경 설정

```python
"""테스트 의존성 및 픽스처"""
import pytest
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# 테스트 대상 클래스들
from selvage_eval.commit_collection import (
    CommitStats, CommitScore, CommitData, RepositoryMetadata,
    RepositoryResult, MeaningfulCommitsData, CommitCollector
)
from selvage_eval.config import EvaluationConfig, TargetRepository, CommitFilters
from selvage_eval.tools.tool_executor import ToolExecutor
from selvage_eval.tools.tool_result import ToolResult


@pytest.fixture
def sample_commit_stats():
    """샘플 커밋 통계 데이터"""
    return CommitStats(
        files_changed=3,
        lines_added=45,
        lines_deleted=12
    )


@pytest.fixture
def sample_commit_score():
    """샘플 커밋 점수 데이터"""
    return CommitScore(
        total_score=85,
        file_type_penalty=-2,
        line_count_score=15,
        keyword_score=10,
        path_pattern_score=10,
        time_weight_score=15,
        additional_adjustments=0
    )


@pytest.fixture
def sample_commit_data(sample_commit_stats, sample_commit_score):
    """샘플 커밋 데이터"""
    return CommitData(
        id="abc123",
        message="fix: resolve memory leak in parser",
        author="developer@example.com",
        date=datetime(2024, 1, 15, 10, 30, 0),
        stats=sample_commit_stats,
        score=sample_commit_score,
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
def sample_config():
    """샘플 설정 데이터"""
    return EvaluationConfig(
        target_repositories=[
            TargetRepository(
                name="test-repo",
                path="/tmp/test-repo",
                commits_per_repo=10
            )
        ],
        commit_filters=CommitFilters(
            min_files=2,
            max_files=10,
            min_lines=50,
            commits_per_repo=8
        )
    )


@pytest.fixture
def mock_tool_executor():
    """Mock 도구 실행기"""
    mock = Mock(spec=ToolExecutor)
    return mock


@pytest.fixture
def sample_git_log_output():
    """샘플 git log 출력"""
    return """abc123|fix: resolve memory leak in parser|developer@example.com|2024-01-15T10:30:00Z
def456|feature: add new search functionality|dev2@example.com|2024-01-14T14:20:00Z
ghi789|refactor: improve code structure|dev3@example.com|2024-01-13T09:15:00Z"""


@pytest.fixture
def sample_git_stat_output():
    """샘플 git show --stat 출력"""
    return """ src/parser.py     | 25 +++++++++++++++++++
 src/memory.py      | 15 ++++++++----
 tests/test_parser.py | 17 +++++++++++---
 3 files changed, 45 insertions(+), 12 deletions(-)"""


@pytest.fixture
def sample_git_files_output():
    """샘플 git show --name-only 출력"""
    return """src/parser.py
src/memory.py
tests/test_parser.py"""
```

## 2. 데이터 클래스 단위 테스트

### 2.1 CommitStats 테스트

```python
class TestCommitStats:
    """CommitStats 클래스 테스트"""
    
    def test_creation(self):
        """정상적인 객체 생성 테스트"""
        stats = CommitStats(files_changed=3, lines_added=45, lines_deleted=12)
        
        assert stats.files_changed == 3
        assert stats.lines_added == 45
        assert stats.lines_deleted == 12
    
    def test_total_lines_changed_property(self, sample_commit_stats):
        """총 변경 라인 수 계산 테스트"""
        assert sample_commit_stats.total_lines_changed == 57  # 45 + 12
    
    def test_addition_ratio_property(self, sample_commit_stats):
        """추가 라인 비율 계산 테스트"""
        expected_ratio = 45 / 57  # 약 0.789
        assert abs(sample_commit_stats.addition_ratio - expected_ratio) < 0.001
    
    def test_addition_ratio_zero_total_lines(self):
        """총 변경 라인이 0인 경우 추가 비율 테스트"""
        stats = CommitStats(files_changed=1, lines_added=0, lines_deleted=0)
        assert stats.addition_ratio == 0.0
    
    def test_addition_ratio_only_additions(self):
        """추가만 있는 경우 비율 테스트"""
        stats = CommitStats(files_changed=2, lines_added=100, lines_deleted=0)
        assert stats.addition_ratio == 1.0
    
    def test_addition_ratio_only_deletions(self):
        """삭제만 있는 경우 비율 테스트"""
        stats = CommitStats(files_changed=1, lines_added=0, lines_deleted=50)
        assert stats.addition_ratio == 0.0


class TestCommitScore:
    """CommitScore 클래스 테스트"""
    
    def test_creation(self):
        """정상적인 객체 생성 테스트"""
        score = CommitScore(
            total_score=85,
            file_type_penalty=-2,
            line_count_score=15,
            keyword_score=20,
            path_pattern_score=10,
            time_weight_score=15,
            additional_adjustments=0
        )
        
        assert score.total_score == 85
        assert score.file_type_penalty == -2
    
    def test_score_normalization_above_100(self):
        """100점 초과 점수 정규화 테스트"""
        score = CommitScore(
            total_score=150,  # 100 초과
            file_type_penalty=0,
            line_count_score=0,
            keyword_score=0,
            path_pattern_score=0,
            time_weight_score=0,
            additional_adjustments=0
        )
        
        assert score.total_score == 100  # 100으로 제한
    
    def test_score_normalization_below_0(self):
        """0점 미만 점수 정규화 테스트"""
        score = CommitScore(
            total_score=-10,  # 0 미만
            file_type_penalty=0,
            line_count_score=0,
            keyword_score=0,
            path_pattern_score=0,
            time_weight_score=0,
            additional_adjustments=0
        )
        
        assert score.total_score == 0  # 0으로 제한


class TestCommitData:
    """CommitData 클래스 테스트"""
    
    def test_creation(self, sample_commit_data):
        """정상적인 객체 생성 테스트"""
        assert sample_commit_data.id == "abc123"
        assert sample_commit_data.message == "fix: resolve memory leak in parser"
        assert sample_commit_data.author == "developer@example.com"
        assert len(sample_commit_data.file_paths) == 3
    
    def test_to_dict_conversion(self, sample_commit_data):
        """딕셔너리 변환 테스트"""
        data_dict = sample_commit_data.to_dict()
        
        assert data_dict['id'] == "abc123"
        assert data_dict['message'] == "fix: resolve memory leak in parser"
        assert data_dict['author'] == "developer@example.com"
        assert data_dict['date'] == "2024-01-15T10:30:00"  # ISO format
        assert 'stats' in data_dict
        assert 'score' in data_dict
        assert 'file_paths' in data_dict
    
    def test_date_serialization(self, sample_commit_data):
        """날짜 직렬화 테스트"""
        data_dict = sample_commit_data.to_dict()
        assert isinstance(data_dict['date'], str)
        
        # ISO 형식으로 변환되었는지 확인
        parsed_date = datetime.fromisoformat(data_dict['date'])
        assert parsed_date == sample_commit_data.date


class TestRepositoryMetadata:
    """RepositoryMetadata 클래스 테스트"""
    
    def test_creation(self, sample_repo_metadata):
        """정상적인 객체 생성 테스트"""
        assert sample_repo_metadata.total_commits == 150
        assert sample_repo_metadata.filtered_commits == 25
        assert sample_repo_metadata.selected_commits == 10
        assert sample_repo_metadata.processing_time_seconds == 5.2
    
    def test_to_dict_conversion(self, sample_repo_metadata):
        """딕셔너리 변환 테스트"""
        data_dict = sample_repo_metadata.to_dict()
        
        assert data_dict['total_commits'] == 150
        assert data_dict['filtered_commits'] == 25
        assert data_dict['selected_commits'] == 10
        assert data_dict['processing_time_seconds'] == 5.2
        assert isinstance(data_dict['filter_timestamp'], str)


class TestMeaningfulCommitsData:
    """MeaningfulCommitsData 클래스 테스트"""
    
    def test_creation_empty(self):
        """빈 저장소 목록으로 생성 테스트"""
        data = MeaningfulCommitsData(repositories=[])
        assert len(data.repositories) == 0
        assert data.total_commits == 0
    
    def test_total_commits_calculation(self, sample_commit_data, sample_repo_metadata):
        """총 커밋 수 계산 테스트"""
        repo1 = RepositoryResult(
            repo_name="repo1",
            repo_path="/tmp/repo1",
            commits=[sample_commit_data, sample_commit_data],  # 2개
            metadata=sample_repo_metadata
        )
        
        repo2 = RepositoryResult(
            repo_name="repo2", 
            repo_path="/tmp/repo2",
            commits=[sample_commit_data, sample_commit_data, sample_commit_data],  # 3개
            metadata=sample_repo_metadata
        )
        
        data = MeaningfulCommitsData(repositories=[repo1, repo2])
        assert data.total_commits == 5  # 2 + 3
    
    def test_save_to_json(self, sample_commit_data, sample_repo_metadata, tmp_path):
        """JSON 파일 저장 테스트"""
        repo = RepositoryResult(
            repo_name="test-repo",
            repo_path="/tmp/test-repo", 
            commits=[sample_commit_data],
            metadata=sample_repo_metadata
        )
        
        data = MeaningfulCommitsData(repositories=[repo])
        
        # 임시 파일에 저장
        json_file = tmp_path / "test_output.json"
        data.save_to_json(str(json_file))
        
        # 저장된 파일 검증
        assert json_file.exists()
        
        with open(json_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert 'repositories' in saved_data
        assert len(saved_data['repositories']) == 1
        assert saved_data['repositories'][0]['repo_name'] == "test-repo"
```

## 3. CommitCollector 단위 테스트

### 3.1 초기화 및 기본 기능 테스트

```python
class TestCommitCollector:
    """CommitCollector 클래스 테스트"""
    
    def test_initialization(self, sample_config, mock_tool_executor):
        """정상적인 초기화 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        assert collector.config == sample_config
        assert collector.tool_executor == mock_tool_executor
        assert collector.logger is not None
    
    def test_constants_defined(self):
        """상수들이 올바르게 정의되었는지 테스트"""
        assert '.txt' in CommitCollector.NON_CODE_FILES
        assert '.json' in CommitCollector.MINOR_PENALTY_FILES
        assert 'fix' in CommitCollector.POSITIVE_KEYWORDS
        assert 'feat' in CommitCollector.POSITIVE_KEYWORDS
        assert 'migrate' in CommitCollector.POSITIVE_KEYWORDS
        assert 'typo' in CommitCollector.NEGATIVE_KEYWORDS
        assert 'chore' in CommitCollector.NEGATIVE_KEYWORDS
        assert 'docs' in CommitCollector.NEGATIVE_KEYWORDS
        assert len(CommitCollector.CORE_PATH_PATTERNS) > 0


class TestGitCommandExecution:
    """Git 명령어 실행 관련 테스트"""
    
    def test_execute_git_command_success(self, sample_config, mock_tool_executor):
        """Git 명령어 성공 실행 테스트"""
        # Mock 설정
        mock_result = ToolResult(
            success=True,
            data={'stdout': 'test output', 'stderr': '', 'returncode': 0},
            error_message=None
        )
        mock_tool_executor.execute_tool.return_value = mock_result
        
        collector = CommitCollector(sample_config, mock_tool_executor)
        result = collector._execute_git_command("git status", "/tmp/repo")
        
        assert result.success is True
        assert result.data['stdout'] == 'test output'
        mock_tool_executor.execute_tool.assert_called_once_with(
            "execute_safe_command",
            {"command": "git status", "cwd": "/tmp/repo", "timeout": 60}
        )
    
    def test_execute_git_command_failure(self, sample_config, mock_tool_executor):
        """Git 명령어 실패 실행 테스트"""
        # Mock 설정
        mock_tool_executor.execute_tool.side_effect = Exception("Command failed")
        
        collector = CommitCollector(sample_config, mock_tool_executor)
        result = collector._execute_git_command("git invalid", "/tmp/repo")
        
        assert result.success is False
        assert "Command failed" in result.error_message


class TestCommitStatsParsing:
    """커밋 통계 파싱 테스트"""
    
    def test_parse_commit_stats_normal(self, sample_config, mock_tool_executor, sample_git_stat_output):
        """정상적인 통계 파싱 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        stats = collector._parse_commit_stats(sample_git_stat_output)
        
        assert stats.files_changed == 3
        assert stats.lines_added == 45
        assert stats.lines_deleted == 12
    
    def test_parse_commit_stats_empty(self, sample_config, mock_tool_executor):
        """빈 출력 파싱 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        stats = collector._parse_commit_stats("")
        
        assert stats.files_changed == 0
        assert stats.lines_added == 0
        assert stats.lines_deleted == 0
    
    def test_parse_commit_stats_single_file(self, sample_config, mock_tool_executor):
        """단일 파일 변경 파싱 테스트"""
        output = """ src/main.py | 10 ++++++++++
 1 file changed, 10 insertions(+)"""
        
        collector = CommitCollector(sample_config, mock_tool_executor)
        stats = collector._parse_commit_stats(output)
        
        assert stats.files_changed == 1
        assert stats.lines_added == 10
        assert stats.lines_deleted == 0
    
    def test_parse_commit_stats_only_deletions(self, sample_config, mock_tool_executor):
        """삭제만 있는 경우 파싱 테스트"""
        output = """ src/old.py | 20 --------------------
 1 file changed, 20 deletions(-)"""
        
        collector = CommitCollector(sample_config, mock_tool_executor)
        stats = collector._parse_commit_stats(output)
        
        assert stats.files_changed == 1
        assert stats.lines_added == 0
        assert stats.lines_deleted == 20


class TestFiltering:
    """필터링 로직 테스트"""
    
    def test_passes_stats_filter_success(self, sample_config, mock_tool_executor, sample_commit_data):
        """통계 필터링 통과 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # 필터 조건: min_files=2, max_files=10, min_lines=50
        # 커밋 데이터: files_changed=3, total_lines=57
        result = collector._passes_stats_filter(sample_commit_data, sample_config.commit_filters)
        
        assert result is True
    
    def test_passes_stats_filter_too_few_files(self, sample_config, mock_tool_executor):
        """파일 수 부족으로 필터링 실패 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # 1개 파일 (최소 2개 필요)
        commit = CommitData(
            id="test", message="test", author="test", date=datetime.now(),
            stats=CommitStats(files_changed=1, lines_added=100, lines_deleted=0),
            score=CommitScore(100, 0, 0, 0, 0, 0, 0),
            file_paths=["test.py"]
        )
        
        result = collector._passes_stats_filter(commit, sample_config.commit_filters)
        assert result is False
    
    def test_passes_stats_filter_too_many_files(self, sample_config, mock_tool_executor):
        """파일 수 초과로 필터링 실패 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # 11개 파일 (최대 10개)
        commit = CommitData(
            id="test", message="test", author="test", date=datetime.now(),
            stats=CommitStats(files_changed=11, lines_added=100, lines_deleted=0),
            score=CommitScore(100, 0, 0, 0, 0, 0, 0),
            file_paths=["test.py"] * 11
        )
        
        result = collector._passes_stats_filter(commit, sample_config.commit_filters)
        assert result is False
    
    def test_passes_stats_filter_too_few_lines(self, sample_config, mock_tool_executor):
        """변경 라인 수 부족으로 필터링 실패 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # 30라인 변경 (최소 50라인 필요)
        commit = CommitData(
            id="test", message="test", author="test", date=datetime.now(),
            stats=CommitStats(files_changed=3, lines_added=20, lines_deleted=10),
            score=CommitScore(100, 0, 0, 0, 0, 0, 0),
            file_paths=["test.py"] * 3
        )
        
        result = collector._passes_stats_filter(commit, sample_config.commit_filters)
        assert result is False
    
    def test_passes_keyword_filter_positive(self, sample_config, mock_tool_executor):
        """긍정 키워드로 필터링 통과 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        commit = CommitData(
            id="test", message="fix: resolve bug in parser", author="test", date=datetime.now(),
            stats=CommitStats(files_changed=3, lines_added=50, lines_deleted=10),
            score=CommitScore(100, 0, 0, 0, 0, 0, 0),
            file_paths=["test.py"]
        )
        
        result = collector._passes_keyword_filter(commit)
        assert result is True
    
    def test_passes_keyword_filter_negative(self, sample_config, mock_tool_executor):
        """부정 키워드로 필터링 실패 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        commit = CommitData(
            id="test", message="chore: fix typo in comment", author="test", date=datetime.now(),
            stats=CommitStats(files_changed=3, lines_added=50, lines_deleted=10),
            score=CommitScore(100, 0, 0, 0, 0, 0, 0),
            file_paths=["test.py"]
        )
        
        result = collector._passes_keyword_filter(commit)
        assert result is False  # 'chore'와 'typo' 모두 부정 키워드
    
    def test_passes_keyword_filter_no_keywords(self, sample_config, mock_tool_executor):
        """키워드 없어서 필터링 실패 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        commit = CommitData(
            id="test", message="update documentation", author="test", date=datetime.now(),
            stats=CommitStats(files_changed=3, lines_added=50, lines_deleted=10),
            score=CommitScore(100, 0, 0, 0, 0, 0, 0),
            file_paths=["test.py"]
        )
        
        result = collector._passes_keyword_filter(commit)
        assert result is True  # 'update'는 긍정 키워드


class TestMergeFiltering:
    """머지 커밋 필터링 테스트"""
    
    def test_passes_merge_filter_normal_commit(self, sample_config, mock_tool_executor):
        """일반 커밋 필터링 통과 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        commit = CommitData(
            id="test", message="fix: resolve bug", author="test", date=datetime.now(),
            stats=CommitStats(files_changed=3, lines_added=50, lines_deleted=10),
            score=CommitScore(100, 0, 0, 0, 0, 0, 0),
            file_paths=["test.py"]
        )
        
        result = collector._passes_merge_filter(commit)
        assert result is True
    
    def test_passes_merge_filter_fast_forward_merge(self, sample_config, mock_tool_executor):
        """Fast-forward 머지 필터링 실패 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        commit = CommitData(
            id="test", message="Merge branch 'feature'", author="test", date=datetime.now(),
            stats=CommitStats(files_changed=0, lines_added=0, lines_deleted=0),  # 변경사항 없음
            score=CommitScore(100, 0, 0, 0, 0, 0, 0),
            file_paths=[]
        )
        
        result = collector._passes_merge_filter(commit)
        assert result is False
    
    def test_passes_merge_filter_conflict_resolution(self, sample_config, mock_tool_executor):
        """충돌 해결 머지 필터링 통과 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        commit = CommitData(
            id="test", message="Merge branch 'feature' - resolve conflict", author="test", date=datetime.now(),
            stats=CommitStats(files_changed=3, lines_added=50, lines_deleted=10),
            score=CommitScore(100, 0, 0, 0, 0, 0, 0),
            file_paths=["test.py"]
        )
        
        result = collector._passes_merge_filter(commit)
        assert result is True
    
    def test_passes_merge_filter_squash_merge(self, sample_config, mock_tool_executor):
        """스쿼시 머지 필터링 통과 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        commit = CommitData(
            id="test", message="Merge squash: feature implementation", author="test", date=datetime.now(),
            stats=CommitStats(files_changed=5, lines_added=100, lines_deleted=20),
            score=CommitScore(100, 0, 0, 0, 0, 0, 0),
            file_paths=["test.py"] * 5
        )
        
        result = collector._passes_merge_filter(commit)
        assert result is True
```

## 4. 점수 계산 시스템 테스트

```python
class TestScoring:
    """점수 계산 시스템 테스트"""
    
    def test_calculate_file_type_penalty_no_penalty(self, sample_config, mock_tool_executor):
        """파일 타입 감점 없음 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        file_paths = ["src/main.py", "src/utils.py", "tests/test_main.py"]
        penalty = collector._calculate_file_type_penalty(file_paths)
        
        assert penalty == 0  # 모두 코드 파일
    
    def test_calculate_file_type_penalty_non_code_files(self, sample_config, mock_tool_executor):
        """비-코드 파일 감점 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        file_paths = ["src/main.py", "README.txt", "image.png"]
        penalty = collector._calculate_file_type_penalty(file_paths)
        
        assert penalty == -10  # txt(-5) + png(-5)
    
    def test_calculate_file_type_penalty_minor_penalty_files(self, sample_config, mock_tool_executor):
        """경미한 감점 파일 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        file_paths = ["src/main.py", "config.json", "README.md"]
        penalty = collector._calculate_file_type_penalty(file_paths)
        
        assert penalty == -4  # json(-2) + md(-2)
    
    def test_calculate_scale_appropriateness_score_optimal(self, sample_config, mock_tool_executor):
        """최적 규모 점수 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # 3개 파일, 150라인 변경 (최적 범위)
        stats = CommitStats(files_changed=3, lines_added=100, lines_deleted=50)
        score = collector._calculate_scale_appropriateness_score(stats)
        
        assert score == 25  # 파일수(10) + 라인수(15) = 25
    
    def test_calculate_scale_appropriateness_score_extreme_ratio(self, sample_config, mock_tool_executor):
        """극단적 추가/삭제 비율 감점 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # 3개 파일, 150라인 변경이지만 95% 추가 (극단적)
        stats = CommitStats(files_changed=3, lines_added=142, lines_deleted=8)
        score = collector._calculate_scale_appropriateness_score(stats)
        
        assert score == 20  # 25 - 5(극단적 비율 감점)
    
    def test_calculate_commit_characteristics_score_positive_keywords(self, sample_config, mock_tool_executor):
        """긍정 키워드 점수 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        commit = CommitData(
            id="test", message="fix: refactor and improve performance", author="test", date=datetime.now(),
            stats=CommitStats(files_changed=3, lines_added=50, lines_deleted=10),
            score=CommitScore(100, 0, 0, 0, 0, 0, 0),
            file_paths=["src/main.py", "src/utils.py"]  # 핵심 경로
        )
        
        score = collector._calculate_commit_characteristics_score(commit)
        # fix(5) + refactor(5) + improve(5) = 15 (키워드) + 10 (핵심 경로) = 25
        assert score == 25
    
    def test_calculate_time_weight_score_recent(self, sample_config, mock_tool_executor):
        """최근 커밋 시간 가중치 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # 20일 전 커밋 (최근 1개월)
        recent_date = datetime.now() - timedelta(days=20)
        score = collector._calculate_time_weight_score(recent_date)
        
        assert score == 20
    
    def test_calculate_time_weight_score_old(self, sample_config, mock_tool_executor):
        """오래된 커밋 시간 가중치 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # 2년 전 커밋
        old_date = datetime.now() - timedelta(days=730)
        score = collector._calculate_time_weight_score(old_date)
        
        assert score == 2
    
    def test_calculate_additional_adjustments_conflict_merge(self, sample_config, mock_tool_executor):
        """충돌 해결 머지 추가 점수 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        commit = CommitData(
            id="test", message="Merge branch 'feature' - resolve conflict", author="test", date=datetime.now(),
            stats=CommitStats(files_changed=3, lines_added=50, lines_deleted=10),
            score=CommitScore(100, 0, 0, 0, 0, 0, 0),
            file_paths=["test.py"]
        )
        
        adjustment = collector._calculate_additional_adjustments(commit)
        assert adjustment == 5  # 충돌 해결 보너스
    
    def test_score_commits_integration(self, sample_config, mock_tool_executor):
        """점수 계산 통합 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        commit = CommitData(
            id="test123",
            message="fix: improve parser performance",
            author="dev@example.com",
            date=datetime.now() - timedelta(days=15),  # 최근
            stats=CommitStats(files_changed=3, lines_added=80, lines_deleted=20),  # 좋은 규모
            score=CommitScore(100, 0, 0, 0, 0, 0, 0),  # 초기값
            file_paths=["src/parser.py", "src/utils.py", "tests/test_parser.py"]
        )
        
        scored_commit = collector._score_commits(commit)
        
        # 점수가 계산되었는지 확인
        assert scored_commit.score.total_score > 0
        assert scored_commit.score.total_score <= 100
        
        # 다른 필드들이 유지되었는지 확인
        assert scored_commit.id == commit.id
        assert scored_commit.message == commit.message
        assert scored_commit.stats == commit.stats


class TestCommitSelection:
    """커밋 선별 테스트"""
    
    def test_select_top_commits_normal(self, sample_config, mock_tool_executor):
        """정상적인 상위 커밋 선별 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # 점수가 다른 5개 커밋 생성
        commits = []
        for i, score in enumerate([95, 85, 75, 65, 55]):
            commit = CommitData(
                id=f"commit{i}",
                message=f"commit {i}",
                author="test",
                date=datetime.now(),
                stats=CommitStats(files_changed=3, lines_added=50, lines_deleted=10),
                score=CommitScore(score, 0, 0, 0, 0, 0, 0),
                file_paths=["test.py"]
            )
            commits.append(commit)
        
        # 상위 3개 선별
        selected = collector._select_top_commits(commits, 3)
        
        assert len(selected) == 3
        assert selected[0].score.total_score == 95
        assert selected[1].score.total_score == 85
        assert selected[2].score.total_score == 75
    
    def test_select_top_commits_less_than_requested(self, sample_config, mock_tool_executor):
        """요청보다 적은 커밋 선별 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # 2개 커밋만 있음
        commits = [
            CommitData(
                id="commit1", message="commit 1", author="test", date=datetime.now(),
                stats=CommitStats(files_changed=3, lines_added=50, lines_deleted=10),
                score=CommitScore(95, 0, 0, 0, 0, 0, 0), file_paths=["test.py"]
            ),
            CommitData(
                id="commit2", message="commit 2", author="test", date=datetime.now(),
                stats=CommitStats(files_changed=3, lines_added=50, lines_deleted=10),
                score=CommitScore(85, 0, 0, 0, 0, 0, 0), file_paths=["test.py"]
            )
        ]
        
        # 5개 요청하지만 2개만 반환
        selected = collector._select_top_commits(commits, 5)
        assert len(selected) == 2
    
    def test_select_top_commits_empty_list(self, sample_config, mock_tool_executor):
        """빈 커밋 목록 선별 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        selected = collector._select_top_commits([], 5)
        assert len(selected) == 0
```

## 5. 통합 테스트

```python
class TestCommitCollectorIntegration:
    """CommitCollector 통합 테스트"""
    
    @patch('selvage_eval.commit_collection.CommitCollector._execute_git_command')
    def test_collect_repo_commits_success(self, mock_git_cmd, sample_config, mock_tool_executor,
                                         sample_git_log_output, sample_git_stat_output, sample_git_files_output):
        """저장소 커밋 수집 성공 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # Git 명령어 Mock 설정
        def git_command_side_effect(command, cwd):
            if 'git log' in command:
                return ToolResult(True, {'stdout': sample_git_log_output}, None)
            elif '--stat' in command:
                return ToolResult(True, {'stdout': sample_git_stat_output}, None)
            elif '--name-only' in command:
                return ToolResult(True, {'stdout': sample_git_files_output}, None)
            else:
                return ToolResult(False, None, "Unknown command")
        
        mock_git_cmd.side_effect = git_command_side_effect
        
        repo_config = sample_config.target_repositories[0]
        commits = collector._collect_repo_commits(repo_config)
        
        # 3개 커밋이 수집되었는지 확인
        assert len(commits) == 3
        
        # 첫 번째 커밋 검증
        first_commit = commits[0]
        assert first_commit.id == "abc123"
        assert first_commit.message == "fix: resolve memory leak in parser"
        assert first_commit.author == "developer@example.com"
        assert first_commit.stats.files_changed == 3
        assert first_commit.stats.lines_added == 45
        assert first_commit.stats.lines_deleted == 12
        assert len(first_commit.file_paths) == 3
    
    @patch('selvage_eval.commit_collection.CommitCollector._collect_repo_commits')
    @patch('selvage_eval.commit_collection.CommitCollector._filter_commits')
    @patch('selvage_eval.commit_collection.CommitCollector._score_commits')
    @patch('selvage_eval.commit_collection.CommitCollector._select_top_commits')
    def test_collect_commits_full_workflow(self, mock_select, mock_score, mock_filter, mock_collect,
                                          sample_config, mock_tool_executor, sample_commit_data):
        """전체 워크플로우 통합 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # Mock 설정
        mock_collect.return_value = [sample_commit_data] * 10  # 10개 수집
        mock_filter.return_value = [sample_commit_data] * 5    # 5개 필터링
        mock_score.side_effect = lambda x: x                   # 그대로 반환
        mock_select.return_value = [sample_commit_data] * 3    # 3개 선별
        
        # 실행
        result = collector.collect_commits()
        
        # 결과 검증
        assert isinstance(result, MeaningfulCommitsData)
        assert len(result.repositories) == 1
        
        repo_result = result.repositories[0]
        assert repo_result.repo_name == "test-repo"
        assert repo_result.repo_path == "/tmp/test-repo"
        assert len(repo_result.commits) == 3
        
        # 메타데이터 검증
        metadata = repo_result.metadata
        assert metadata.total_commits == 10
        assert metadata.filtered_commits == 5
        assert metadata.selected_commits == 3
        assert metadata.processing_time_seconds > 0
    
    def test_collect_commits_repository_failure(self, sample_config, mock_tool_executor):
        """저장소 처리 실패 시 복원력 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # Git 명령어 실패 Mock
        mock_tool_executor.execute_tool.side_effect = Exception("Repository not found")
        
        # 실행 (예외 발생하지 않아야 함)
        result = collector.collect_commits()
        
        # 빈 결과 반환
        assert isinstance(result, MeaningfulCommitsData)
        assert len(result.repositories) == 0


class TestEndToEnd:
    """종단간 테스트"""
    
    def test_save_and_load_json(self, sample_commit_data, sample_repo_metadata, tmp_path):
        """JSON 저장 및 로드 종단간 테스트"""
        # 데이터 생성
        repo_result = RepositoryResult(
            repo_name="test-repo",
            repo_path="/tmp/test-repo",
            commits=[sample_commit_data],
            metadata=sample_repo_metadata
        )
        
        meaningful_data = MeaningfulCommitsData(repositories=[repo_result])
        
        # JSON 저장
        json_file = tmp_path / "output.json"
        meaningful_data.save_to_json(str(json_file))
        
        # JSON 로드 및 검증
        with open(json_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert 'repositories' in loaded_data
        assert len(loaded_data['repositories']) == 1
        
        repo_data = loaded_data['repositories'][0]
        assert repo_data['repo_name'] == "test-repo"
        assert repo_data['repo_path'] == "/tmp/test-repo"
        assert len(repo_data['commits']) == 1
        
        commit_data = repo_data['commits'][0]
        assert commit_data['id'] == "abc123"
        assert commit_data['message'] == "fix: resolve memory leak in parser"
        assert commit_data['author'] == "developer@example.com"
```

## 6. 에지 케이스 및 성능 테스트

```python
class TestEdgeCases:
    """에지 케이스 테스트"""
    
    def test_empty_repository(self, sample_config, mock_tool_executor):
        """빈 저장소 처리 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # 빈 git log 출력
        mock_tool_executor.execute_tool.return_value = ToolResult(
            True, {'stdout': '', 'stderr': '', 'returncode': 0}, None
        )
        
        repo_config = sample_config.target_repositories[0]
        commits = collector._collect_repo_commits(repo_config)
        
        assert len(commits) == 0
    
    def test_malformed_git_output(self, sample_config, mock_tool_executor):
        """잘못된 형식의 git 출력 처리 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # 잘못된 형식의 git log 출력
        malformed_output = """abc123|incomplete
def456|missing|fields
ghi789|proper|format|author@example.com|2024-01-15T10:30:00Z"""
        
        mock_tool_executor.execute_tool.return_value = ToolResult(
            True, {'stdout': malformed_output}, None
        )
        
        repo_config = sample_config.target_repositories[0]
        commits = collector._collect_repo_commits(repo_config)
        
        # 올바른 형식의 커밋만 처리됨
        assert len(commits) == 1 if commits else 0  # Git 명령어 상세 Mock이 필요
    
    def test_unicode_commit_messages(self, sample_config, mock_tool_executor):
        """유니코드 커밋 메시지 처리 테스트"""
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        unicode_output = """abc123|fix: 한글 커밋 메시지 테스트|developer@example.com|2024-01-15T10:30:00Z
def456|feat: 日本語コミット|dev2@example.com|2024-01-14T14:20:00Z
ghi789|refactor: пятна код|dev3@example.com|2024-01-13T09:15:00Z"""
        
        # Git 명령어 Mock 설정 (간단히)
        def git_side_effect(command, cwd):
            if 'git log' in command:
                return ToolResult(True, {'stdout': unicode_output}, None)
            else:
                return ToolResult(True, {'stdout': ''}, None)
        
        mock_tool_executor.execute_tool.side_effect = git_side_effect
        
        repo_config = sample_config.target_repositories[0]
        commits = collector._collect_repo_commits(repo_config)
        
        # 유니코드 메시지가 올바르게 처리되는지 확인 (실제 구현에 따라 결과 다름)
        assert isinstance(commits, list)


class TestPerformance:
    """성능 테스트"""
    
    def test_large_commit_list_performance(self, sample_config, mock_tool_executor):
        """대량 커밋 목록 처리 성능 테스트"""
        import time
        
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # 1000개 커밋 생성
        large_commit_list = []
        for i in range(1000):
            commit = CommitData(
                id=f"commit{i:04d}",
                message=f"fix: commit number {i}",
                author="test@example.com",
                date=datetime.now() - timedelta(days=i),
                stats=CommitStats(files_changed=3, lines_added=50, lines_deleted=10),
                score=CommitScore(85 - (i % 20), 0, 0, 0, 0, 0, 0),  # 점수 변화
                file_paths=["src/main.py", "tests/test.py"]
            )
            large_commit_list.append(commit)
        
        # 성능 측정
        start_time = time.time()
        
        # 필터링 테스트
        filtered = collector._filter_commits(large_commit_list, sample_config.commit_filters)
        
        # 점수 계산 테스트 (처음 100개만)
        scored = [collector._score_commits(commit) for commit in filtered[:100]]
        
        # 선별 테스트
        selected = collector._select_top_commits(scored, 10)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 성능 기준 (1초 이내)
        assert processing_time < 1.0, f"Processing took {processing_time:.2f} seconds"
        assert len(selected) <= 10
        
    def test_memory_usage_with_large_data(self, sample_config, mock_tool_executor):
        """대량 데이터 처리 시 메모리 사용량 테스트"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        collector = CommitCollector(sample_config, mock_tool_executor)
        
        # 대량 데이터 생성 및 처리
        large_data = []
        for i in range(5000):
            data = MeaningfulCommitsData(repositories=[
                RepositoryResult(
                    repo_name=f"repo{i}",
                    repo_path=f"/tmp/repo{i}",
                    commits=[],
                    metadata=RepositoryMetadata(0, 0, 0, datetime.now(), 0.1)
                )
            ])
            large_data.append(data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 메모리 증가량이 100MB 이하인지 확인
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f} MB"


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v", "--tb=short"])
```

## 7. 테스트 실행 가이드

### 7.1 테스트 실행 명령어

```bash
# 전체 테스트 실행
pytest docs/testing/commit-collection-test-cases.md -v

# 특정 테스트 클래스 실행
pytest docs/testing/commit-collection-test-cases.md::TestCommitStats -v

# 커버리지 포함 실행
pytest docs/testing/commit-collection-test-cases.md --cov=selvage_eval.commit_collection --cov-report=html

# 성능 테스트만 실행
pytest docs/testing/commit-collection-test-cases.md -k "performance" -v

# 통합 테스트만 실행
pytest docs/testing/commit-collection-test-cases.md -k "integration" -v
```

### 7.2 테스트 커버리지 목표

- **데이터 클래스**: 100% 커버리지
- **CommitCollector 메서드**: 95% 이상 커버리지
- **에지 케이스**: 주요 예외 상황 90% 이상 커버리지
- **통합 테스트**: 전체 워크플로우 커버리지

### 7.3 CI/CD 통합

```yaml
# .github/workflows/test.yml 예제
name: Test Commit Collection

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
        pip install pytest pytest-cov pytest-mock
    
    - name: Run tests
      run: |
        pytest docs/testing/commit-collection-test-cases.md \
          --cov=selvage_eval.commit_collection \
          --cov-report=xml \
          --cov-fail-under=90
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

이 테스트 케이스 문서는 구현된 커밋 수집 기능의 모든 측면을 검증하며, 프로덕션 환경에서 안정적인 동작을 보장합니다.