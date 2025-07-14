"""커밋 다양성 기능 테스트 모듈"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

# commit_collection.py 파일에서 직접 import 
import importlib.util
spec = importlib.util.spec_from_file_location("commit_collection_module", "/Users/demin_coder/Dev/selvage-eval-agent/src/selvage_eval/commit_collection.py")
commit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(commit_module)

CommitData = commit_module.CommitData
CommitStats = commit_module.CommitStats
CommitScore = commit_module.CommitScore
from selvage_eval.commit_collection.commit_size_category import CommitSizeCategory
from selvage_eval.commit_collection.diversity_selector import DiversityBasedSelector, CategoryAllocation
from selvage_eval.config.settings import CommitDiversityConfig, CommitCategoryConfig


class TestCommitSizeCategory:
    """CommitSizeCategory 테스트"""
    
    def test_categorize_by_lines_basic(self):
        """기본 라인 수 분류 테스트"""
        assert CommitSizeCategory.categorize_by_lines(3) == CommitSizeCategory.EXTRA_SMALL
        assert CommitSizeCategory.categorize_by_lines(25) == CommitSizeCategory.SMALL
        assert CommitSizeCategory.categorize_by_lines(75) == CommitSizeCategory.MEDIUM
        assert CommitSizeCategory.categorize_by_lines(150) == CommitSizeCategory.LARGE
        assert CommitSizeCategory.categorize_by_lines(200) == CommitSizeCategory.EXTRA_LARGE
    
    def test_categorize_by_lines_with_file_correction(self):
        """파일 수 보정 테스트"""
        # 파일 수가 많으면 LARGE로 상향
        assert CommitSizeCategory.categorize_by_lines(25, files_changed=15) == CommitSizeCategory.LARGE
        
        # 단일 파일 큰 변경은 LARGE로 제한
        assert CommitSizeCategory.categorize_by_lines(300, files_changed=1) == CommitSizeCategory.LARGE
    
    def test_get_category_info(self):
        """카테고리 정보 반환 테스트"""
        info = CommitSizeCategory.get_category_info()
        assert "extra_small" in info
        assert info["small"]["research_ratio"] == 0.553
        assert len(info["medium"]["characteristics"]) > 0
    
    def test_get_research_distribution(self):
        """연구 데이터 분포 테스트"""
        distribution = CommitSizeCategory.get_research_distribution()
        total_ratio = sum(distribution.values())
        assert abs(total_ratio - 1.0) < 0.01  # 합계가 100%에 근사


# class TestDiversityBasedSelector:
#     """DiversityBasedSelector 테스트"""
    
    @pytest.fixture
    def sample_config(self):
        """테스트용 다양성 설정"""
        return CommitDiversityConfig(
            enabled=True,
            categories={
                "extra_small": CommitCategoryConfig(
                    target_ratio=0.2, min_count=1, max_count=3, score_boost=15
                ),
                "small": CommitCategoryConfig(
                    target_ratio=0.5, min_count=2, max_count=8, score_boost=10
                ),
                "medium": CommitCategoryConfig(
                    target_ratio=0.2, min_count=1, max_count=3, score_boost=5
                ),
                "large": CommitCategoryConfig(
                    target_ratio=0.05, min_count=0, max_count=1, score_boost=0
                ),
                "extra_large": CommitCategoryConfig(
                    target_ratio=0.05, min_count=0, max_count=1, score_boost=0
                )
            }
        )
    
    @pytest.fixture
    def sample_commits(self):
        """테스트용 커밋 데이터"""
        commits = []
        
        # 다양한 크기의 커밋 생성
        test_cases = [
            (3, 1, 80),    # EXTRA_SMALL
            (3, 1, 75),    # EXTRA_SMALL
            (20, 2, 85),   # SMALL
            (25, 3, 90),   # SMALL
            (30, 2, 70),   # SMALL
            (80, 4, 88),   # MEDIUM
            (90, 5, 82),   # MEDIUM
            (150, 6, 95),  # LARGE
            (200, 8, 92),  # EXTRA_LARGE
        ]
        
        for i, (lines, files, score) in enumerate(test_cases):
            commit = CommitData(
                id=f"commit_{i}",
                message=f"Test commit {i}",
                author="test@example.com",
                date=datetime.now(),
                stats=CommitStats(
                    files_changed=files,
                    lines_added=lines // 2,
                    lines_deleted=lines // 2
                ),
                score=CommitScore(
                    total_score=score,
                    file_type_penalty=0,
                    scale_appropriateness_score=10,
                    commit_characteristics_score=15,
                    time_weight_score=20,
                    additional_adjustments=0
                ),
                file_paths=[f"file_{j}.py" for j in range(files)]
            )
            commits.append(commit)
        
        return commits
    
    def test_select_diverse_commits_basic(self, sample_config, sample_commits):
        """기본 다양성 선택 테스트"""
        selector = DiversityBasedSelector(sample_config)
        selected = selector.select_diverse_commits(sample_commits, 5)
        
        assert len(selected) == 5
        
        # 카테고리별 분포 확인
        categorized = selector._categorize_commits(selected)
        assert len(categorized[CommitSizeCategory.EXTRA_SMALL]) >= 1  # 최소 1개
        assert len(categorized[CommitSizeCategory.SMALL]) >= 2        # 최소 2개
    
    def test_categorize_commits(self, sample_config, sample_commits):
        """커밋 분류 테스트"""
        selector = DiversityBasedSelector(sample_config)
        categorized = selector._categorize_commits(sample_commits)
        
        # 카테고리별 개수 확인
        assert len(categorized[CommitSizeCategory.EXTRA_SMALL]) == 2
        assert len(categorized[CommitSizeCategory.SMALL]) == 3
        assert len(categorized[CommitSizeCategory.MEDIUM]) == 2
        assert len(categorized[CommitSizeCategory.LARGE]) == 1
        assert len(categorized[CommitSizeCategory.EXTRA_LARGE]) == 1
    
    def test_calculate_allocations(self, sample_config, sample_commits):
        """할당량 계산 테스트"""
        selector = DiversityBasedSelector(sample_config)
        categorized = selector._categorize_commits(sample_commits)
        allocations = selector._calculate_allocations(categorized, 10)
        
        # 할당량 검증
        assert allocations[CommitSizeCategory.SMALL].target_count == 5  # 50% of 10
        assert allocations[CommitSizeCategory.EXTRA_SMALL].target_count == 2  # 20% of 10
        
        # 최소/최대 제약 확인
        for allocation in allocations.values():
            assert allocation.target_count >= allocation.min_count
            assert allocation.target_count <= allocation.max_count
    
    def test_select_with_shortage(self, sample_config):
        """부족한 커밋 상황 테스트"""
        # 특정 카테고리만 있는 커밋들
        commits = [
            self._create_test_commit(0, 25, 2, 80),  # SMALL
            self._create_test_commit(1, 30, 3, 85),  # SMALL
        ]
        
        selector = DiversityBasedSelector(sample_config)
        selected = selector.select_diverse_commits(commits, 5)
        
        # 요청한 개수보다 적을 수 있음 (재배분 후에도)
        assert len(selected) <= 5
        assert len(selected) == 2  # 사용 가능한 모든 커밋
    
    def test_quality_minimum_threshold(self, sample_config):
        """품질 최소 기준 테스트"""
        # 품질이 낮은 커밋들
        low_quality_commits = [
            self._create_test_commit(0, 25, 2, 40),  # 낮은 점수
            self._create_test_commit(1, 30, 3, 45),  # 낮은 점수
        ]
        
        # 품질 기준 설정
        sample_config.quality_scoring.min_quality_scores["small"] = 60
        
        selector = DiversityBasedSelector(sample_config)
        selected = selector.select_diverse_commits(low_quality_commits, 2)
        
        # 품질 기준을 만족하지 않으면 선택되지 않을 수 있음
        assert len(selected) == 0
    
    def test_generate_selection_report(self, sample_config, sample_commits):
        """선택 보고서 생성 테스트"""
        selector = DiversityBasedSelector(sample_config)
        selected = selector.select_diverse_commits(sample_commits, 5)
        
        report = selector.generate_selection_report(sample_commits, selected)
        
        assert report.total_available == len(sample_commits)
        assert report.total_selected == len(selected)
        assert len(report.category_distributions) == 5  # 5개 카테고리
        assert len(report.quality_scores) == 5
        assert len(report.selection_ratios) == 5
    
    def _create_test_commit(self, commit_id: int, lines: int, files: int, score: int) -> CommitData:
        """테스트용 커밋 생성 헬퍼"""
        return CommitData(
            id=f"commit_{commit_id}",
            message=f"Test commit {commit_id}",
            author="test@example.com",
            date=datetime.now(),
            stats=CommitStats(
                files_changed=files,
                lines_added=lines // 2,
                lines_deleted=lines // 2
            ),
            score=CommitScore(
                total_score=score,
                file_type_penalty=0,
                scale_appropriateness_score=10,
                commit_characteristics_score=15,
                time_weight_score=20,
                additional_adjustments=0
            ),
            file_paths=[f"file_{j}.py" for j in range(files)]
        )


class TestCommitCollectorIntegration:
    """CommitCollector와 다양성 기능 통합 테스트"""
    
    @patch('selvage_eval.config.settings.load_commit_diversity_config')
    def test_diversity_selector_initialization(self, mock_load_config):
        """다양성 선택기 초기화 테스트"""
        # 동일한 방식으로 CommitCollector import
        CommitCollector = commit_module.CommitCollector
        from selvage_eval.config.settings import EvaluationConfig
        
        # Mock 설정
        mock_diversity_config = CommitDiversityConfig(enabled=True)
        mock_load_config.return_value = mock_diversity_config
        
        # 기본 설정 객체 생성
        config = Mock(spec=EvaluationConfig)
        config.commit_diversity = None
        config.target_repositories = []
        config.commit_filters = Mock()
        config.commits_per_repo = 20
        
        tool_executor = Mock()
        
        # CommitCollector 생성
        collector = CommitCollector(config, tool_executor)
        
        # 다양성 선택기가 초기화되었는지 확인
        assert collector.diversity_selector is not None
        assert collector.diversity_config.enabled is True
    
    def test_categorize_commit_integration(self):
        """커밋 분류 통합 테스트"""
        # 동일한 방식으로 CommitCollector import
        CommitCollector = commit_module.CommitCollector
        
        # Mock 설정
        config = Mock()
        config.commit_diversity = CommitDiversityConfig(enabled=True)
        config.target_repositories = []
        config.commit_filters = Mock()
        config.commits_per_repo = 20
        
        tool_executor = Mock()
        collector = CommitCollector(config, tool_executor)
        
        # 테스트 커밋
        test_commit = CommitData(
            id="test_commit",
            message="Test commit",
            author="test@example.com",
            date=datetime.now(),
            stats=CommitStats(files_changed=2, lines_added=15, lines_deleted=10),
            score=CommitScore(80, 0, 10, 15, 20, 0),
            file_paths=["file1.py", "file2.py"]
        )
        
        # 분류 테스트
        category = collector._categorize_commit(test_commit)
        assert category == CommitSizeCategory.SMALL  # 25 라인 → SMALL


@pytest.fixture
def mock_diversity_config():
    """Mock 다양성 설정"""
    return CommitDiversityConfig(
        enabled=True,
        categories={
            "small": CommitCategoryConfig(
                target_ratio=0.6, min_count=3, max_count=10, score_boost=10
            ),
            "medium": CommitCategoryConfig(
                target_ratio=0.4, min_count=2, max_count=5, score_boost=5
            )
        }
    )


# class TestEdgeCases:
    """엣지 케이스 테스트"""
    
    def test_empty_commits_list(self, mock_diversity_config):
        """빈 커밋 목록 처리"""
        selector = DiversityBasedSelector(mock_diversity_config)
        result = selector.select_diverse_commits([], 5)
        assert result == []
    
    def test_single_category_commits(self, mock_diversity_config):
        """단일 카테고리만 있는 경우"""
        commits = [
            CommitData(
                id=f"commit_{i}",
                message=f"Small commit {i}",
                author="test@example.com",
                date=datetime.now(),
                stats=CommitStats(files_changed=2, lines_added=10, lines_deleted=5),
                score=CommitScore(80 + i, 0, 10, 15, 20, 0),
                file_paths=["file.py"]
            )
            for i in range(3)
        ]
        
        selector = DiversityBasedSelector(mock_diversity_config)
        selected = selector.select_diverse_commits(commits, 5)
        
        # 모든 커밋이 같은 카테고리여도 선택됨
        assert len(selected) == 3
    
    def test_more_requested_than_available(self, mock_diversity_config):
        """요청 개수가 사용 가능한 커밋보다 많은 경우"""
        commits = [
            CommitData(
                id="commit_1",
                message="Small commit",
                author="test@example.com",
                date=datetime.now(),
                stats=CommitStats(files_changed=2, lines_added=10, lines_deleted=5),
                score=CommitScore(80, 0, 10, 15, 20, 0),
                file_paths=["file.py"]
            )
        ]
        
        selector = DiversityBasedSelector(mock_diversity_config)
        selected = selector.select_diverse_commits(commits, 10)
        
        # 사용 가능한 모든 커밋 반환
        assert len(selected) == 1