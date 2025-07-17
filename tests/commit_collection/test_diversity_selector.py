"""DiversityBasedSelector 단위 테스트"""

import pytest
from datetime import datetime

# 일반적인 import 사용
from selvage_eval.commit_collection.commit_data import CommitData
from selvage_eval.commit_collection.commit_stats import CommitStats
from selvage_eval.commit_collection.commit_score import CommitScore
from selvage_eval.commit_collection.commit_size_category import CommitSizeCategory
from selvage_eval.commit_collection.diversity_selector import DiversityBasedSelector, CategoryAllocation, SelectionReport
from selvage_eval.config.settings import CommitDiversityConfig, CommitCategoryConfig, CommitSizeThresholds, FileCorrectionConfig


class TestDiversityBasedSelectorDetailed:
    """DiversityBasedSelector 상세 테스트"""
    
    @pytest.fixture
    def full_config(self):
        """완전한 다양성 설정"""
        return CommitDiversityConfig(
            enabled=True,
            size_thresholds=CommitSizeThresholds(
                extra_small_max=5,
                small_max=46,
                medium_max=106,
                large_max=166
            ),
            file_correction=FileCorrectionConfig(
                large_file_threshold=11,
                single_file_large_lines=100
            ),
            categories={
                "extra_small": CommitCategoryConfig(
                    target_ratio=0.20, min_count=2, max_count=6, score_boost=15
                ),
                "small": CommitCategoryConfig(
                    target_ratio=0.55, min_count=8, max_count=14, score_boost=10
                ),
                "medium": CommitCategoryConfig(
                    target_ratio=0.11, min_count=1, max_count=4, score_boost=5
                ),
                "large": CommitCategoryConfig(
                    target_ratio=0.04, min_count=0, max_count=2, score_boost=0
                ),
                "extra_large": CommitCategoryConfig(
                    target_ratio=0.10, min_count=1, max_count=3, score_boost=0
                )
            }
        )
    
    def test_categorize_commit_with_thresholds(self, full_config):
        """임계값 기반 커밋 분류 테스트"""
        selector = DiversityBasedSelector(full_config)
        
        # 경계값 테스트
        test_cases = [
            (5, 1, CommitSizeCategory.EXTRA_SMALL),    # 경계값
            (6, 1, CommitSizeCategory.SMALL),          # 경계값 + 1
            (46, 2, CommitSizeCategory.SMALL),         # 경계값
            (47, 3, CommitSizeCategory.MEDIUM),        # 경계값 + 1
            (106, 4, CommitSizeCategory.MEDIUM),       # 경계값
            (107, 5, CommitSizeCategory.LARGE),        # 경계값 + 1
            (166, 6, CommitSizeCategory.LARGE),        # 경계값
            (167, 7, CommitSizeCategory.EXTRA_LARGE),  # 경계값 + 1
        ]
        
        for lines, files, expected_category in test_cases:
            commit = self._create_commit(f"test_{lines}", lines, files, 80)
            actual_category = selector._categorize_commit(commit)
            assert actual_category == expected_category, f"Lines: {lines}, Files: {files}"
    
    def test_file_correction_logic(self, full_config):
        """파일 수 보정 로직 테스트"""
        selector = DiversityBasedSelector(full_config)
        
        # 많은 파일 수 → LARGE로 상향
        commit_many_files = self._create_commit("many_files", 30, 15, 80)  # 원래 SMALL
        category = selector._categorize_commit(commit_many_files)
        assert category == CommitSizeCategory.LARGE
        
        # 단일 파일 큰 변경 → LARGE로 제한
        commit_single_large = self._create_commit("single_large", 300, 1, 80)  # 원래 EXTRA_LARGE
        category = selector._categorize_commit(commit_single_large)
        assert category == CommitSizeCategory.LARGE
        
        # 정상적인 경우
        commit_normal = self._create_commit("normal", 30, 3, 80)
        category = selector._categorize_commit(commit_normal)
        assert category == CommitSizeCategory.SMALL
    
    def test_allocation_calculation_precision(self, full_config):
        """할당량 계산 정밀도 테스트"""
        selector = DiversityBasedSelector(full_config)
        
        # 20개 커밋에 대한 할당량 계산
        categorized = {category: [] for category in CommitSizeCategory}
        for category in CommitSizeCategory:
            # 각 카테고리에 충분한 커밋 생성
            for i in range(20):
                categorized[category].append(self._create_commit(f"{category.name}_{i}", 50, 2, 80))
        
        allocations = selector._calculate_allocations(categorized, 20)
        
        # 비율 기반 할당량 확인
        assert allocations[CommitSizeCategory.EXTRA_SMALL].target_count == 4   # 20 * 0.20
        assert allocations[CommitSizeCategory.SMALL].target_count == 11        # 20 * 0.55
        assert allocations[CommitSizeCategory.MEDIUM].target_count == 2        # 20 * 0.11
        assert allocations[CommitSizeCategory.LARGE].target_count == 1         # 20 * 0.04, max(min_count=0)
        assert allocations[CommitSizeCategory.EXTRA_LARGE].target_count == 2   # 20 * 0.10
        
        # 총합이 목표치와 일치하는지 확인
        total_allocated = sum(a.target_count for a in allocations.values())
        assert total_allocated == 20
    
    def test_quality_selection_with_boost(self, full_config):
        """점수 보정을 고려한 품질 선택 테스트"""
        commits = [
            self._create_commit("small_low", 20, 2, 70),    # SMALL, 낮은 점수
            self._create_commit("small_high", 25, 2, 60),   # SMALL, 더 낮은 점수
            self._create_commit("xs_low", 3, 1, 50),        # EXTRA_SMALL, 매우 낮은 점수
        ]
        
        selector = DiversityBasedSelector(full_config)
        categorized = selector._categorize_commits(commits)
        allocations = selector._calculate_allocations(categorized, 2)
        
        # SMALL 카테고리에서 1개, EXTRA_SMALL에서 1개 선택 예상 (재배분 전)
        selected = selector._select_by_quality(categorized, allocations)
        
        # 점수 보정으로 인해 원래 점수가 낮아도 선택될 수 있음
        # 재배분 로직으로 인해 추가 선택이 일어날 수 있음
        assert len(selected) >= 2
        
        # EXTRA_SMALL 커밋이 선택되었는지 확인 (score_boost=15로 인해)
        xs_selected = any(c.id == "xs_low" for c in selected)
        assert xs_selected, "EXTRA_SMALL 커밋이 점수 보정으로 선택되어야 함"
    
    def test_redistribution_algorithm(self, full_config):
        """재배분 알고리즘 테스트"""
        # 일부 카테고리에만 커밋이 있는 상황
        commits = [
            self._create_commit("small_1", 20, 2, 90),
            self._create_commit("small_2", 25, 2, 85),
            self._create_commit("small_3", 30, 2, 80),
            self._create_commit("medium_1", 60, 3, 95),
            # EXTRA_SMALL, LARGE, EXTRA_LARGE 카테고리는 없음
        ]
        
        selector = DiversityBasedSelector(full_config)
        selected = selector.select_diverse_commits(commits, 6)
        
        # 재배분으로 인해 사용 가능한 모든 커밋이 선택되어야 함
        assert len(selected) == 4
        
        # 점수 순으로 정렬되어 있는지 확인
        scores = [c.score.total_score for c in selected]
        assert scores == sorted(scores, reverse=True)
    
    def test_max_count_constraints(self, full_config):
        """최대 개수 제약 테스트"""
        # SMALL 카테고리에 많은 커밋 생성
        commits = []
        for i in range(20):
            commits.append(self._create_commit(f"small_{i}", 20, 2, 80 + i))
        
        selector = DiversityBasedSelector(full_config)
        selected = selector.select_diverse_commits(commits, 20)
        
        # SMALL 카테고리 최대 개수(14개) 제한 확인
        small_selected = [c for c in selected if c.id.startswith("small_")]
        assert len(small_selected) <= 14
    
    def test_selection_report_accuracy(self, full_config):
        """선택 보고서 정확성 테스트"""
        commits = [
            self._create_commit("xs_1", 3, 1, 70),
            self._create_commit("small_1", 20, 2, 80),
            self._create_commit("small_2", 25, 2, 85),
            self._create_commit("medium_1", 60, 3, 90),
            self._create_commit("large_1", 120, 5, 95),
        ]
        
        selector = DiversityBasedSelector(full_config)
        selected = selector.select_diverse_commits(commits, 3)
        
        report = selector.generate_selection_report(commits, selected)
        
        # 보고서 정확성 검증
        assert report.total_available == 5
        assert report.total_selected == 3
        
        # 카테고리별 분포 확인
        assert "EXTRA_SMALL" in report.category_distributions
        assert "SMALL" in report.category_distributions
        
        # 선택 비율 확인
        for category_name, ratio in report.selection_ratios.items():
            assert 0.0 <= ratio <= 1.0
    
    def test_logging_integration(self, full_config, caplog):
        """로깅 통합 테스트"""
        full_config.debug.log_category_distribution = True
        full_config.debug.log_selection_details = True
        
        commits = [self._create_commit("test", 20, 2, 80)]
        
        selector = DiversityBasedSelector(full_config)
        selector.select_diverse_commits(commits, 1)
        
        # 로그 메시지 확인
        assert "커밋 카테고리별 분포" in caplog.text
        assert "다양성 기반 커밋 선택 결과" in caplog.text
    
    def _create_commit(self, commit_id: str, lines: int, files: int, score: int) -> CommitData:
        """테스트용 커밋 생성 헬퍼"""
        return CommitData(
            id=commit_id,
            message=f"Test commit {commit_id}",
            author="test@example.com",
            date=datetime.now(),
            stats=CommitStats(
                files_changed=files,
                lines_added=lines // 2,
                lines_deleted=lines - (lines // 2)
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


class TestCategoryAllocation:
    """CategoryAllocation 데이터클래스 테스트"""
    
    def test_category_allocation_creation(self):
        """CategoryAllocation 생성 테스트"""
        allocation = CategoryAllocation(
            category=CommitSizeCategory.SMALL,
            target_count=5,
            min_count=2,
            max_count=10,
            available_commits=8
        )
        
        assert allocation.category == CommitSizeCategory.SMALL
        assert allocation.target_count == 5
        assert allocation.selected_count == 0  # 기본값


class TestSelectionReport:
    """SelectionReport 데이터클래스 테스트"""
    
    def test_selection_report_creation(self):
        """SelectionReport 생성 테스트"""
        report = SelectionReport(
            total_available=10,
            total_selected=5,
            category_distributions={"SMALL": {"original": 6, "selected": 3}},
            quality_scores={"SMALL": [80.0, 85.0, 90.0]},
            selection_ratios={"SMALL": 0.5}
        )
        
        assert report.total_available == 10
        assert report.total_selected == 5
        assert report.selection_ratios["SMALL"] == 0.5


class TestBasicFunctionality:
    """기본 기능 테스트"""
    
    @pytest.fixture
    def simple_config(self):
        """간단한 테스트 설정"""
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
    
    def test_empty_commits_list(self, simple_config):
        """빈 커밋 목록 처리"""
        selector = DiversityBasedSelector(simple_config)
        result = selector.select_diverse_commits([], 5)
        assert result == []
    
    def test_single_category_commits(self, simple_config):
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
        
        selector = DiversityBasedSelector(simple_config)
        selected = selector.select_diverse_commits(commits, 5)
        
        # 모든 커밋이 같은 카테고리여도 선택됨
        assert len(selected) == 3
    
    def test_more_requested_than_available(self, simple_config):
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
        
        selector = DiversityBasedSelector(simple_config)
        selected = selector.select_diverse_commits(commits, 10)
        
        # 사용 가능한 모든 커밋 반환
        assert len(selected) == 1