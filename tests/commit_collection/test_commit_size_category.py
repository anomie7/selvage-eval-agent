"""CommitSizeCategory 테스트 모듈"""

import pytest

from selvage_eval.commit_collection.commit_size_category import CommitSizeCategory


class TestCommitSizeCategory:
    """CommitSizeCategory 테스트"""
    
    def test_categorize_by_lines_basic(self):
        """기본 라인 수 분류 테스트"""
        assert CommitSizeCategory.categorize_by_lines(3) == CommitSizeCategory.EXTRA_SMALL
        assert CommitSizeCategory.categorize_by_lines(25) == CommitSizeCategory.SMALL
        assert CommitSizeCategory.categorize_by_lines(75) == CommitSizeCategory.MEDIUM
        assert CommitSizeCategory.categorize_by_lines(150) == CommitSizeCategory.LARGE
        # 단일 파일 큰 변경은 LARGE로 제한되므로 여러 파일로 테스트
        assert CommitSizeCategory.categorize_by_lines(200, files_changed=5) == CommitSizeCategory.EXTRA_LARGE
    
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
    
    def test_line_thresholds(self):
        """라인 임계값 테스트"""
        thresholds = CommitSizeCategory.get_line_thresholds()
        
        assert thresholds["extra_small"] == (0, 5)
        assert thresholds["small"] == (6, 46)
        assert thresholds["medium"] == (47, 106)
        assert thresholds["large"] == (107, 166)
        assert thresholds["extra_large"] == (167, float('inf'))
    
    def test_string_representation(self):
        """문자열 표현 테스트"""
        xs = CommitSizeCategory.EXTRA_SMALL
        assert "극소규모 변경" in str(xs)
        assert "0-5 라인" in str(xs)
        
        xl = CommitSizeCategory.EXTRA_LARGE
        assert "극대규모 변경" in str(xl)
        assert "167+ 라인" in str(xl)
    
    def test_repr(self):
        """개발자용 표현 테스트"""
        assert repr(CommitSizeCategory.SMALL) == "CommitSizeCategory.SMALL"
    
    def test_boundary_values(self):
        """경계값 테스트"""
        # 경계값들 (파일 수 보정 없이)
        assert CommitSizeCategory.categorize_by_lines(5) == CommitSizeCategory.EXTRA_SMALL
        assert CommitSizeCategory.categorize_by_lines(6) == CommitSizeCategory.SMALL
        assert CommitSizeCategory.categorize_by_lines(46) == CommitSizeCategory.SMALL
        assert CommitSizeCategory.categorize_by_lines(47) == CommitSizeCategory.MEDIUM
        assert CommitSizeCategory.categorize_by_lines(106) == CommitSizeCategory.MEDIUM
        assert CommitSizeCategory.categorize_by_lines(107) == CommitSizeCategory.LARGE
        assert CommitSizeCategory.categorize_by_lines(166) == CommitSizeCategory.LARGE
        # 단일 파일 큰 변경은 LARGE로 제한되므로 여러 파일로 테스트
        assert CommitSizeCategory.categorize_by_lines(167, files_changed=5) == CommitSizeCategory.EXTRA_LARGE