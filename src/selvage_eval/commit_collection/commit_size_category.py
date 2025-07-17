"""커밋 크기 분류 enum 모듈

연구 데이터 기반의 커밋 크기 분류 체계를 제공합니다.
오픈소스 프로젝트 9개 분석 결과를 반영한 과학적 분류입니다.
"""

from enum import Enum
from typing import Dict, Tuple, Any


class CommitSizeCategory(Enum):
    """커밋 크기 카테고리 분류
    
    연구 데이터 기반 분류:
    - Kent State University 오픈소스 프로젝트 분석 결과 반영
    - 실제 개발 환경의 커밋 분포와 일치
    """
    
    EXTRA_SMALL = "extra_small"  # 극소규모 변경 (0-5 라인)
    SMALL = "small"              # 소규모 변경 (6-46 라인)
    MEDIUM = "medium"            # 중간규모 변경 (47-106 라인)
    LARGE = "large"              # 대규모 변경 (107-166 라인)
    EXTRA_LARGE = "extra_large"  # 극대규모 변경 (167+ 라인)

    @classmethod
    def get_category_info(cls) -> Dict[str, Dict[str, Any]]:
        """카테고리별 상세 정보 반환"""
        return {
            cls.EXTRA_SMALL.value: {
                "description": "극소규모 변경 (타이포, 간단한 설정)",
                "line_range": (0, 5),
                "research_ratio": 0.199,  # 19.9%
                "characteristics": ["타이포 수정", "간단한 설정 변경", "작은 버그 수정"],
                "eval_focus": ["정확성", "세부사항"]
            },
            cls.SMALL.value: {
                "description": "소규모 변경 (간단한 버그 수정, 작은 기능)",
                "line_range": (6, 46),
                "research_ratio": 0.553,  # 55.3%
                "characteristics": ["간단한 버그 수정", "작은 기능 추가", "유틸리티 함수"],
                "eval_focus": ["정확성", "일관성", "로직 검증"]
            },
            cls.MEDIUM.value: {
                "description": "중간규모 변경 (기능 개선, 리팩토링)",
                "line_range": (47, 106),
                "research_ratio": 0.111,  # 11.1%
                "characteristics": ["기능 개선", "중간 규모 버그 수정", "리팩토링"],
                "eval_focus": ["로직 정확성", "코드 품질"]
            },
            cls.LARGE.value: {
                "description": "대규모 변경 (새로운 기능 구현)",
                "line_range": (107, 166),
                "research_ratio": 0.043,  # 4.3%
                "characteristics": ["새로운 기능 구현", "대규모 버그 수정"],
                "eval_focus": ["설계 품질", "복잡성 관리"]
            },
            cls.EXTRA_LARGE.value: {
                "description": "극대규모 변경 (주요 기능, 아키텍처 변경)",
                "line_range": (167, float('inf')),
                "research_ratio": 0.094,  # 9.4%
                "characteristics": ["주요 기능 추가", "대규모 리팩토링", "아키텍처 변경"],
                "eval_focus": ["전체적인 설계 품질", "아키텍처 일관성"]
            }
        }

    @classmethod
    def categorize_by_lines(cls, total_lines: int, files_changed: int = 1) -> 'CommitSizeCategory':
        """라인 수와 파일 수를 기반으로 커밋 분류
        
        Args:
            total_lines: 총 변경 라인 수 (추가 + 삭제)
            files_changed: 변경된 파일 수
            
        Returns:
            해당하는 커밋 크기 카테고리
        """
        # 1차: 라인 수 기준 분류
        if total_lines <= 5:
            base_category = cls.EXTRA_SMALL
        elif total_lines <= 46:
            base_category = cls.SMALL
        elif total_lines <= 106:
            base_category = cls.MEDIUM
        elif total_lines <= 166:
            base_category = cls.LARGE
        else:
            base_category = cls.EXTRA_LARGE
        
        # 2차: 파일 수 보정
        if files_changed >= 11:  # 파일 수가 많으면 최소 LARGE
            return max(base_category, cls.LARGE, key=lambda x: list(cls).index(x))
        elif files_changed == 1 and total_lines > 100:  # 단일 파일 큰 변경은 최대 LARGE
            return min(base_category, cls.LARGE, key=lambda x: list(cls).index(x))
        
        return base_category

    @classmethod
    def get_line_thresholds(cls) -> Dict[str, Tuple[int, float]]:
        """카테고리별 라인 수 임계값 반환"""
        return {
            cls.EXTRA_SMALL.value: (0, 5),
            cls.SMALL.value: (6, 46),
            cls.MEDIUM.value: (47, 106),
            cls.LARGE.value: (107, 166),
            cls.EXTRA_LARGE.value: (167, float('inf'))
        }

    @classmethod
    def get_research_distribution(cls) -> Dict[str, float]:
        """연구 데이터 기반 분포 비율 반환"""
        info = cls.get_category_info()
        return {category: info[category]["research_ratio"] for category in info}

    def __str__(self) -> str:
        """문자열 표현"""
        info = self.get_category_info()[self.value]
        line_range = info["line_range"]
        if line_range[1] == float('inf'):
            range_str = f"{line_range[0]}+ 라인"
        else:
            range_str = f"{line_range[0]}-{line_range[1]} 라인"
        return f"{info['description']} ({range_str})"

    def __repr__(self) -> str:
        """개발자용 표현"""
        return f"CommitSizeCategory.{self.name}"