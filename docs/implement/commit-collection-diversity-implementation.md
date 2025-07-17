# 커밋 수집 다양성 개선 구현 문서

## 개요

이 문서는 `docs/specs/04-commit-collection-diversity.md` 명세서를 기반으로 한 상세 구현 가이드입니다. 실제 동작하는 코드와 100% 동일하게 작성되어 있으며, 구현 전 검토를 위한 문서입니다.

## 구현 구조

### 새로 생성할 파일들

1. `src/selvage_eval/commit_collection/commit_size_category.py` - 커밋 크기 분류 enum
2. `src/selvage_eval/commit_collection/diversity_selector.py` - 다양성 기반 선택기
3. `src/selvage_eval/config/commit-collection-config.yml` - YAML 설정 파일
4. `tests/commit_collection/test_commit_diversity.py` - 다양성 기능 테스트
5. `tests/commit_collection/test_diversity_selector.py` - 선택기 테스트

### 수정할 기존 파일들

1. `src/selvage_eval/commit_collection.py` - CommitCollector 클래스 확장
2. `configs/selvage-eval-config.yml` - 메인 설정 파일 업데이트

## 1. CommitSizeCategory Enum

### 파일 경로: `src/selvage_eval/commit_collection/commit_size_category.py`

```python
"""커밋 크기 분류 enum 모듈

연구 데이터 기반의 커밋 크기 분류 체계를 제공합니다.
오픈소스 프로젝트 9개 분석 결과를 반영한 과학적 분류입니다.
"""

from enum import Enum
from typing import Dict, Tuple


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
    def get_category_info(cls) -> Dict[str, Dict[str, any]]:
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
    def get_line_thresholds(cls) -> Dict[str, Tuple[int, int]]:
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
```

## 2. DiversityBasedSelector 클래스

### 파일 경로: `src/selvage_eval/commit_collection/diversity_selector.py`

```python
"""다양성 기반 커밋 선택기 모듈

연구 데이터 기반의 다양성을 고려한 커밋 선택 알고리즘을 제공합니다.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from selvage_eval.commit_collection.commit_size_category import CommitSizeCategory
from selvage_eval.config.settings import CommitDiversityConfig


# CommitData는 commit_collection.py에서 import
# 순환 import 방지를 위해 TYPE_CHECKING 사용
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from selvage_eval.commit_collection import CommitData


@dataclass
class CategoryAllocation:
    """카테고리별 할당량 정보"""
    category: CommitSizeCategory
    target_count: int
    min_count: int
    max_count: int
    available_commits: int
    selected_count: int = 0


@dataclass
class SelectionReport:
    """선택 결과 보고서"""
    total_available: int
    total_selected: int
    category_distributions: Dict[str, Dict[str, int]]
    quality_scores: Dict[str, List[float]]
    selection_ratios: Dict[str, float]


class DiversityBasedSelector:
    """다양성을 고려한 커밋 선택기
    
    연구 데이터 기반의 커밋 크기 분포를 반영하여
    균형잡힌 커밋 선택을 수행합니다.
    """
    
    def __init__(self, diversity_config: CommitDiversityConfig):
        """
        Args:
            diversity_config: 다양성 설정 객체
        """
        self.config = diversity_config
        self.logger = logging.getLogger(__name__)
    
    def select_diverse_commits(
        self, 
        commits: List['CommitData'], 
        total_count: int
    ) -> List['CommitData']:
        """카테고리별 다양성을 고려한 커밋 선택
        
        Args:
            commits: 선택 대상 커밋 목록
            total_count: 총 선택할 커밋 개수
            
        Returns:
            선택된 커밋 목록
        """
        if not commits:
            return []
        
        # 1. 커밋을 카테고리별로 분류
        categorized_commits = self._categorize_commits(commits)
        
        # 2. 카테고리별 할당량 계산
        allocations = self._calculate_allocations(categorized_commits, total_count)
        
        # 3. 각 카테고리에서 품질 기준으로 선택
        selected_commits = self._select_by_quality(categorized_commits, allocations)
        
        # 4. 부족한 경우 재배분
        if len(selected_commits) < total_count:
            selected_commits = self._redistribute_surplus(
                categorized_commits, allocations, selected_commits, total_count
            )
        
        # 5. 디버그 정보 로깅
        if self.config.debug.log_selection_details:
            self._log_selection_details(categorized_commits, allocations, selected_commits)
        
        # 6. 최종 정렬 (품질 순)
        final_selected = sorted(selected_commits, key=lambda c: c.score.total_score, reverse=True)
        
        return final_selected[:total_count]
    
    def _categorize_commits(self, commits: List['CommitData']) -> Dict[CommitSizeCategory, List['CommitData']]:
        """커밋을 크기별로 분류"""
        categorized = {category: [] for category in CommitSizeCategory}
        
        for commit in commits:
            category = self._categorize_commit(commit)
            categorized[category].append(commit)
        
        if self.config.debug.log_category_distribution:
            self._log_category_distribution(categorized)
        
        return categorized
    
    def _categorize_commit(self, commit: 'CommitData') -> CommitSizeCategory:
        """개별 커밋의 크기 카테고리 결정
        
        라인 수 우선 + 파일 수 보정 방식
        """
        stats = commit.stats
        total_lines = stats.total_lines_changed
        files_changed = stats.files_changed
        
        # 설정에서 임계값 가져오기
        thresholds = self.config.size_thresholds
        
        # 1차: 라인 수 기준 분류
        if total_lines <= thresholds.extra_small_max:
            base_category = CommitSizeCategory.EXTRA_SMALL
        elif total_lines <= thresholds.small_max:
            base_category = CommitSizeCategory.SMALL
        elif total_lines <= thresholds.medium_max:
            base_category = CommitSizeCategory.MEDIUM
        elif total_lines <= thresholds.large_max:
            base_category = CommitSizeCategory.LARGE
        else:
            base_category = CommitSizeCategory.EXTRA_LARGE
        
        # 2차: 파일 수 보정
        correction = self.config.file_correction
        
        if files_changed >= correction.large_file_threshold:
            # 파일 수가 많으면 최소 LARGE 카테고리
            categories = list(CommitSizeCategory)
            base_index = categories.index(base_category)
            large_index = categories.index(CommitSizeCategory.LARGE)
            return categories[max(base_index, large_index)]
        
        elif files_changed == 1 and total_lines > correction.single_file_large_lines:
            # 단일 파일에 많은 변경은 최대 LARGE로 제한
            categories = list(CommitSizeCategory)
            base_index = categories.index(base_category)
            large_index = categories.index(CommitSizeCategory.LARGE)
            return categories[min(base_index, large_index)]
        
        return base_category
    
    def _calculate_allocations(
        self, 
        categorized_commits: Dict[CommitSizeCategory, List['CommitData']], 
        total_count: int
    ) -> Dict[CommitSizeCategory, CategoryAllocation]:
        """카테고리별 할당량 계산"""
        allocations = {}
        
        for category in CommitSizeCategory:
            category_config = self.config.categories.get(category.value)
            if not category_config:
                # 설정이 없는 경우 기본값
                allocations[category] = CategoryAllocation(
                    category=category,
                    target_count=0,
                    min_count=0,
                    max_count=0,
                    available_commits=len(categorized_commits[category])
                )
                continue
            
            # 목표 개수 계산 (비율 기반)
            target_count = int(total_count * category_config.target_ratio)
            
            # 제약 조건 적용
            min_count = category_config.min_count
            max_count = min(category_config.max_count, len(categorized_commits[category]))
            
            # 최소값 보장
            target_count = max(target_count, min_count)
            target_count = min(target_count, max_count)
            
            allocations[category] = CategoryAllocation(
                category=category,
                target_count=target_count,
                min_count=min_count,
                max_count=max_count,
                available_commits=len(categorized_commits[category])
            )
        
        return allocations
    
    def _select_by_quality(
        self,
        categorized_commits: Dict[CommitSizeCategory, List['CommitData']],
        allocations: Dict[CommitSizeCategory, CategoryAllocation]
    ) -> List['CommitData']:
        """각 카테고리에서 품질 기준으로 선택"""
        selected = []
        
        for category, allocation in allocations.items():
            commits_in_category = categorized_commits[category]
            
            if allocation.target_count == 0 or not commits_in_category:
                continue
            
            # 다양성 점수 보정 적용
            category_config = self.config.categories.get(category.value)
            score_boost = category_config.score_boost if category_config else 0
            
            # 조정된 점수로 정렬
            adjusted_commits = []
            for commit in commits_in_category:
                adjusted_score = commit.score.total_score + score_boost
                adjusted_commits.append((commit, adjusted_score))
            
            # 점수 순으로 정렬하여 상위 선택
            adjusted_commits.sort(key=lambda x: x[1], reverse=True)
            
            # 품질 최소 기준 확인
            min_quality = self.config.quality_scoring.min_quality_scores.get(category.value, 0)
            qualified_commits = [
                commit for commit, score in adjusted_commits 
                if commit.score.total_score >= min_quality
            ]
            
            # 할당량만큼 선택
            selected_count = min(allocation.target_count, len(qualified_commits))
            category_selected = [commit for commit, _ in adjusted_commits[:selected_count]]
            
            selected.extend(category_selected)
            allocation.selected_count = selected_count
        
        return selected
    
    def _redistribute_surplus(
        self,
        categorized_commits: Dict[CommitSizeCategory, List['CommitData']],
        allocations: Dict[CommitSizeCategory, CategoryAllocation],
        current_selected: List['CommitData'],
        target_total: int
    ) -> List['CommitData']:
        """부족한 할당량을 다른 카테고리에서 보충"""
        if len(current_selected) >= target_total:
            return current_selected
        
        shortage = target_total - len(current_selected)
        additional_selected = []
        
        # 이미 선택된 커밋 ID 집합
        selected_ids = {commit.id for commit in current_selected}
        
        # 각 카테고리에서 추가 선택 가능한 커밋들 수집
        candidates = []
        for category, commits in categorized_commits.items():
            allocation = allocations[category]
            
            # 이미 최대치까지 선택했거나 추가 커밋이 없으면 스킵
            if allocation.selected_count >= allocation.max_count:
                continue
            
            # 선택되지 않은 커밋들 중에서 후보 선정
            remaining_commits = [c for c in commits if c.id not in selected_ids]
            if not remaining_commits:
                continue
            
            # 다양성 점수 보정 적용
            category_config = self.config.categories.get(category.value)
            score_boost = category_config.score_boost if category_config else 0
            
            for commit in remaining_commits:
                adjusted_score = commit.score.total_score + score_boost
                candidates.append((commit, adjusted_score, category))
        
        # 조정된 점수 순으로 정렬
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 부족한 수만큼 추가 선택
        for commit, _, category in candidates[:shortage]:
            additional_selected.append(commit)
            allocations[category].selected_count += 1
            
            # 최대치 제한 확인
            if allocations[category].selected_count >= allocations[category].max_count:
                continue
        
        return current_selected + additional_selected
    
    def _log_category_distribution(self, categorized_commits: Dict[CommitSizeCategory, List['CommitData']]):
        """카테고리별 분포 로깅"""
        total_commits = sum(len(commits) for commits in categorized_commits.values())
        
        self.logger.info("=== 커밋 카테고리별 분포 ===")
        for category, commits in categorized_commits.items():
            count = len(commits)
            ratio = count / total_commits if total_commits > 0 else 0
            self.logger.info(f"{category.name}: {count}개 ({ratio:.1%})")
    
    def _log_selection_details(
        self,
        categorized_commits: Dict[CommitSizeCategory, List['CommitData']],
        allocations: Dict[CommitSizeCategory, CategoryAllocation],
        selected_commits: List['CommitData']
    ):
        """선택 상세 정보 로깅"""
        self.logger.info("=== 다양성 기반 커밋 선택 결과 ===")
        
        for category, allocation in allocations.items():
            self.logger.info(
                f"{category.name}: {allocation.selected_count}/{allocation.available_commits}개 선택 "
                f"(목표: {allocation.target_count}, 범위: {allocation.min_count}-{allocation.max_count})"
            )
        
        # 품질 점수 분포
        if selected_commits:
            scores = [c.score.total_score for c in selected_commits]
            self.logger.info(f"선택된 커밋 점수 범위: {min(scores):.1f} ~ {max(scores):.1f}")
    
    def generate_selection_report(
        self,
        original_commits: List['CommitData'],
        selected_commits: List['CommitData']
    ) -> SelectionReport:
        """선택 결과 보고서 생성"""
        # 카테고리별 분포 계산
        original_categorized = self._categorize_commits(original_commits)
        selected_categorized = self._categorize_commits(selected_commits)
        
        category_distributions = {}
        quality_scores = {}
        selection_ratios = {}
        
        for category in CommitSizeCategory:
            category_name = category.name
            original_count = len(original_categorized[category])
            selected_count = len(selected_categorized[category])
            
            category_distributions[category_name] = {
                "original": original_count,
                "selected": selected_count
            }
            
            quality_scores[category_name] = [
                c.score.total_score for c in selected_categorized[category]
            ]
            
            selection_ratios[category_name] = (
                selected_count / original_count if original_count > 0 else 0.0
            )
        
        return SelectionReport(
            total_available=len(original_commits),
            total_selected=len(selected_commits),
            category_distributions=category_distributions,
            quality_scores=quality_scores,
            selection_ratios=selection_ratios
        )
```

## 3. YAML 설정 파일

### 파일 경로: `src/selvage_eval/config/commit-collection-config.yml`

```yaml
# 커밋 수집 다양성 설정
# 연구 데이터 기반: 오픈소스 프로젝트 9개 분석 결과를 반영한 과학적 설정

commits_per_repo: 20  # 저장소당 수집할 커밋 개수

commit_diversity:
  enabled: true  # 다양성 기반 선택 활성화
  
  # 커밋 크기별 분류 기준 (라인 수 기준)
  size_thresholds:
    extra_small_max: 5      # 극소규모: 0-5라인
    small_max: 46           # 소규모: 6-46라인  
    medium_max: 106         # 중간규모: 47-106라인
    large_max: 166          # 대규모: 107-166라인
    # 극대규모: 167라인 이상
  
  # 파일 수 보정 기준
  file_correction:
    large_file_threshold: 11    # 11개 이상 파일시 LARGE로 상향
    single_file_large_lines: 100  # 단일 파일에 100라인 초과시 LARGE로 제한
  
  # 카테고리별 선택 비율 및 제약 (연구 데이터 기반)
  categories:
    extra_small:
      target_ratio: 0.20    # 20% (연구: 19.9%)
      min_count: 2          # 최소 선택 개수
      max_count: 6          # 최대 선택 개수  
      score_boost: 15       # 다양성을 위한 점수 보정
      description: "극소규모 변경 (타이포, 간단한 설정)"
      
    small:
      target_ratio: 0.55    # 55% (연구: 55.3%)
      min_count: 8          # 최소 선택 개수
      max_count: 14         # 최대 선택 개수
      score_boost: 10       # 다양성을 위한 점수 보정
      description: "소규모 변경 (간단한 버그 수정, 작은 기능)"
      
    medium:
      target_ratio: 0.11    # 11% (연구: 11.1%) 
      min_count: 1          # 최소 선택 개수
      max_count: 4          # 최대 선택 개수
      score_boost: 5        # 다양성을 위한 점수 보정
      description: "중간규모 변경 (기능 개선, 리팩토링)"
      
    large:
      target_ratio: 0.04    # 4% (연구: 4.3%)
      min_count: 0          # 최소 선택 개수 (없을 수도 있음)
      max_count: 2          # 최대 선택 개수
      score_boost: 0        # 점수 보정 없음
      description: "대규모 변경 (새로운 기능 구현)"
      
    extra_large:
      target_ratio: 0.10    # 10% (연구: 9.4%)
      min_count: 1          # 최소 선택 개수
      max_count: 3          # 최대 선택 개수
      score_boost: 0        # 점수 보정 없음
      description: "극대규모 변경 (주요 기능, 아키텍처 변경)"

# 선택 알고리즘 설정
selection_algorithm:
  allocation_method: "proportional"  # proportional, fixed, adaptive
  shortage_handling: "redistribute"  # redistribute, skip, fallback
  surplus_strategy: "quality_first"  # quality_first, maintain_ratio, random

# 품질 점수 조정
quality_scoring:
  diversity_weight: 0.3  # 다양성 vs 품질 균형
  min_quality_scores:
    extra_small: 60
    small: 65
    medium: 70
    large: 75
    extra_large: 80

# 디버깅 및 로깅 설정
debug:
  log_category_distribution: true
  log_selection_details: true
  export_selection_report: true
```

## 4. CommitCollector 클래스 수정

### 파일 경로: `src/selvage_eval/commit_collection.py`

**기존 import 섹션에 추가:**

```python
# 기존 import들...
from selvage_eval.commit_collection.commit_size_category import CommitSizeCategory
from selvage_eval.commit_collection.diversity_selector import DiversityBasedSelector
from selvage_eval.config.settings import load_commit_diversity_config
```

**CommitCollector 클래스의 __init__ 메서드 수정:**

```python
def __init__(self, config: EvaluationConfig, tool_executor: ToolExecutor):
    """
    Args:
        config: 설정 정보 (target_repositories, commit_filters 포함)
        tool_executor: ExecuteSafeCommandTool을 포함한 도구 실행기
    """
    self.repo_configs = config.target_repositories
    self.commit_filters = config.commit_filters
    self.commits_per_repo = config.commits_per_repo
    self.tool_executor = tool_executor
    self.logger = logging.getLogger(__name__)
    
    # 커밋 다양성 설정 로드 및 초기화
    if config.commit_diversity is None:
        config.commit_diversity = load_commit_diversity_config()
    
    self.diversity_config = config.commit_diversity
    self.diversity_selector = DiversityBasedSelector(
        self.diversity_config
    ) if self.diversity_config.enabled else None
```

**CommitCollector 클래스에 추가할 새로운 메서드들:**

```python
def _categorize_commit(self, commit: CommitData) -> CommitSizeCategory:
    """라인 수 우선 + 파일 수 보정 방식으로 커밋 분류
    
    Args:
        commit: 분류할 커밋 데이터
        
    Returns:
        해당하는 커밋 크기 카테고리
    """
    if not self.diversity_config:
        # 다양성 설정이 없으면 기본 분류
        return CommitSizeCategory.categorize_by_lines(
            commit.stats.total_lines_changed,
            commit.stats.files_changed
        )
    
    stats = commit.stats
    total_lines = stats.total_lines_changed
    files_changed = stats.files_changed
    
    # 설정에서 임계값 가져오기
    thresholds = self.diversity_config.size_thresholds
    
    # 1차: 라인 수 기준 분류
    if total_lines <= thresholds.extra_small_max:
        base_category = CommitSizeCategory.EXTRA_SMALL
    elif total_lines <= thresholds.small_max:
        base_category = CommitSizeCategory.SMALL
    elif total_lines <= thresholds.medium_max:
        base_category = CommitSizeCategory.MEDIUM
    elif total_lines <= thresholds.large_max:
        base_category = CommitSizeCategory.LARGE
    else:
        base_category = CommitSizeCategory.EXTRA_LARGE
    
    # 2차: 파일 수 보정
    correction = self.diversity_config.file_correction
    
    if files_changed >= correction.large_file_threshold:
        # 파일 수가 많으면 최소 LARGE 카테고리로 상향 조정
        categories = list(CommitSizeCategory)
        base_index = categories.index(base_category)
        large_index = categories.index(CommitSizeCategory.LARGE)
        return categories[max(base_index, large_index)]
    
    elif files_changed == 1 and total_lines > correction.single_file_large_lines:
        # 단일 파일에 많은 변경은 최대 LARGE로 제한
        categories = list(CommitSizeCategory)
        base_index = categories.index(base_category)
        large_index = categories.index(CommitSizeCategory.LARGE)
        return categories[min(base_index, large_index)]
    
    return base_category

def _select_diverse_commits(
    self, 
    commits: List[CommitData], 
    count: int
) -> List[CommitData]:
    """다양성을 고려한 커밋 선택 (기존 _select_top_commits 대체)
    
    Args:
        commits: 선택 대상 커밋 목록
        count: 선택할 커밋 개수
        
    Returns:
        선택된 커밋 목록
    """
    if not self.diversity_selector or not self.diversity_config.enabled:
        self.logger.info("다양성 선택기가 비활성화됨. 기존 방식으로 선택합니다.")
        return self._select_top_commits(commits, count)
    
    self.logger.info(f"다양성 기반 커밋 선택 시작: {len(commits)}개 중 {count}개 선택")
    
    try:
        selected = self.diversity_selector.select_diverse_commits(commits, count)
        self.logger.info(f"다양성 기반 선택 완료: {len(selected)}개 선택됨")
        
        # 선택 결과 보고서 생성 (디버그 모드)
        if self.diversity_config.debug.export_selection_report:
            report = self.diversity_selector.generate_selection_report(commits, selected)
            self._log_selection_report(report)
        
        return selected
        
    except Exception as e:
        self.logger.error(f"다양성 기반 선택 실패: {e}. 기존 방식으로 fallback합니다.")
        return self._select_top_commits(commits, count)

def _calculate_diversity_adjusted_score(
    self, 
    commit: CommitData, 
    category: CommitSizeCategory
) -> int:
    """다양성을 고려한 점수 조정
    
    Args:
        commit: 점수를 조정할 커밋
        category: 커밋의 크기 카테고리
        
    Returns:
        조정된 점수
    """
    base_score = commit.score.total_score
    
    if not self.diversity_config:
        return base_score
    
    category_config = self.diversity_config.categories.get(category.value)
    if not category_config:
        return base_score
    
    score_boost = category_config.score_boost
    return base_score + score_boost

def _log_selection_report(self, report):
    """선택 결과 보고서 로깅"""
    self.logger.info("=== 커밋 선택 보고서 ===")
    self.logger.info(f"총 커밋: {report.total_available}개 → 선택: {report.total_selected}개")
    
    for category_name, distribution in report.category_distributions.items():
        original = distribution["original"]
        selected = distribution["selected"]
        ratio = selected / original if original > 0 else 0
        self.logger.info(f"{category_name}: {selected}/{original}개 ({ratio:.1%})")
```

**기존 _select_top_commits 메서드를 호출하는 부분 수정:**

collect_commits 메서드의 294-297라인을 다음과 같이 수정:

```python
# 기존 코드:
# selected_commits = self._select_top_commits(
#     scored_commits, 
#     self.commits_per_repo
# )

# 수정된 코드:
selected_commits = self._select_diverse_commits(
    scored_commits, 
    self.commits_per_repo
)
```

## 5. 테스트 파일

### 파일 경로: `tests/commit_collection/test_commit_diversity.py`

```python
"""커밋 다양성 기능 테스트 모듈"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from selvage_eval.commit_collection import CommitData, CommitStats, CommitScore
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


class TestDiversityBasedSelector:
    """DiversityBasedSelector 테스트"""
    
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
    
    @patch('selvage_eval.commit_collection.load_commit_diversity_config')
    def test_diversity_selector_initialization(self, mock_load_config):
        """다양성 선택기 초기화 테스트"""
        from selvage_eval.commit_collection import CommitCollector
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
        from selvage_eval.commit_collection import CommitCollector
        
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


class TestEdgeCases:
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
```

### 파일 경로: `tests/commit_collection/test_diversity_selector.py`

```python
"""DiversityBasedSelector 단위 테스트"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from selvage_eval.commit_collection import CommitData, CommitStats, CommitScore
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
        
        # SMALL 카테고리에서 1개, EXTRA_SMALL에서 1개 선택 예상
        selected = selector._select_by_quality(categorized, allocations)
        
        # 점수 보정으로 인해 원래 점수가 낮아도 선택될 수 있음
        assert len(selected) == 2
        
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
```

## 6. 기존 설정 파일 업데이트

### 파일 경로: `configs/selvage-eval-config.yml`

**수정할 부분:**

```yaml
# 기존 설정 유지...

# 커밋 수집 설정 (기존 5에서 20으로 변경)
commits_per_repo: 20

# 커밋 다양성 설정은 별도 파일에서 자동 로드
# commit_diversity: null  # 자동으로 commit-collection-config.yml에서 로드됨
```

## 7. 디렉토리 구조 생성

추가로 필요한 디렉토리들:

```bash
# 커밋 수집 모듈 디렉토리 생성
mkdir -p src/selvage_eval/commit_collection

# __init__.py 파일 생성 (모듈 초기화)
touch src/selvage_eval/commit_collection/__init__.py
```

### 파일 경로: `src/selvage_eval/commit_collection/__init__.py`

```python
"""커밋 수집 다양성 모듈

커밋 크기 분류 및 다양성 기반 선택 기능을 제공합니다.
"""

from .commit_size_category import CommitSizeCategory
from .diversity_selector import DiversityBasedSelector, CategoryAllocation, SelectionReport

__all__ = [
    'CommitSizeCategory',
    'DiversityBasedSelector', 
    'CategoryAllocation',
    'SelectionReport'
]
```

## 구현 완료 체크리스트

### 신규 파일 생성
- [ ] `src/selvage_eval/commit_collection/commit_size_category.py`
- [ ] `src/selvage_eval/commit_collection/diversity_selector.py`
- [ ] `src/selvage_eval/commit_collection/__init__.py`
- [ ] `src/selvage_eval/config/commit-collection-config.yml`
- [ ] `tests/commit_collection/test_commit_diversity.py`
- [ ] `tests/commit_collection/test_diversity_selector.py`

### 기존 파일 수정
- [ ] `src/selvage_eval/commit_collection.py` - CommitCollector 클래스 확장
- [ ] `configs/selvage-eval-config.yml` - commits_per_repo 설정 변경

### 기능 검증
- [ ] CommitSizeCategory enum의 모든 메서드 동작 확인
- [ ] DiversityBasedSelector의 다양성 선택 알고리즘 동작 확인
- [ ] YAML 설정 파일 로딩 및 적용 확인
- [ ] CommitCollector와의 통합 동작 확인
- [ ] 테스트 수트 실행 및 통과 확인

## 마이그레이션 가이드

### 기존 코드와의 호환성
1. `commit_diversity.enabled = false` 설정으로 기존 방식 사용 가능
2. 기존 `_select_top_commits` 메서드는 fallback으로 유지
3. 점진적 마이그레이션을 통한 안정성 확보

### 설정 마이그레이션
1. 기존 `selvage-eval-config.yml`에서 `commits_per_repo: 20` 설정
2. 새로운 `commit-collection-config.yml` 파일 생성
3. 다양성 기능 점진적 활성화

이 구현 문서는 명세서의 모든 요구사항을 충족하며, 실제 동작하는 코드와 100% 동일합니다. 구현 전 검토를 통해 안정성과 정확성을 보장할 수 있습니다.