"""다양성 기반 커밋 선택기 모듈

연구 데이터 기반의 다양성을 고려한 커밋 선택 알고리즘을 제공합니다.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from selvage_eval.commit_collection.commit_data import CommitData

from selvage_eval.commit_collection.commit_size_category import CommitSizeCategory
from selvage_eval.config.settings import CommitDiversityConfig




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
            target_count = round(total_count * category_config.target_ratio)
            
            # 제약 조건 적용
            min_count = category_config.min_count
            available_count = len(categorized_commits[category])
            max_count = min(category_config.max_count, available_count)
            
            # 사용 가능한 커밋이 없으면 0
            if available_count == 0:
                target_count = 0
            else:
                # 최소값 보장 및 최대값 제한
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
        added_count = 0
        for commit, _, category in candidates:
            if added_count >= shortage:
                break
                
            allocation = allocations[category]
            # 최대치 제한 확인
            if allocation.selected_count >= allocation.max_count:
                continue
                
            additional_selected.append(commit)
            allocation.selected_count += 1
            added_count += 1
        
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