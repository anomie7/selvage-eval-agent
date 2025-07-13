# 커밋 수집 다양성 개선 명세서

## 개요

현재 Selvage 평가 에이전트의 커밋 수집 시스템은 변경량이 많은 커밋을 과도하게 선호하여 리뷰 대상 커밋의 다양성이 부족한 문제가 있습니다. 이 명세서는 DeepEval 분석 결과를 토대로 커밋 수집의 다양성을 개선하여 더 균형잡힌 평가를 가능하게 하는 개선 방안을 제시합니다.

## 현재 문제점 분석

### 1. 편향된 커밋 선택
- `_calculate_scale_appropriateness_score` 메서드에서 50-200라인 변경 시 최고점(15점) 부여
- 큰 변경사항에 과도한 가중치 부여로 소규모 의미있는 커밋 누락
- 변경량 기준의 단순한 우선순위 체계

### 2. 다양성 부족
- `_select_top_commits` 메서드에서 총점만을 기준으로 상위 선택
- 커밋 크기별, 특성별 다양성 고려 없음
- 유사한 성격의 커밋들만 선별되는 경향

### 3. 커밋 개수 부족
- 현재 저장소당 5개 커밋 → 평가 품질 향상을 위해 10개 이상 필요
- 제한된 샘플로 인한 평가 신뢰성 저하

### 4. 카테고리별 배분 미고려
- 대/중/소 규모 변경의 적절한 비율 부재
- 실제 개발 환경의 커밋 분포와 괴리

## 해결 방안 설계

### 1. 커밋 크기 분류 시스템

#### CommitSizeCategory Enum
```python
from enum import Enum

class CommitSizeCategory(Enum):
    LARGE = "large"      # 대규모 변경 (200+ 라인)
    MEDIUM = "medium"    # 중규모 변경 (50-199 라인)
    SMALL = "small"      # 소규모 변경 (10-49 라인)
```

#### 분류 기준
- **LARGE (대규모)**: 총 변경 라인 수 200+ 또는 파일 수 8+
  - 특징: 기능 추가, 대규모 리팩토링, 아키텍처 변경
  - 평가 중점: 전체적인 설계 품질, 복잡성 관리

- **MEDIUM (중규모)**: 총 변경 라인 수 50-199 또는 파일 수 3-7
  - 특징: 기능 개선, 버그 수정, 중간 규모 리팩토링
  - 평가 중점: 로직 정확성, 코드 품질

- **SMALL (소규모)**: 총 변경 라인 수 10-49 또는 파일 수 1-2
  - 특징: 간단한 버그 수정, 작은 개선사항, 유틸리티 추가
  - 평가 중점: 정확성, 일관성, 세부사항

### 2. 다양성 기반 선택 알고리즘

#### DiversityBasedSelector 클래스 설계
```python
@dataclass
class CommitCategoryConfig:
    """커밋 카테고리별 설정"""
    target_ratio: float      # 목표 비율 (0.0-1.0)
    min_count: int          # 최소 선택 개수
    max_count: int          # 최대 선택 개수
    score_boost: int        # 다양성을 위한 점수 보정

class DiversityBasedSelector:
    """다양성을 고려한 커밋 선택기"""
    
    def __init__(self, category_configs: Dict[CommitSizeCategory, CommitCategoryConfig]):
        self.category_configs = category_configs
    
    def select_diverse_commits(
        self, 
        commits: List[CommitData], 
        total_count: int
    ) -> List[CommitData]:
        """카테고리별 다양성을 고려한 커밋 선택"""
        pass
```

#### 선택 알고리즘 로직
1. **카테고리별 분류**: 모든 커밋을 크기 기준으로 분류
2. **할당량 계산**: 총 개수와 설정 비율에 따른 카테고리별 할당량 계산
3. **품질 기반 선택**: 각 카테고리 내에서 점수 기준 상위 선택
4. **최소 보장**: 각 카테고리별 최소 개수 보장
5. **여분 배분**: 남은 할당량을 품질 순으로 재배분

### 3. 설정 확장 사항

#### 새로운 설정 클래스
```python
class CommitDiversityConfig(BaseModel):
    """커밋 다양성 설정"""
    enabled: bool = True
    category_configs: Dict[str, CommitCategoryConfig] = Field(default_factory=dict)
    
    # 기본 카테고리별 설정
    large_commits: CommitCategoryConfig = CommitCategoryConfig(
        target_ratio=0.5,    # 50%
        min_count=2,
        max_count=8,
        score_boost=0
    )
    medium_commits: CommitCategoryConfig = CommitCategoryConfig(
        target_ratio=0.3,    # 30%
        min_count=1,
        max_count=5,
        score_boost=5
    )
    small_commits: CommitCategoryConfig = CommitCategoryConfig(
        target_ratio=0.2,    # 20%
        min_count=1,
        max_count=3,
        score_boost=10
    )

class EvaluationConfig(BaseModel):
    # 기존 필드들...
    commits_per_repo: int = 12  # 5에서 12로 증가
    commit_diversity: CommitDiversityConfig = Field(default_factory=CommitDiversityConfig)
```

### 4. 구현 인터페이스 및 변경점

#### CommitCollector 클래스 확장
```python
class CommitCollector:
    def __init__(self, config: EvaluationConfig, tool_executor: ToolExecutor):
        # 기존 초기화...
        self.diversity_selector = DiversityBasedSelector(
            config.commit_diversity.get_category_configs()
        ) if config.commit_diversity.enabled else None
    
    def _categorize_commit(self, commit: CommitData) -> CommitSizeCategory:
        """커밋을 크기별로 분류"""
        stats = commit.stats
        total_lines = stats.total_lines_changed
        
        if total_lines >= 200 or stats.files_changed >= 8:
            return CommitSizeCategory.LARGE
        elif total_lines >= 50 or stats.files_changed >= 3:
            return CommitSizeCategory.MEDIUM
        else:
            return CommitSizeCategory.SMALL
    
    def _select_diverse_commits(
        self, 
        commits: List[CommitData], 
        count: int
    ) -> List[CommitData]:
        """다양성을 고려한 커밋 선택 (기존 _select_top_commits 대체)"""
        if not self.diversity_selector:
            return self._select_top_commits(commits, count)
        
        return self.diversity_selector.select_diverse_commits(commits, count)
```

#### 점수 체계 개선
```python
def _calculate_diversity_adjusted_score(
    self, 
    commit: CommitData, 
    category: CommitSizeCategory
) -> int:
    """다양성을 고려한 점수 조정"""
    base_score = self._calculate_base_score(commit)
    category_boost = self.diversity_config.get_score_boost(category)
    
    return base_score + category_boost
```

### 5. 테스트 시나리오

#### 다양성 검증 테스트
1. **카테고리별 분포 테스트**: 선택된 커밋이 설정된 비율에 맞는지 검증
2. **품질 유지 테스트**: 다양성 확보로 인한 품질 저하가 없는지 확인
3. **엣지 케이스 테스트**: 특정 카테고리 커밋이 부족한 경우 처리 검증

#### 성능 측정
- 기존 대비 선택 다양성 개선 정도 측정
- DeepEval 메트릭별 평가 품질 변화 분석
- 처리 시간 및 메모리 사용량 변화 모니터링

## 기대 효과

1. **평가 신뢰성 향상**: 다양한 크기의 커밋으로 더 포괄적인 평가
2. **실제 환경 반영**: 실제 개발 환경의 커밋 분포와 유사한 샘플링
3. **세밀한 분석 가능**: 커밋 크기별 특성에 맞는 세분화된 평가
4. **편향 제거**: 큰 변경사항 선호 편향 해소

## 구현 우선순위

1. **1단계**: CommitSizeCategory 및 분류 로직 구현
2. **2단계**: DiversityBasedSelector 클래스 구현
3. **3단계**: 설정 확장 및 통합
4. **4단계**: 테스트 및 검증
5. **5단계**: 성능 최적화 및 문서화

## 마이그레이션 가이드

### 기존 코드와의 호환성
- 기존 `_select_top_commits` 메서드는 fallback으로 유지
- `commit_diversity.enabled = false` 설정으로 기존 방식 사용 가능
- 점진적 마이그레이션을 통한 안정성 확보

### 설정 파일 업데이트
```yaml
commits_per_repo: 12
commit_diversity:
  enabled: true
  large_commits:
    target_ratio: 0.5
    min_count: 2
    max_count: 8
  medium_commits:
    target_ratio: 0.3
    min_count: 1
    max_count: 5
  small_commits:
    target_ratio: 0.2
    min_count: 1
    max_count: 3
```

이 명세서를 통해 커밋 수집의 다양성을 개선하여 더 균형잡히고 신뢰할 수 있는 Selvage 평가 시스템을 구축할 수 있습니다.