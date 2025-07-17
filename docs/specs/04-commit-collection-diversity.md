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
- 현재 저장소당 5개 커밋 → 평가 품질 향상을 위해 20개 필요
- 제한된 샘플로 인한 평가 신뢰성 저하
- 연구 데이터 기반 5개 카테고리 분산을 위한 충분한 샘플 필요

### 4. 카테고리별 배분 미고려
- 극소/소/중/대/극대 규모 변경의 적절한 비율 부재 (연구: 20/55/11/4/10%)
- 실제 개발 환경의 커밋 분포와 심각한 괴리 (현재: 대규모 편향)
- 오픈소스 프로젝트에서 75%가 소규모인 현실과 불일치

## 해결 방안 설계

### 1. 커밋 크기 분류 시스템

#### 연구 데이터 기반 분류 체계

오픈소스 프로젝트 연구 결과에 따르면, 커밋 크기 분포는 다음과 같습니다:
- 극소규모: 20% (0-5라인)
- 소규모: 55% (6-46라인)
- 중간규모: 11% (47-106라인)
- 대규모: 4% (107-166라인)
- 극대규모: 10% (167라인 이상)

#### CommitSizeCategory Enum
```python
from enum import Enum

class CommitSizeCategory(Enum):
    EXTRA_SMALL = "extra_small"  # 극소규모 변경 (0-5 라인)
    SMALL = "small"              # 소규모 변경 (6-46 라인)
    MEDIUM = "medium"            # 중간규모 변경 (47-106 라인)
    LARGE = "large"              # 대규모 변경 (107-166 라인)
    EXTRA_LARGE = "extra_large"  # 극대규모 변경 (167+ 라인)
```

#### 분류 기준 (라인 수 우선 + 파일 수 보정)
- **EXTRA_SMALL (극소규모)**: 총 변경 라인 수 0-5라인
  - 특징: 타이포 수정, 간단한 설정 변경, 작은 버그 수정
  - 평가 중점: 정확성, 세부사항

- **SMALL (소규모)**: 총 변경 라인 수 6-46라인
  - 특징: 간단한 버그 수정, 작은 기능 추가, 유틸리티 함수
  - 평가 중점: 정확성, 일관성, 로직 검증

- **MEDIUM (중간규모)**: 총 변경 라인 수 47-106라인
  - 특징: 기능 개선, 중간 규모 버그 수정, 리팩토링
  - 평가 중점: 로직 정확성, 코드 품질

- **LARGE (대규모)**: 총 변경 라인 수 107-166라인
  - 특징: 새로운 기능 구현, 대규모 버그 수정
  - 평가 중점: 설계 품질, 복잡성 관리

- **EXTRA_LARGE (극대규모)**: 총 변경 라인 수 167라인 이상
  - 특징: 주요 기능 추가, 대규모 리팩토링, 아키텍처 변경
  - 평가 중점: 전체적인 설계 품질, 아키텍처 일관성

#### 파일 수 보정 규칙
- **파일 수 11개 이상**: 최소 LARGE 카테고리로 상향 조정
- **단일 파일 + 100라인 초과**: 최대 LARGE 카테고리로 제한

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

#### YAML 기반 설정 시스템

하드코딩된 설정 대신 유연한 YAML 설정 파일을 사용합니다.

**설정 파일 위치**: `src/selvage_eval/config/commit-collection-config.yml`

```python
# 지원 설정 클래스들
class CommitCategoryConfig(BaseModel):
    """커밋 카테고리별 설정"""
    target_ratio: float = Field(ge=0.0, le=1.0)
    min_count: int = Field(ge=0)
    max_count: int = Field(ge=0)
    score_boost: int = Field(ge=0)
    description: Optional[str] = None

class CommitSizeThresholds(BaseModel):
    """커밋 크기 분류 임계값"""
    extra_small_max: int = 5
    small_max: int = 46
    medium_max: int = 106
    large_max: int = 166

class FileCorrectionConfig(BaseModel):
    """파일 수 보정 설정"""
    large_file_threshold: int = 11
    single_file_large_lines: int = 100

class CommitDiversityConfig(BaseModel):
    """커밋 다양성 설정 (YAML에서 로드)"""
    enabled: bool = True
    size_thresholds: CommitSizeThresholds = Field(default_factory=CommitSizeThresholds)
    file_correction: FileCorrectionConfig = Field(default_factory=FileCorrectionConfig)
    categories: Dict[str, CommitCategoryConfig] = Field(default_factory=dict)
    selection_algorithm: SelectionAlgorithmConfig = Field(default_factory=SelectionAlgorithmConfig)
    quality_scoring: QualityScoringConfig = Field(default_factory=QualityScoringConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

class EvaluationConfig(BaseModel):
    # 기존 필드들...
    commits_per_repo: int = 20  # 5에서 20으로 증가
    commit_diversity: Optional[CommitDiversityConfig] = None  # 자동 로드

# 설정 로딩 함수
def load_commit_diversity_config(config_path: Optional[str] = None) -> CommitDiversityConfig:
    """커밋 다양성 설정 파일 로드"""
    if config_path is None:
        config_dir = Path(__file__).parent
        config_path = str(config_dir / "commit-collection-config.yml")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        diversity_data = config_data.get('commit_diversity', {})
        return CommitDiversityConfig(**diversity_data)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return CommitDiversityConfig()  # 기본값 사용
```

**실제 구현된 YAML 설정 파일**:
```yaml
# 커밋 수집 다양성 설정
commits_per_repo: 20

commit_diversity:
  enabled: true
  
  # 커밋 크기별 분류 기준
  size_thresholds:
    extra_small_max: 5
    small_max: 46
    medium_max: 106
    large_max: 166
  
  # 파일 수 보정 기준
  file_correction:
    large_file_threshold: 11
    single_file_large_lines: 100
  
  # 카테고리별 선택 비율 및 제약
  categories:
    extra_small:
      target_ratio: 0.20
      min_count: 2
      max_count: 6
      score_boost: 15
      description: "극소규모 변경 (타이포, 간단한 설정)"
    small:
      target_ratio: 0.55
      min_count: 8
      max_count: 14
      score_boost: 10
      description: "소규모 변경 (간단한 버그 수정, 작은 기능)"
    medium:
      target_ratio: 0.11
      min_count: 1
      max_count: 4
      score_boost: 5
      description: "중간규모 변경 (기능 개선, 리팩토링)"
    large:
      target_ratio: 0.04
      min_count: 0
      max_count: 2
      score_boost: 0
      description: "대규모 변경 (새로운 기능 구현)"
    extra_large:
      target_ratio: 0.10
      min_count: 1
      max_count: 3
      score_boost: 0
      description: "극대규모 변경 (주요 기능, 아키텍처 변경)"

# 선택 알고리즘 설정
selection_algorithm:
  allocation_method: "proportional"
  shortage_handling: "redistribute"
  surplus_strategy: "quality_first"

# 품질 점수 조정
quality_scoring:
  diversity_weight: 0.3
  min_quality_scores:
    extra_small: 60
    small: 65
    medium: 70
    large: 75
    extra_large: 80
```

### 4. 구현 인터페이스 및 변경점

#### CommitCollector 클래스 확장
```python
class CommitCollector:
    def __init__(self, config: EvaluationConfig, tool_executor: ToolExecutor):
        # 기존 초기화...
        # 커밋 다양성 설정 로드
        if config.commit_diversity is None:
            config.commit_diversity = load_commit_diversity_config()
        
        self.diversity_selector = DiversityBasedSelector(
            config.commit_diversity.categories
        ) if config.commit_diversity.enabled else None
    
    def _categorize_commit(self, commit: CommitData) -> CommitSizeCategory:
        """라인 수 우선 + 파일 수 보정 방식으로 커밋 분류"""
        stats = commit.stats
        total_lines = stats.total_lines_changed
        files_changed = stats.files_changed
        
        # 1차: 라인 수 기준 분류
        if total_lines <= 5:
            base_category = CommitSizeCategory.EXTRA_SMALL
        elif total_lines <= 46:
            base_category = CommitSizeCategory.SMALL
        elif total_lines <= 106:
            base_category = CommitSizeCategory.MEDIUM
        elif total_lines <= 166:
            base_category = CommitSizeCategory.LARGE
        else:
            base_category = CommitSizeCategory.EXTRA_LARGE
        
        # 2차: 파일 수 보정
        if files_changed >= 11:  # 극대규모 파일 수
            return max(base_category, CommitSizeCategory.LARGE, 
                      key=lambda x: list(CommitSizeCategory).index(x))
        elif files_changed == 1 and total_lines > 100:  # 단일 파일 큰 변경
            return min(base_category, CommitSizeCategory.LARGE,
                      key=lambda x: list(CommitSizeCategory).index(x))
        
        return base_category
    
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

1. **평가 신뢰성 향상**: 5가지 크기의 커밋으로 더 포괄적이고 세밀한 평가
2. **실제 환경 반영**: 오픈소스 프로젝트 연구 결과(75% 소규모)와 일치하는 분포
3. **세밀한 분석 가능**: 커밋 크기별 특성에 맞는 세분화된 평가 및 AI 분석
4. **편향 제거**: 대규모 커밋 선호 편향 완전 해소, 극소규모 커밋도 적절히 포함
5. **커밋 수 증가**: 20개 커밋으로 더 많은 데이터 포인트와 통계적 유의성

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

**메인 설정 파일** (`selvage-eval-config.yml`):
```yaml
commits_per_repo: 20
commit_diversity: null  # commit-collection-config.yml에서 자동 로드
```

**전용 커밋 수집 설정 파일** (`src/selvage_eval/config/commit-collection-config.yml`):

실제 구현된 전체 설정 파일은 다음과 같습니다:
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

**YAML 설정 시스템 장점**:
- 설정 변경 시 코드 수정 불필요
- 프로덕션 환경에서 유연한 실시간 조정 가능
- 설정 버전 관리 및 백업 용이
- 다른 프로젝트에서 재사용 가능
- 연구 데이터 기반 과학적 기본값 제공
- 다양한 환경별 맞춤 설정 가능

### 사용 방법

```python
# 기본 설정 로드
from selvage_eval.config.settings import load_commit_diversity_config

# 자동 로딩 (기본 경로)
config = load_commit_diversity_config()

# 커스텀 경로 지정
config = load_commit_diversity_config("/path/to/custom-config.yml")

# 메인 설정과 통합
main_config = load_config("selvage-eval-config.yml")
# commit_diversity는 자동으로 로드됨
```

## 최종 결론

이 명세서를 통해 **연구 데이터에 기반한 과학적이고 현실적인 커밋 수집 다양성**을 달성하여, 더 정확하고 신뢰할 수 있는 Selvage 평가 시스템을 구축할 수 있습니다.

### 핵심 성과

1. **과학적 기반**: 9개 오픈소스 프로젝트 분석 결과 반영
2. **현실적 분포**: 75% 소규모 커밋 현실과 일치
3. **유연한 설정**: YAML 기반 외부 설정 시스템
4. **다양성 보장**: 5가지 커밋 크기로 세밀한 분석
5. **편향 제거**: 대규모 커밋 선호 편향 완전 해소

### 기대 효과

- **DeepEval 평가 신뢰성 향상**: 다양한 커밋 크기로 더 정확한 평가
- **AI 분석 개선**: 커밋 크기별 맞춤형 분석 전략
- **운영 효율성**: 설정 기반 유연한 조정 및 관리

### 연구 데이터 참고

본 명세서는 다음 학술 연구를 기반으로 작성되었습니다:
- Kent State University의 9개 오픈소스 프로젝트 커밋 분석 연구
- IEEE 및 기타 학술지에 발표된 커밋 크기 분포 연구
- Linux 커널 및 주요 오픈소스 프로젝트 분석 결과

이를 통해 실제 소프트웨어 개발 환경의 커밋 패턴을 정확히 반영하여 DeepEval 평가의 신뢰성과 정확성을 크게 향상시킬 수 있습니다.