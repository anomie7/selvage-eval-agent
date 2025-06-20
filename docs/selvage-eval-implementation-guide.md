# Selvage 평가 에이전트 - 구현 가이드

## 프로젝트 개요
AI 기반 코드 리뷰 도구인 Selvage를 평가하는 자동화 에이전트입니다. 4단계 워크플로우를 통해 모델별 성능과 프롬프트 버전 효과성을 정량적으로 측정합니다.

## 에이전트 아키텍처

### 설계 원칙
- **모듈화**: 각 단계를 독립적인 모듈로 구현
- **재현성**: JSON 기반 데이터 저장으로 테스트 재현 가능
- **확장성**: 새로운 모델 및 평가 지표 추가 용이
- **견고성**: 에러 처리 및 재시도 로직 내장

### 핵심 구현 요구사항
- **Python 3.10+** (타입 힌팅 필수)
- **Google 스타일 독스트링** (한국어 주석)
- **PEP 8 준수**
- **비동기 처리** (다중 모델 병렬 평가)

## 사용 모델 전략
- **Primary**: `gemini-2.5-flash` (속도/비용 최적화)

## 대상 repo-path
- cline
    - path: /Users/demin_coder/Dev/cline
    - description: typescript로 구현된 coding assistant
- selvage-deprecated
    - path: /Users/demin_coder/Dev/selvage-deprecated
    - description: selvage가 정식 배포되기 전 commit history를 가지고 있는 repository (주의: 현재 selvage으 이전 작업 폴더이므로 review 대상으로서만 접근할 것)
- ecommerce-microservices
    - path: /Users/demin_coder/Dev/ecommerce-microservices
    - description: java, spring, jpa로 구현된 MSA 서버 애플리케이션
- kotlin-realworld
    - path: /Users/demin_coder/Dev/kotlin-realworld
    - description: java, kotlin, jpa로 구현된 호텔 예약 서버 애플리케이션


# 설정 파일

## CLI 실행 방식
터미널에서 `selvage-eval` 명령어로 바로 실행 가능하도록 설정 파일 기반 구성

### 설정 파일 스키마 (selvage-eval-config.yml)
```yaml
# Selvage 평가 에이전트 설정
agent-model: gemini-2.5-flash

evaluation:
  output_dir: "./selvage-eval-results"
  auto_session_id: true  # 자동 생성: eval_20240120_143022_abc123
  
target_repositories:
  - name: cline
    path: /Users/demin_coder/Dev/cline
    tech_stack: typescript
    description: "typescript로 구현된 coding assistant"
    filter_overrides:
      min_changed_lines: 30  # TS는 더 작은 단위 변경 허용
      file_types: [".ts", ".tsx", ".js", ".jsx"]
      
  - name: selvage-deprecated
    path: /Users/demin_coder/Dev/selvage-deprecated
    tech_stack: mixed
    description: "selvage 이전 버전 commit history"
    access_mode: readonly  # 읽기 전용 접근
    security_constraints:
      - no_write_operations
      - review_target_only
    filter_overrides:
      min_changed_lines: 50
      
  - name: ecommerce-microservices
    path: /Users/demin_coder/Dev/ecommerce-microservices
    tech_stack: java_spring
    description: "java, spring, jpa로 구현된 MSA 서버 애플리케이션"
    filter_overrides:
      min_changed_lines: 100  # Java는 더 큰 단위 변경
      file_types: [".java", ".kt", ".xml"]
      
  - name: kotlin-realworld
    path: /Users/demin_coder/Dev/kotlin-realworld
    tech_stack: kotlin_jpa
    description: "java, kotlin, jpa로 구현된 호텔 예약 서버 애플리케이션"
    filter_overrides:
      min_changed_lines: 80
      file_types: [".kt", ".java"]

review_models:
  - gemini-2.5-pro
  - claude-sonnet-4
  - claude-sonnet-4-thinking

commit_filters:
  keywords:
    include: [fix, feature, refactor, improve, add, update]
    exclude: [typo, format, style, docs, chore]
  stats:
    min_files: 2
    max_files: 10
    min_lines: 50
  merge_handling:
    fast_forward: exclude
    conflict_resolution: include
    squash_merge: include
    feature_branch: conditional  # 변경량 기준

workflow:
  skip_existing:
    commit_filtering: true  # 필터링된 commit JSON 존재 시 skip
    review_results: true    # 동일 commit-model 조합 결과 존재 시 skip
  parallel_execution:
    max_concurrent_repos: 2
    max_concurrent_models: 3
  cache_enabled: true
```

### 실행 플래그 옵션
```bash
# 기본 실행
selvage-eval

# 설정 파일 지정
selvage-eval --config custom-config.yml

# 특정 저장소만 실행
selvage-eval --repos cline,ecommerce-microservices

# 특정 모델만 실행
selvage-eval --models gemini-2.5-flash

# 강제 재실행 (캐시 무시)
selvage-eval --force-refresh

# 특정 단계만 실행
selvage-eval --steps filter,review
```

### Skip 로직 상세
- **Meaningful Commit 필터링**: 이미 필터링된 commit 목록 JSON이 존재하면 skip
- **Selvage 리뷰**: 동일한 commit-model 조합의 결과가 존재하면 skip  
- **DeepEval 변환**: 동일한 평가 설정의 결과가 존재하면 skip
- **목적**: 동일한 data source로 재현 가능한 테스트 환경 제공 

## 4단계 에이전트 워크플로우

### 1단계: Meaningful Commit 수집 및 필터링

#### 목표
평가 가치가 있는 의미있는 커밋들을 자동으로 식별하고 JSON으로 저장

#### 구현 명세
```python
# Git 명령어 패턴
commands = [
    "git log --oneline --since='30 days ago'",
    "git log --grep='fix|feature|refactor' --oneline",
    "git show --stat <commit_id>"
]
```

#### 필터링 기준
1. **커밋 메시지 키워드**:
   - 포함: `fix`, `feature`, `refactor`, `improve`, `add`, `update`
   - 제외: `typo`, `format`, `style`, `docs`, `chore`

2. **Merge 커밋 특별 처리**:
   - Fast-forward merge (변경사항 없음): 제외
   - Conflict resolution merge: 포함 (실제 코드 변경 발생)
   - Squash merge: 포함 (압축된 의미있는 변경)
   - Feature branch merge: 조건부 포함 (변경량 기준)

3. **변경 통계 기준**:
   - 최소 변경 파일 수: 2-10개
   - 최소 변경 라인 수: 50+ 라인
   - 파일 타입 필터: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.rs`

4. **파일명 패턴 필터링**:
   - 포함: 소스코드 파일
   - 제외: 설정 파일, 문서, 테스트 전용 파일

#### 데이터 스키마
```json
{
  "commits": [
    {
      "id": "abc123",
      "message": "fix: resolve memory leak in parser",
      "author": "developer@example.com",
      "date": "2024-01-15T10:30:00Z",
      "stats": {
        "files_changed": 3,
        "lines_added": 45,
        "lines_deleted": 12
      },
      "files": ["src/parser.py", "src/utils.py", "tests/test_parser.py"]
    }
  ],
  "metadata": {
    "total_commits": 150,
    "filtered_commits": 25,
    "filter_timestamp": "2024-01-20T15:00:00Z",
  }
}
```

### 2단계: Selvage 리뷰 실행

#### 목표
선별된 커밋들에 대해 다중 모델로 Selvage 리뷰 실행 및 결과 수집

#### 구체적인 명령어 패턴
```python
# 기본 리뷰 실행
subprocess.run([
    "/Users/demin_coder/.local/bin/selvage",
    "review",
    "--target-commit", commit_id,
    "--model", model_name,
    "--output-format", "json",
    "--include-context", "true",
    "--max-files", "10"
], capture_output=True, text=True, timeout=300)

# 성능 메트릭 수집을 위한 추가 옵션
subprocess.run([
    "/Users/demin_coder/.local/bin/selvage",
    "review",
    "--target-commit", commit_id,
    "--model", model_name,
    "--metrics", "true",
    "--timing", "true",
    "--token-usage", "true"
], capture_output=True, text=True)
```

#### 다중 모델 병렬 처리
```python
# 설정 파일에서 review_models 로딩
review_models = config.review_models  # ["gemini-2.5-pro", "claude-sonnet-4", "claude-sonnet-4-thinking"]
agent_model = config.agent_model      # "gemini-2.5-flash"

# 리뷰 모델별 병렬 실행
tasks = [
    run_selvage_review(commit_id, model) 
    for model in review_models
]
results = await asyncio.gather(*tasks)
```

#### 결과 저장 구조
```
results/
├── {commit_id}/
│   ├── gemini-2.5-flash/
│   │   ├── review.json
│   │   ├── metrics.json
│   │   └── timing.json
│   ├── claude-sonnet-4/
│   │   └── ...
│   └── metadata.json
```

### 3단계: DeepEval 변환 및 평가

#### 목표
Selvage 결과를 DeepEval 형식으로 변환하고 정량적 평가 실행

#### 변환 스키마
```python
# DeepEval 테스트 케이스 형식
test_case = {
    "input": {
        "commit_id": commit_id,
        "diff_content": diff_text,
        "model": model_name
    },
    "actual_output": selvage_review_result,
    "expected_output": None,  # 참조 기준이 없는 경우
    "context": {
        "performance_metrics": {
            "response_time": 2.5,
            "token_usage": 1500,
            "cost": 0.03
        }
    }
}
```

#### 평가 메트릭 구현
```python
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)

# 커스텀 메트릭 정의
class CodeReviewQualityMetric(BaseMetric):
    def measure(self, test_case):
        # 코드 리뷰 품질 측정 로직
        pass

class CostEfficiencyMetric(BaseMetric):
    def measure(self, test_case):
        # 비용 효율성 측정 로직
        pass
```

### 4단계: 결과 분석 및 비교

#### 목표
모델별 성능 비교 및 프롬프트 버전 효과성 분석

#### 분석 차원
1. **모델별 성능 비교**:
   - 리뷰 품질 점수
   - 응답 시간
   - 토큰 사용량
   - 비용 효율성

2. **프롬프트 버전 비교**:
   - A/B 테스트 방식
   - 동일 커밋에 대한 다른 프롬프트 적용
   - 개선 효과 측정

#### 분석 결과 형식
```json
{
  "summary": {
    "total_commits_evaluated": 25,
    "models_compared": 3,
    "evaluation_date": "2024-01-20"
  },
  "model_comparison": {
    "gemini-2.5-flash": {
      "avg_quality_score": 7.2,
      "avg_response_time": 2.1,
      "avg_cost": 0.025,
      "success_rate": 0.96
    }
  },
  "prompt_comparison": {
    "version_1": { "avg_score": 7.0 },
    "version_2": { "avg_score": 7.8 }
  }
}
```

## 환경 설정

### 필수 API 키
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key" 
export GEMINI_API_KEY="your-key"
```

### Selvage 통합 설정
- **바이너리 위치**: `/Users/demin_coder/.local/bin/selvage` (v0.1.2)
- **소스 코드**: `/Users/demin_coder/Dev/selvage`
- **통신 방식**: subprocess만 사용 (직접 API 호출 금지)

## 성능 최적화 전략

### 병렬 처리 설계
```python
# 커밋별 병렬 처리
async def process_commits_parallel(commits, models):
    semaphore = asyncio.Semaphore(5)  # 동시 실행 제한
    tasks = [
        process_single_commit(commit, models, semaphore)
        for commit in commits
    ]
    return await asyncio.gather(*tasks)
```

### 캐싱 전략
- **Git 데이터**: 커밋 정보 및 diff 내용 캐싱
- **Selvage 결과**: 동일 커밋/모델 조합 결과 재사용
- **DeepEval 메트릭**: 계산 결과 캐싱

### 성능 측정 지점
1. **Git 작업**: diff 추출, 통계 수집 시간
2. **Selvage 실행**: 프로세스 시작부터 완료까지
3. **API 호출**: 모델별 응답 시간 및 토큰 사용량
4. **데이터 변환**: JSON 파싱 및 변환 시간
5. **평가 실행**: DeepEval 메트릭 계산 시간

## 에러 처리 및 복구 전략

### 계층적 에러 처리
```python
class SelvageEvaluationError(Exception):
    """Selvage 평가 관련 기본 예외"""
    pass

class GitOperationError(SelvageEvaluationError):
    """Git 작업 실패"""
    pass

class SelvageExecutionError(SelvageEvaluationError):
    """Selvage 실행 실패"""
    pass

class ModelAPIError(SelvageEvaluationError):
    """모델 API 호출 실패"""
    pass
```

### 재시도 로직
```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def run_selvage_with_retry(commit_id: str, model: str) -> dict:
    """재시도 로직이 포함된 Selvage 실행"""
    try:
        return await run_selvage_review(commit_id, model)
    except subprocess.TimeoutExpired:
        raise SelvageExecutionError(f"Timeout for {commit_id} with {model}")
    except subprocess.CalledProcessError as e:
        raise SelvageExecutionError(f"Process failed: {e.stderr}")
```

### 실패 복구 전략
1. **부분 실패 허용**: 일부 커밋/모델 조합 실패 시 계속 진행
2. **체크포인트 저장**: 단계별 중간 결과 저장
3. **재개 가능**: 실패 지점부터 재시작 가능

## 데이터 관리 및 저장

### 디렉토리 구조
```
selvage-eval-results/
├── commits/
│   └── filtered_commits.json
├── reviews/
│   ├── {commit_id}/
│   │   ├── {model_name}/
│   │   │   ├── review.json
│   │   │   ├── metrics.json
│   │   │   └── timing.json
│   │   └── metadata.json
├── evaluations/
│   ├── deepeval_results.json
│   └── comparative_analysis.json
└── logs/
    ├── agent.log
    └── error.log
```

### 메타데이터 관리 (자동 생성)
```json
{
  "evaluation_session": {
    "id": "eval_20240620_143022_a1b2c3d",  // 자동 생성: 날짜_시간_git_hash
    "start_time": "2024-06-20T14:30:22Z",
    "end_time": "2024-06-20T16:45:30Z",
    "configuration": {
      "agent_model": "gemini-2.5-flash",
      "review_models": ["gemini-2.5-pro", "claude-sonnet-4", "claude-sonnet-4-thinking"],
      "target_repositories": [
        {"name": "cline", "path": "/Users/demin_coder/Dev/cline"},
        {"name": "ecommerce-microservices", "path": "/Users/demin_coder/Dev/ecommerce-microservices"}
      ],
      "commit_filter_criteria": {...},
      "evaluation_metrics": [...]
    },
    "results_summary": {
      "total_commits_per_repo": {
        "cline": 15,
        "ecommerce-microservices": 10
      },
      "successful_evaluations": 72,  // 25 commits × 3 models - 3 failures
      "failed_evaluations": 3,
      "repository_breakdown": {
        "cline": {"commits": 15, "success_rate": 0.96},
        "ecommerce-microservices": {"commits": 10, "success_rate": 0.94}
      }
    }
  }
}
```

## 평가 지표 및 분석

### 정량적 메트릭
1. **리뷰 품질**:
   - 이슈 탐지 정확도
   - 제안 사항의 실용성
   - 코드 이해도

2. **성능 메트릭**:
   - 평균 응답 시간
   - 토큰 사용 효율성
   - API 호출 비용

3. **신뢰성 메트릭**:
   - 성공률
   - 에러 빈도
   - 일관성

### 비교 분석 방법
```python
def compare_models(results: Dict[str, List[EvaluationResult]]) -> ComparisonReport:
    """모델별 성능 비교 분석
    
    Args:
        results: 모델별 평가 결과
        
    Returns:
        비교 분석 보고서
    """
    report = ComparisonReport()
    
    for model, evals in results.items():
        metrics = calculate_aggregate_metrics(evals)
        report.add_model_metrics(model, metrics)
    
    report.generate_statistical_comparison()
    return report
```

## 구현 체크리스트

### Phase 1: 기본 인프라
- [ ] YAML 설정 파일 파서 구현
- [ ] CLI 인터페이스 및 플래그 처리
- [ ] Git 작업 모듈 구현
- [ ] Selvage 실행 래퍼 구현
- [ ] JSON 데이터 스키마 정의
- [ ] 기본 에러 처리 구현
- [ ] 자동 session ID 생성 로직

### Phase 2: 코어 워크플로우
- [ ] agent-model과 review-models 분리 처리
- [ ] 다중 저장소 순회 로직
- [ ] 커밋 필터링 로직 구현 (저장소별 override 지원)
- [ ] 다중 모델 병렬 실행
- [ ] 결과 저장 시스템
- [ ] Skip 로직 구현 (기존 결과 재사용)
- [ ] 재시도 및 복구 로직

### Phase 3: 평가 시스템
- [ ] DeepEval 변환 모듈
- [ ] 커스텀 메트릭 구현
- [ ] 저장소별/기술스택별 분석 도구
- [ ] 모델 간 비교 분석 도구
- [ ] 종합 보고서 생성 기능

### Phase 4: 최적화 및 확장
- [ ] 성능 프로파일링
- [ ] 캐싱 시스템 구현
- [ ] 저장소별 로그 분리
- [ ] 보안 제약 사항 처리 (selvage-deprecated readonly)
- [ ] 웹 대시보드 (선택사항)
- [ ] CI/CD 통합 (선택사항)