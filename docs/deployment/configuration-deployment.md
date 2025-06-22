# 설정 및 배포

## 사용 모델 전략
- **Primary**: `gemini-2.5-pro` (속도/비용 최적화)

## 대상 repo-path
- **cline**
    - path: /Users/demin_coder/Dev/cline
    - description: typescript로 구현된 coding assistant
- **selvage-deprecated**
    - path: /Users/demin_coder/Dev/selvage-deprecated
    - description: selvage가 정식 배포되기 전 commit history를 가지고 있는 repository (주의: 현재 selvage의 이전 작업 폴더이므로 review 대상으로서만 접근할 것)
- **ecommerce-microservices**
    - path: /Users/demin_coder/Dev/ecommerce-microservices
    - description: java, spring, jpa로 구현된 MSA 서버 애플리케이션
- **kotlin-realworld**
    - path: /Users/demin_coder/Dev/kotlin-realworld
    - description: java, kotlin, jpa로 구현된 호텔 예약 서버 애플리케이션

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
commits_per_repo: 5

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

## Docker 배포 (선택사항)

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    git \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# Selvage 바이너리 복사 (빌드 시점에서)
COPY selvage /usr/local/bin/

# 실행 권한 부여
RUN chmod +x /usr/local/bin/selvage

# 환경 변수 설정
ENV PYTHONPATH=/app

# 기본 명령어
CMD ["python", "-m", "selvage_eval_agent"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  selvage-eval:
    build: .
    volumes:
      - ./selvage-eval-results:/app/selvage-eval-results
      - ./config:/app/config:ro
      - ~/.gitconfig:/root/.gitconfig:ro
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    command: ["selvage-eval", "--config", "/app/config/selvage-eval-config.yml"]
```

## 보안 고려사항

### 환경 변수 관리
```bash
# .env 파일 사용 (git에서 제외)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# 또는 시스템 키링 사용
keyring set selvage-eval openai_api_key
keyring set selvage-eval anthropic_api_key
keyring set selvage-eval gemini_api_key
```

### 접근 권한 제한
- 읽기 전용 저장소 접근
- 결과 디렉토리 내에서만 쓰기 허용
- 시스템 명령어 제한
- 네트워크 접근 제한