# Selvage 평가 에이전트 - 커밋 수집 단계 구현 명세

## 4단계 에이전트 워크플로우

### 1단계: Meaningful Commit 수집 및 필터링

#### 목표
평가 가치가 있는 의미있는 커밋들을 자동으로 식별하고 JSON으로 저장

#### 구현 명세

#### 커밋 수집 시작점
```python
# src/selvage_eval/cli.py
if args.commit_collection:
    commit_collector = CommitCollector(config)
    commit_collector.collect_commits()
    return
```

#### config file parsing to get target repositories
```python
config = load_config(config_path)

# 설정 파일에서 target_repositories 로딩
target_repositories = config.target_repositories

# 저장소별 커밋 수집
for repo in target_repositories:
    repo_name = repo.name
    repo_path = repo.path
    repo_filter_overrides = repo.filter_overrides or {}
```

#### CommitCollector 클래스 인터페이스
```python
from typing import Dict, Any, List
from selvage_eval.tools.tool_executor import ToolExecutor

class CommitCollector:
    """의미있는 커밋 수집 및 필터링 클래스"""
    
    def __init__(self, config: EvaluationConfig, tool_executor: ToolExecutor):
        """
        Args:
            config: 설정 정보 (target_repositories, commit_filters 포함)
            tool_executor: ExecuteSafeCommandTool을 포함한 도구 실행기
        """
        
    def collect_commits(self) -> MeaningfulCommitsData:
        """
        모든 대상 저장소에서 의미있는 커밋을 수집하고 필터링
        
        Returns:
          MeaningfulCommitsData: 전체 수집 결과
        """
        
    def _collect_repo_commits(self, repo_config: TargetRepository) -> List[CommitData]:
        """단일 저장소에서 커밋 수집"""
        
    def _filter_commits(self, commits: List[CommitData], filters: CommitFilters) -> List[CommitData]:
        """키워드 및 통계 기준으로 커밋 필터링"""
        
    def _score_commits(self, commit: CommitData) -> CommitData:
        """커밋 배점 계산 (0-100점)"""
        
    def _select_top_commits(self, commits: List[CommitData], count: int) -> List[CommitData]:
        """점수 기준 상위 커밋 선별"""

# Git 명령어 실행은 ExecuteSafeCommandTool 활용
# 예시: tool_executor.execute_tool("execute_safe_command", {
#     "command": "git log --grep='fix|feature|refactor' --oneline",
#     "cwd": repo_path
# })
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

3. **변경 통계 기준** (selvage-eval-config.yml의 commit_filters.stats 기반):
   - **파일 수 범위**: 2-10개 (min_files: 2, max_files: 10)
     - 최소 2개: 단일 파일 변경은 너무 제한적
     - 최대 10개: 과도한 변경은 리뷰 복잡도 증가로 평가 어려움
   - **변경 라인 수**: 최소 50라인 (min_lines: 50)
     - trivial한 변경 제외하고 의미있는 코드 변경만 선별
     - 단순 포맷팅, 오타 수정 등 배제
   - **파일 타입 필터 없음**: 
     - 모든 파일 포함
     - 특정 파일 타입에 대한 가중치는 배점 단계에서 처리
   - **추가 필터링 조건**:
     - 머지 커밋 중 fast-forward는 제외 (실제 변경사항 없음)
     - 충돌 해결, 스쿼시 머지는 포함 (의미있는 변경)
     - 키워드 기반 필터링: include(fix, feature, refactor, improve, add, update) / exclude(typo, format, style, docs, chore)


5. **커밋 배점 기준** (git 명령어 기반 측정, 총 100점):

   **A. 파일 타입 감점 조정 (기본 100점에서 감점)**
   ```bash
   # git show --name-only <commit-hash> | 비-코드 파일 감점 처리
   ```
   - **모든 파일을 코드로 간주** (jsx, tsx, vue, svelte 등 자동 포함)
   - **비-코드 파일 감점**: 각 -5점
     - 문서: `.txt`, `.rst`, `.adoc`, `.doc`, `.docx`, `.pdf`
     - 이미지/미디어: `.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`, `.ico`, `.mp4`, `.mp3`, `.wav`
     - 압축/바이너리: `.zip`, `.tar`, `.gz`, `.exe`, `.dll`, `.so`, `.dylib`, `.bin`
     - 자동생성: `package-lock.json`, `*.lock`, `*.cache`
   - **경미한 감점 파일**: 각 -2점
     - 설정: `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.env`, `md`, `mdc` (복잡한 로직 가능)
     - 빌드: `Dockerfile`, `Makefile`, `requirements.txt` (스크립트 로직 포함)
   - **예시**: 소스코드 3개 + README.md 1개 = 100점 - 2점 = 98점

   **B. 변경 규모 적정성 (25점)**
   ```bash
   # git show --stat <commit-hash> | 변경 통계 분석
   ```
   - **파일 수 적정성**: 10점
     - 2-4개 파일: +10점 (리뷰 집중도 최적)
     - 5-7개 파일: +7점 (적정 범위)
     - 8-10개 파일: +4점 (복잡도 증가)
   - **변경 라인 수 밸런스**: 15점
     - 50-200라인: +15점 (의미있는 변경)
     - 201-400라인: +10점 (중규모 변경)  
     - 401-600라인: +5점 (대규모, 리뷰 어려움)
     - 추가/삭제 비율이 극단적인 경우 -5점 (단순 삭제나 복붙)

   **C. 커밋 특성 (25점)**
   ```bash
   # git log --oneline <commit-hash> | 메시지 키워드 분석
   # git show --name-only <commit-hash> | 경로 패턴 분석
   ```
   - **긍정 키워드**: 각 +5점 (최대 15점)
     - `fix`, `refactor`, `improve`, `optimize`, `enhance`
     - `feature`, `implement`, `add`, `update`
     - `security`, `performance`, `bug`
   - **부정 키워드**: 각 -3점
     - `typo`, `format`, `style`, `whitespace`, `lint`
     - `merge`, `revert`, `backup`
   - **경로 패턴 가중치**: 10점
     - 핵심 로직 경로 (`src/`, `lib/`, `core/`): +10점
     - 유틸리티 (`utils/`, `helpers/`): +7점
     - 설정/빌드 (`config/`, `build/`, `.github/`): -5점

   **D. 시간 가중치 (20점)**
   ```bash
   # git log --date=short <commit-hash> | 날짜 확인
   ```
   - **커밋 시기 신선도**:
     - 최근 1개월: +20점 (최신 코드 스타일)
     - 최근 3개월: +15점
     - 최근 6개월: +10점
     - 최근 1년: +5점
     - 그 이상: +2점

   **E. 추가 조정 사항**
   - **머지 커밋 처리** (`git show --merges`):
     - Fast-forward 머지: -10점 (변경사항 없음)
     - 충돌 해결 머지: +5점 (복잡한 변경)
     - 스쿼시 머지: +3점 (정리된 변경)
   - **작성자 다양성**: 동일 작성자 연속 커밋 시 각 -2점

   **최종 점수**: A + B + C + D + 조정사항 (0-100점으로 정규화)

6. ** 배점 계산 후 commits_per_repo 만큼 커밋을 선별하여 저장**


#### 데이터 스키마
파일명 : meaningful_commits.json
```json
{ "repositories": [
    {
        "repo_name": "cline",
        "repo_path": "/Users/demin_coder/Dev/cline",
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
            "total_score": 85,
            "score_details": {
                "file_type_penalty": -2,
                "line_count_penalty": -5,
                "keyword_penalty": -3,
                "path_penalty": 10,
                "time_penalty": 10,
            }
        ],
        "metadata": {
            "total_commits": 150,
            "filtered_commits": 25,
            "filter_timestamp": "2024-01-20T15:00:00Z",
        }
        },
        // ...(중략)
    ]
}
```

#### 에러 처리 및 복원력 요구사항

1. **Git 명령어 실행 실패**
   - 개별 명령어 실패 시 로그 기록 후 계속 진행
   - 저장소 접근 불가 시 해당 저장소 건너뛰기
   - 부분 실패 시에도 수집 가능한 데이터는 저장

2. **데이터 검증 및 복구**
   - 잘못된 커밋 데이터 발견 시 기본값 적용
   - JSON 스키마 검증 후 저장
   - 중간 결과 임시 저장으로 재시작 지원

3. **진행 상황 및 로깅**
   - 저장소별 처리 진행 상황 표시
   - 에러 및 경고 레벨 로그 구분
   - 성능 메트릭 수집 (처리 시간, 커밋 수 등)

## 구현 체크리스트

### Phase 1: 기본 인프라
1. CommitCollector 클래스 구현 (인터페이스 준수)
2. ExecuteSafeCommandTool 연동
3. 에러 처리 및 로깅 시스템