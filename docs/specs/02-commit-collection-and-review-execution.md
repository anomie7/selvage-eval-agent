# Selvage 평가 에이전트 - 구현 가이드 (1-2단계)

## 4단계 에이전트 워크플로우

### 1단계: Meaningful Commit 수집 및 필터링

#### 목표
평가 가치가 있는 의미있는 커밋들을 자동으로 식별하고 JSON으로 저장

#### 구현 명세
```python
# 파일 읽기
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# 설정 파일에서 target_repositories 로딩
target_repositories = config.target_repositories

# 저장소별 커밋 수집
for repo in target_repositories:
    repo_name = repo['name']
    repo_path = repo['path']
    repo_filter_overrides = repo.get('filter_overrides', {})

```

```python
# Git 명령어 패턴
commands = [
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
            "files": ["src/parser.py", "src/utils.py", "tests/test_parser.py"]
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

### 2단계: Selvage 리뷰 실행

#### 목표
선별된 커밋들에 대해 다중 모델로 Selvage 리뷰 실행 및 결과 수집

#### 구체적인 명령어 패턴
```python
# json 파일 읽기
with open('meaningful_commits.json', 'r') as f:
    meaningful_commits = json.load(f)

# repo_name 별 커밋 목록 순회   
for repo in meaningful_commits['repositories']:
    repo_path = repo['repo_path']
    repo_name = repo['repo_name']
    for commit in repo['commits']:
        TODO("문서 아래에 명시된 프로세스를 구현할 것")
```

```python
# commit checkout
checkout_result = subprocess.run([
    "git", "checkout", commit_id
], capture_output=True, text=True, cwd=repo_path)

# get parent commit id
parent_commit_id = subprocess.run([
    "git", "rev-parse", "HEAD^"
], capture_output=True, text=True, cwd=repo_path)

# 리뷰 실행
subprocess.run([
    "selvage",
    "review",
    "--target-commit", parent_commit_id,
    "--model", model_name,
    "--log-dir", f"~/Library/selvage-eval-agent/review_logs/{repo_name}/{commit_id}/{model_name}" # 리뷰 결과를 eval-agent 폴더 내에 저장
], capture_output=True, text=True, timeout=300, cwd=repo_path)

# checkout HEAD로 초기화 (커밋 체크아웃 후 초기화)
subprocess.run([
    "git", "checkout", "main"
], capture_output=True, text=True, cwd=repo_path)
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

#### 리뷰 결과(review_log) 저장 구조
```
~/Library/selvage-eval-agent/review_logs/
├── {repo_name}/
│   ├── {commit_id}/
│   │   ├── {model_name}/
│   │   │   └── {%Y%m%d_%H%M%S}_{model_name}_review_log.json
│   │   └── metadata.json
│   └── ...
```

**파일명 형식**: `{%Y%m%d_%H%M%S}_{model_name}_review_log.json`  
**예시**: `20250621_171058_gemini-2.5-flash-preview-05-20_review_log.json`

#### 리뷰 결과(review_log) 파일 구조
```json
{
  "id": "google-gemini-2.5-flash-preview-05-20-1750493437",
  "model": {
    "provider": "google",
    "name": "gemini-2.5-flash-preview-05-20"
  },
  "created_at": "2025-06-21T17:10:58.947588",
  "prompt": [
    {
      "role": "system", 
      "content": "You are \"Selvage\", the world's best AI code reviewer..."
    },
    {
      "role": "user",
      "content": "{\"file_name\": \"selvage/cli.py\", \"file_content\": \"...\", \"formatted_hunks\": [...]}"
    }
  ],
  "review_request": {
    "diff_content": "diff --git a/selvage/cli.py b/selvage/cli.py\nindex 1234567..abcdefg 100644\n--- a/selvage/cli.py\n+++ b/selvage/cli.py\n@@ -270,14 +270,24 @@ def save_review_log(\n...",
    "processed_diff": {
      "files": [
        {
          "filename": "selvage/cli.py",
          "file_content": "전체 파일 내용...",
          "language": "python",
          "additions": 28,
          "deletions": 2,
          "hunks": [
            {
              "header": "@@ -270,14 +270,24 @@ def save_review_log(",
              "content": "hunk의 diff 내용...",
              "before_code": "변경 전 코드",
              "after_code": "변경 후 코드",
              "start_line_original": 270,
              "line_count_original": 14,
              "start_line_modified": 270,
              "line_count_modified": 24
            }
          ]
        },
        {
          "filename": "selvage/src/utils/token/models.py",
          "file_content": "전체 파일 내용...",
          "language": "python", 
          "additions": 1,
          "deletions": 1,
          "hunks": [...]
        }
      ]
    },
    "file_paths": ["selvage/cli.py", "selvage/src/utils/token/models.py"],
    "use_full_context": true,
    "model": "gemini-2.5-flash-preview-05-20",
    "repo_path": "/Users/demin_coder/Dev/selvage"
  },
  "review_response": {
    "issues": [
      {
        "type": "style",
        "line_number": 274,
        "file": "selvage/cli.py", 
        "description": "save_review_log 함수의 repo_path 파라미터는 review_request.repo_path를 통해 동일한 정보가 전달되므로 중복됩니다.",
        "suggestion": "save_review_log 함수 시그니처에서 repo_path 파라미터를 제거하고, 함수 내부에서 review_request.repo_path를 사용하도록 유지합니다.",
        "severity": "info",
        "target_code": "repo_path: str = \".\",",
        "suggested_code": "# 코드 라인 제거\\n# repo_path: str = \".\""
      }
    ],
    "summary": "리뷰 로그 저장 디렉토리를 사용자 정의할 수 있는 --log-dir 옵션이 review CLI 명령어에 추가되었습니다.",
    "score": 9.0,
    "recommendations": [
      "새로운 --log-dir 옵션과 save_review_log 함수의 통합은 사용자에게 더 많은 유연성을 제공합니다."
    ]
  },
  "status": "SUCCESS",
  "error": null,
  "prompt_version": "v3",
  "repo_path": "/Users/demin_coder/Dev/selvage"
}
```

**주요 필드 설명:**
- `id`: 고유 로그 식별자 (`{provider}-{model_name}-{timestamp}`)
- `model`: 사용된 모델 정보 (provider, name)  
- `created_at`: 로그 생성 시각 (ISO 형식)
- `prompt`: LLM에 전송된 프롬프트 메시지 배열
- `review_request`: 리뷰 요청 데이터
  - `diff_content`: 원본 git diff 텍스트
  - `processed_diff`: 파싱된 diff 데이터
    - `files[]`: 변경된 파일 목록
      - `filename`: 파일 경로
      - `file_content`: 전체 파일 내용
      - `language`: 프로그래밍 언어
      - `additions/deletions`: 추가/삭제된 라인 수
      - `hunks[]`: 변경 블록 배열
        - `header`: diff 헤더 (`@@ -270,14 +270,24 @@`)
        - `before_code/after_code`: 변경 전후 코드
        - `start_line_*`: 시작 라인 번호
        - `line_count_*`: 변경 라인 수
  - `file_paths`: 변경된 파일 경로 배열
  - `use_full_context`: 전체 파일 컨텍스트 사용 여부
  - `model`: 사용 모델명
  - `repo_path`: 저장소 경로
- `review_response`: LLM 리뷰 결과 (issues, summary, score, recommendations)
- `status`: 리뷰 실행 상태 (`SUCCESS`, `FAILED`)
- `error`: 오류 발생 시 오류 메시지 (성공 시 null)
- `prompt_version`: 사용된 프롬프트 버전
- `repo_path`: 리뷰 대상 저장소 경로

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