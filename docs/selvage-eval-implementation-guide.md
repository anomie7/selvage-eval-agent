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
- **Primary**: `gemini-2.5-pro` (속도/비용 최적화)

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

### 3단계: DeepEval 변환 및 평가

#### 목표
Selvage 결과를 DeepEval 형식으로 변환하고 평가

#### 변환 스키마
# DeepEval Test case 구조
**주요 필드 설명:**
-  input: review_log의 prompt와 대응됨
-  actual_output: review_log의 review_response와 대응됨
-  expected_output: None(현재 사용하지 않음)

```python
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


def scan_review_logs(base_path: str = "~/Library/selvage-eval-agent/review_logs") -> List[Dict[str, Any]]:
    """
    review_logs 디렉토리 구조를 탐색하여 모든 리뷰 로그 파일을 찾는다.
    
    Args:
        base_path: 리뷰 로그 기본 경로
    
    Returns:
        리뷰 로그 파일 정보 리스트 (repo_name, commit_id, model_name, file_path 포함)
    """
    review_logs = []
    base_path = Path(base_path).expanduser()
    
    if not base_path.exists():
        print(f"경로가 존재하지 않습니다: {base_path}")
        return review_logs
    
    # repo_name 폴더 순회
    for repo_dir in base_path.iterdir():
        if not repo_dir.is_dir():
            continue
            
        repo_name = repo_dir.name
        
        # commit_id 폴더 순회
        for commit_dir in repo_dir.iterdir():
            if not commit_dir.is_dir():
                continue
                
            commit_id = commit_dir.name
            
            # model_name 폴더 순회
            for model_dir in commit_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                model_name = model_dir.name
                
                # 리뷰 로그 파일 찾기 (.json 파일)
                for log_file in model_dir.glob("*.json"):
                    review_logs.append({
                        "repo_name": repo_name,
                        "commit_id": commit_id,
                        "model_name": model_name,
                        "file_path": log_file,
                        "file_name": log_file.name
                    })
    
    return review_logs


def extract_prompt_and_response(log_file_path: Path) -> Optional[Dict[str, Any]]:
    """
    리뷰 로그 파일에서 prompt와 review_response를 추출한다.
    
    Args:
        log_file_path: 리뷰 로그 파일 경로
    
    Returns:
        추출된 prompt와 review_response 데이터
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        # prompt와 review_response 필드 추출
        prompt = log_data.get("prompt", [])
        review_response = log_data.get("review_response", {})
        
        if not prompt or not review_response:
            print(f"필수 필드가 없습니다: {log_file_path}")
            return None
        
        return {
            "prompt": prompt,
            "review_response": review_response,
            "original_data": log_data  # 원본 데이터도 포함
        }
        
    except Exception as e:
        print(f"파일 읽기 오류 {log_file_path}: {e}")
        return None


def convert_to_deepeval_format(repo_name: str, model_name: str, 
                             extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    추출된 데이터를 DeepEval 테스트 케이스 형식으로 변환한다.
    
    Args:
        repo_name: 리포지토리 이름
        model_name: 모델 이름
        extracted_data: 추출된 prompt와 response 데이터
    
    Returns:
        DeepEval 형식의 테스트 케이스
    """
    return {
        "input": json.dumps(extracted_data["prompt"]),
        "actual_output": json.dumps(extracted_data["review_response"])
    }


def save_deepeval_test_cases(repo_name: str, model_name: str, 
                           test_cases: List[Dict[str, Any]], 
                           output_dir: str = "~/Library/selvage-eval-agent/deep_eval_test_case") -> str:
    """
    DeepEval 테스트 케이스를 파일로 저장한다.
    
    Args:
        repo_name: 리포지토리 이름
        model_name: 모델 이름
        test_cases: 테스트 케이스 리스트
        output_dir: 출력 디렉토리
    
    Returns:
        저장된 파일 경로
    """
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 현재 시간을 기반으로 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_data_{timestamp}_{repo_name}_{model_name}.json"
    
    file_path = output_path / filename
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    
    print(f"테스트 케이스 저장 완료: {file_path}")
    return str(file_path)


def process_review_logs_to_deepeval():
    """
    전체 프로세스를 실행하는 메인 함수
    """
    print("리뷰 로그 스캔 시작...")
    review_logs = scan_review_logs()
    
    if not review_logs:
        print("리뷰 로그 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(review_logs)}개의 리뷰 로그 파일을 발견했습니다.")
    
    # repo_name과 model_name별로 그룹화
    grouped_logs = {}
    for log_info in review_logs:
        key = (log_info["repo_name"], log_info["model_name"])
        if key not in grouped_logs:
            grouped_logs[key] = []
        grouped_logs[key].append(log_info)
    
    # 각 그룹별로 테스트 케이스 생성
    for (repo_name, model_name), logs in grouped_logs.items():
        print(f"\n처리 중: {repo_name} - {model_name} ({len(logs)}개 파일)")
        
        test_cases = []
        for log_info in logs:
            extracted_data = extract_prompt_and_response(log_info["file_path"])
            if extracted_data:
                test_case = convert_to_deepeval_format(repo_name, model_name, extracted_data)
                test_cases.append(test_case)
        
        if test_cases:
            saved_file = save_deepeval_test_cases(repo_name, model_name, test_cases)
            print(f"  → {len(test_cases)}개 테스트 케이스 저장: {saved_file}")
        else:
            print(f"  → 유효한 테스트 케이스가 없습니다.")


# 실행 예시
if __name__ == "__main__":
    process_review_logs_to_deepeval()
```

#### 평가 메트릭

DeepEval을 사용한 4개 핵심 메트릭으로 Selvage 리뷰 품질을 정량화합니다.

**1. Correctness (정확성) - 임계값: 0.7**
```python
correctness = GEval(
    name="Correctness",
    model="gemini-2.5-pro-preview-05-06",
    evaluation_steps=[
        "입력 코드에서 발견된 모든 관련 주요 이슈(버그, 보안 취약점, 성능 문제, 중대한 스타일/설계 결함)가 'issues' 배열에 보고되었는지 확인",
        "'issues' 배열이 비어있는 경우, 입력 코드를 비판적으로 평가하여 탐지 실패가 아닌 실제 이슈 부재인지 확인",
        "이슈가 보고된 경우, 파일명과 라인 번호의 정확성 확인",
        "이슈 유형(버그, 보안, 성능, 스타일, 설계)이 해당 코드에 적절한지 평가",
        "심각도 수준(info, warning, error)이 각 이슈의 실제 영향에 따라 적절히 할당되었는지 확인",
        "이슈 설명이 코드 변경의 영향을 정확하고 사실적으로 반영하는지 검토",
        "'issues' 배열이 정당하게 비어있는 경우, 'summary'가 시스템 프롬프트 가이드라인에 따라 적절히 명시하는지 확인"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    threshold=0.7,
)
```

**2. Clarity (명확성) - 임계값: 0.7**
```python
clarity = GEval(
    name="Clarity",
    model="gemini-2.5-pro-preview-05-06",
    evaluation_steps=[
        "전체 코드 리뷰 출력(요약, 이슈 설명, 제안, 권장사항)이 간결하고 직접적인 언어를 사용하는지 평가",
        "이슈 설명과 제안, 권장사항이 구체적이고 명확한지 평가",
        "코드 변경의 목적과 의도가 명확하게 이해 가능한지 검토",
        "개선된 코드 예시가 제공되고 이해하기 쉬운지 확인"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)
```

**3. Actionability (실행가능성) - 임계값: 0.7**
```python
actionability = GEval(
    name="Actionability",
    model="gemini-2.5-pro-preview-05-06",
    evaluation_steps=[
        "각 이슈에 대해 구체적인 해결책이 제시되었는지 확인",
        "제안된 개선 사항이 실제로 구현 가능한지 평가",
        "코드 개선 예시가 실제 코드베이스에 통합될 수 있을 만큼 구체적인지 검토",
        "제안이 코드 품질, 성능, 보안 등의 측면에서 실질적인 개선을 가져올 수 있는지 평가",
        "전반적인 권장사항이 프로젝트 맥락에서 실행 가능한지 확인"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    threshold=0.7,
)
```

**4. JsonCorrectnessMetric (JSON 스키마 검증)**
```python
jsoncorrectness = JsonCorrectnessMetric(
    expected_schema=StructuredReviewResponse(
        issues=[
            StructuredReviewIssue(
                type="",           # 이슈 유형 (버그, 보안, 성능, 스타일, 설계)
                line_number=0,     # 라인 번호
                file="",           # 파일 경로
                description="",    # 이슈 설명
                suggestion="",     # 개선 제안
                severity=IssueSeverityEnum.INFO,  # 심각도 (info, warning, error)
                target_code="",    # 문제가 있는 원본 코드
                suggested_code="", # 개선된 코드
            )
        ],
        summary="",           # 전체 리뷰 요약
        score=0,             # 0-10 품질 점수
        recommendations=[],   # 전반적인 권장사항 목록
    ),
    model="gemini-2.5-pro-preview-05-06",
    include_reason=True,
)
```

**평가 실행 코드**
```python
@pytest.mark.parametrize("test_case", dataset)
def test_code_review_evaluation(test_case: LLMTestCase):
    """코드 리뷰 평가 테스트."""
    assert_test(
        test_case,
        metrics=[correctness, clarity, actionability, jsoncorrectness],
    )
```

**메트릭 점수 해석**
- **0.7 이상**: 통과 (양질의 리뷰)
- **0.5-0.7**: 보통 (개선 필요)
- **0.5 미만**: 실패 (심각한 문제)

**결과 분석을 위한 프롬프트**
DeepEval 결과의 영어 실패 사유를 한국어로 번역하고 가독성을 개선하기 위해 다음 프롬프트를 사용:

```markdown
# ROLE
당신은 문제 해결 중심의 정확하고 충실한 테크니컬 라이터입니다.

# PROBLEM
deepeval의 metric을 통해 평가한 결과에서 fail reason이 영어로 적혀있어 평가가 어렵습니다.

## INSTRUCTIONS
1. 각 testCase의 metricsData만 추출
2. metricsData.reason들을 한국어로 번역
3. reason 의견을 토대로 input(프롬프트), actualOutput에서 문제 부분만 첨부
4. reason 결과를 종합한 의견 첨부

가독성을 고려해서 편집해서 반환해주세요.
```

### 4단계: 결과 분석 및 비교

#### 목표
DeepEval 평가 결과를 기반으로 모델별 성능을 정량적으로 분석하고, 실제 의사결정을 위한 actionable insights를 자동 도출

#### 4.1 DeepEval 결과 수집 및 전처리

**결과 파일 구조 스캔**
```python
def collect_deepeval_results(base_path: str = "~/Library/selvage-eval-agent/deep_eval_test_case") -> Dict[str, Any]:
    """
    DeepEval 평가 결과 파일들을 수집하고 체계적으로 분류한다.
    
    Returns:
        repo_name, model_name별로 분류된 평가 결과 데이터
    """
    results = {}
    base_path = Path(base_path).expanduser()
    
    for result_file in base_path.glob("deepeval_results_*.json"):
        # 파일명에서 메타데이터 추출: deepeval_results_20250622_143022_cline_gemini-2.5-pro.json
        parts = result_file.stem.split("_")
        if len(parts) >= 6:
            timestamp = f"{parts[2]}_{parts[3]}"
            repo_name = parts[4]
            model_name = "_".join(parts[5:])
            
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
                
            key = f"{repo_name}_{model_name}"
            if key not in results:
                results[key] = []
            
            results[key].append({
                "timestamp": timestamp,
                "repo_name": repo_name,
                "model_name": model_name,
                "file_path": result_file,
                "data": result_data
            })
    
    return results
```

**메트릭별 점수 집계 및 통계 처리**
```python
def analyze_metric_scores(evaluation_results: List[Dict]) -> Dict[str, Any]:
    """
    4개 메트릭별 상세 통계 분석을 수행한다.
    
    Returns:
        메트릭별 점수 분포, 실패 패턴, 통계적 특성
    """
    metrics_analysis = {
        "correctness": {"scores": [], "failures": []},
        "clarity": {"scores": [], "failures": []},
        "actionability": {"scores": [], "failures": []},
        "json_correctness": {"scores": [], "failures": []}
    }
    
    for result in evaluation_results:
        test_cases = result["data"].get("testCases", [])
        
        for test_case in test_cases:
            if not test_case.get("success", True):
                # 실패한 케이스 분석
                metrics_data = test_case.get("metricsData", [])
                
                for metric in metrics_data:
                    metric_name = metric.get("name", "").lower()
                    score = metric.get("score", 0)
                    reason = metric.get("reason", "")
                    
                    if metric_name in metrics_analysis:
                        metrics_analysis[metric_name]["scores"].append(score)
                        
                        if not metric.get("success", True):
                            metrics_analysis[metric_name]["failures"].append({
                                "score": score,
                                "reason": reason,
                                "test_case_id": test_case.get("id", ""),
                                "repo_name": result["repo_name"],
                                "model_name": result["model_name"]
                            })
    
    # 통계 계산
    for metric_name, data in metrics_analysis.items():
        scores = data["scores"]
        if scores:
            data["statistics"] = {
                "count": len(scores),
                "mean": np.mean(scores),
                "median": np.median(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "q25": np.percentile(scores, 25),
                "q75": np.percentile(scores, 75),
                "pass_rate": len([s for s in scores if s >= 0.7]) / len(scores)
            }
    
    return metrics_analysis
```

#### 4.2 모델별 성능 분석

**종합 성능 매트릭스 생성**
```python
def generate_model_performance_matrix(results: Dict[str, Any]) -> pd.DataFrame:
    """
    모델별 성능을 다차원적으로 비교하는 매트릭스를 생성한다.
    
    Returns:
        모델별 성능 지표가 포함된 DataFrame
    """
    performance_data = []
    
    for key, model_results in results.items():
        repo_name, model_name = key.split("_", 1)
        
        # 각 모델별 메트릭 집계
        all_scores = {"correctness": [], "clarity": [], "actionability": [], "json_correctness": []}
        total_cost = 0
        total_time = 0
        total_tests = 0
        
        for result in model_results:
            test_cases = result["data"].get("testCases", [])
            total_tests += len(test_cases)
            
            for test_case in test_cases:
                metrics_data = test_case.get("metricsData", [])
                for metric in metrics_data:
                    metric_name = metric.get("name", "").lower()
                    if metric_name in all_scores:
                        all_scores[metric_name].append(metric.get("score", 0))
        
        # 성능 지표 계산
        performance_metrics = {
            "repo_name": repo_name,
            "model_name": model_name,
            "total_test_cases": total_tests,
            "correctness_mean": np.mean(all_scores["correctness"]) if all_scores["correctness"] else 0,
            "correctness_std": np.std(all_scores["correctness"]) if all_scores["correctness"] else 0,
            "clarity_mean": np.mean(all_scores["clarity"]) if all_scores["clarity"] else 0,
            "clarity_std": np.std(all_scores["clarity"]) if all_scores["clarity"] else 0,
            "actionability_mean": np.mean(all_scores["actionability"]) if all_scores["actionability"] else 0,
            "actionability_std": np.std(all_scores["actionability"]) if all_scores["actionability"] else 0,
            "json_correctness_mean": np.mean(all_scores["json_correctness"]) if all_scores["json_correctness"] else 0,
            "overall_pass_rate": calculate_overall_pass_rate(all_scores),
            "consistency_score": calculate_consistency_score(all_scores),  # 낮은 표준편차 = 높은 일관성
            "weighted_score": calculate_weighted_performance_score(all_scores)
        }
        
        performance_data.append(performance_metrics)
    
    return pd.DataFrame(performance_data)

def calculate_weighted_performance_score(scores: Dict[str, List[float]]) -> float:
    """
    메트릭별 가중치를 적용한 종합 성능 점수를 계산한다.
    
    가중치: Correctness(40%), Clarity(25%), Actionability(25%), JsonCorrectness(10%)
    """
    weights = {"correctness": 0.4, "clarity": 0.25, "actionability": 0.25, "json_correctness": 0.1}
    weighted_sum = 0
    
    for metric, weight in weights.items():
        if scores[metric]:
            weighted_sum += np.mean(scores[metric]) * weight
    
    return weighted_sum
```

**기술스택별 성능 차이 분석**
```python
def analyze_tech_stack_performance(performance_df: pd.DataFrame) -> Dict[str, Any]:
    """
    저장소별/기술스택별 모델 성능 차이를 분석한다.
    """
    tech_stack_mapping = {
        "cline": "typescript",
        "ecommerce-microservices": "java_spring", 
        "kotlin-realworld": "kotlin_jpa",
        "selvage-deprecated": "mixed"
    }
    
    performance_df["tech_stack"] = performance_df["repo_name"].map(tech_stack_mapping)
    
    # 기술스택별 모델 성능 분석
    tech_analysis = {}
    
    for tech_stack in performance_df["tech_stack"].unique():
        stack_data = performance_df[performance_df["tech_stack"] == tech_stack]
        
        tech_analysis[tech_stack] = {
            "best_model": {
                "correctness": stack_data.loc[stack_data["correctness_mean"].idxmax(), "model_name"],
                "clarity": stack_data.loc[stack_data["clarity_mean"].idxmax(), "model_name"],
                "actionability": stack_data.loc[stack_data["actionability_mean"].idxmax(), "model_name"],
                "overall": stack_data.loc[stack_data["weighted_score"].idxmax(), "model_name"]
            },
            "performance_gap": {
                "max_correctness": stack_data["correctness_mean"].max(),
                "min_correctness": stack_data["correctness_mean"].min(),
                "gap": stack_data["correctness_mean"].max() - stack_data["correctness_mean"].min()
            },
            "consistency_ranking": stack_data.nsmallest(3, "correctness_std")[["model_name", "correctness_std"]].to_dict("records")
        }
    
    return tech_analysis
```

#### 4.3 실패 패턴 분석 및 인사이트 도출

**실패 케이스 분류 및 패턴 분석**
```python
def analyze_failure_patterns(metrics_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    실패한 테스트 케이스들을 분석하여 공통 패턴과 개선 방향을 도출한다.
    """
    failure_patterns = {}
    
    for metric_name, data in metrics_analysis.items():
        failures = data["failures"]
        
        if not failures:
            continue
            
        # 실패 사유 분류
        reason_categories = categorize_failure_reasons(failures)
        
        # 모델별 실패 패턴
        model_failures = {}
        for failure in failures:
            model = failure["model_name"]
            if model not in model_failures:
                model_failures[model] = []
            model_failures[model].append(failure)
        
        # 저장소별 실패 패턴  
        repo_failures = {}
        for failure in failures:
            repo = failure["repo_name"]
            if repo not in repo_failures:
                repo_failures[repo] = []
            repo_failures[repo].append(failure)
        
        failure_patterns[metric_name] = {
            "total_failures": len(failures),
            "reason_categories": reason_categories,
            "worst_performing_models": sorted(model_failures.items(), 
                                           key=lambda x: len(x[1]), reverse=True)[:3],
            "problematic_repos": sorted(repo_failures.items(), 
                                      key=lambda x: len(x[1]), reverse=True)[:3],
            "improvement_suggestions": generate_improvement_suggestions(reason_categories, metric_name)
        }
    
    return failure_patterns

def categorize_failure_reasons(failures: List[Dict]) -> Dict[str, int]:
    """실패 사유를 카테고리별로 분류한다."""
    categories = {
        "missing_issues": 0,          # 이슈 누락
        "incorrect_line_numbers": 0,   # 잘못된 라인 번호
        "inappropriate_severity": 0,   # 부적절한 심각도
        "unclear_descriptions": 0,     # 불명확한 설명
        "non_actionable_suggestions": 0, # 실행 불가능한 제안
        "json_format_errors": 0,       # JSON 형식 오류
        "other": 0
    }
    
    for failure in failures:
        reason = failure["reason"].lower()
        
        if "missing" in reason or "not identified" in reason:
            categories["missing_issues"] += 1
        elif "line number" in reason or "incorrect" in reason:
            categories["incorrect_line_numbers"] += 1
        elif "severity" in reason or "inappropriate" in reason:
            categories["inappropriate_severity"] += 1
        elif "unclear" in reason or "vague" in reason:
            categories["unclear_descriptions"] += 1
        elif "actionable" in reason or "implementable" in reason:
            categories["non_actionable_suggestions"] += 1
        elif "json" in reason or "format" in reason:
            categories["json_format_errors"] += 1
        else:
            categories["other"] += 1
    
    return categories

def generate_improvement_suggestions(reason_categories: Dict[str, int], metric_name: str) -> List[str]:
    """실패 패턴을 기반으로 개선 제안을 생성한다."""
    suggestions = []
    
    if reason_categories["missing_issues"] > 5:
        suggestions.append(f"{metric_name}: 이슈 탐지율 향상을 위해 프롬프트에 더 구체적인 검토 지침 추가 필요")
    
    if reason_categories["incorrect_line_numbers"] > 3:
        suggestions.append(f"{metric_name}: 라인 번호 정확성 향상을 위해 diff 파싱 로직 개선 필요")
    
    if reason_categories["unclear_descriptions"] > 4:
        suggestions.append(f"{metric_name}: 설명 명확성 향상을 위해 예시 기반 프롬프트 개선 필요")
    
    if reason_categories["non_actionable_suggestions"] > 3:
        suggestions.append(f"{metric_name}: 실행가능한 제안을 위해 코드 예시 포함 지침 강화 필요")
    
    return suggestions
```

#### 4.4 프롬프트 버전 효과성 분석

**A/B 테스트 결과 분석**
```python
def analyze_prompt_version_effectiveness(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    동일 커밋에 대한 다른 프롬프트 버전 적용 결과를 통계적으로 비교 분석한다.
    """
    prompt_comparison = {}
    
    # 프롬프트 버전별 결과 그룹화 (review_log의 prompt_version 필드 기준)
    version_results = group_by_prompt_version(results)
    
    for v1, v2 in itertools.combinations(version_results.keys(), 2):
        # 동일 커밋에 대한 결과만 비교
        common_commits = find_common_commits(version_results[v1], version_results[v2])
        
        if len(common_commits) < 5:  # 통계적 유의성을 위한 최소 샘플 수
            continue
            
        comparison_result = perform_statistical_comparison(
            version_results[v1], version_results[v2], common_commits
        )
        
        prompt_comparison[f"{v1}_vs_{v2}"] = comparison_result
    
    return prompt_comparison

def perform_statistical_comparison(v1_results: List, v2_results: List, common_commits: List) -> Dict[str, Any]:
    """두 프롬프트 버전 간 통계적 비교를 수행한다."""
    v1_scores = extract_scores_for_commits(v1_results, common_commits)
    v2_scores = extract_scores_for_commits(v2_results, common_commits)
    
    # 대응표본 t-검정 (paired t-test)
    from scipy import stats
    
    comparison = {}
    
    for metric in ["correctness", "clarity", "actionability", "json_correctness"]:
        v1_metric_scores = [scores[metric] for scores in v1_scores]
        v2_metric_scores = [scores[metric] for scores in v2_scores]
        
        # 통계적 검정
        t_stat, p_value = stats.ttest_rel(v2_metric_scores, v1_metric_scores)
        effect_size = calculate_cohens_d(v2_metric_scores, v1_metric_scores)
        
        comparison[metric] = {
            "v1_mean": np.mean(v1_metric_scores),
            "v2_mean": np.mean(v2_metric_scores),
            "improvement": np.mean(v2_metric_scores) - np.mean(v1_metric_scores),
            "improvement_percentage": ((np.mean(v2_metric_scores) - np.mean(v1_metric_scores)) / np.mean(v1_metric_scores)) * 100,
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "effect_size": effect_size,
            "effect_magnitude": interpret_effect_size(effect_size)
        }
    
    return comparison

def calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Cohen's d 효과 크기를 계산한다."""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / 
                        (len(group1) + len(group2) - 2))
    
    return (mean2 - mean1) / pooled_std

def interpret_effect_size(d: float) -> str:
    """Cohen's d 값을 해석한다."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "무시할 수 있는 효과"
    elif abs_d < 0.5:
        return "작은 효과"
    elif abs_d < 0.8:
        return "중간 효과"
    else:
        return "큰 효과"
```

#### 4.5 자동화된 인사이트 도출 및 의사결정 지원

**최적 모델 조합 자동 추천**
```python
def recommend_optimal_model_configuration(performance_df: pd.DataFrame, 
                                        cost_data: Dict[str, float],
                                        priority: str = "balanced") -> Dict[str, Any]:
    """
    성능, 비용, 일관성을 종합적으로 고려한 최적 모델 조합을 추천한다.
    
    Args:
        priority: "performance", "cost", "balanced" 중 선택
    """
    recommendations = {}
    
    # 우선순위별 가중치 설정
    weight_configs = {
        "performance": {"performance": 0.7, "cost": 0.1, "consistency": 0.2},
        "cost": {"performance": 0.3, "cost": 0.5, "consistency": 0.2},
        "balanced": {"performance": 0.5, "cost": 0.3, "consistency": 0.2}
    }
    
    weights = weight_configs[priority]
    
    # 저장소별 최적 모델 추천
    for repo in performance_df["repo_name"].unique():
        repo_data = performance_df[performance_df["repo_name"] == repo]
        
        # 정규화된 점수 계산
        repo_data["performance_score_norm"] = normalize_scores(repo_data["weighted_score"])
        repo_data["cost_score_norm"] = normalize_scores([1/cost_data.get(model, 1) for model in repo_data["model_name"]])
        repo_data["consistency_score_norm"] = normalize_scores([1/std for std in repo_data["correctness_std"]])
        
        # 종합 점수 계산
        repo_data["total_score"] = (
            repo_data["performance_score_norm"] * weights["performance"] +
            repo_data["cost_score_norm"] * weights["cost"] +
            repo_data["consistency_score_norm"] * weights["consistency"]
        )
        
        best_model = repo_data.loc[repo_data["total_score"].idxmax()]
        
        recommendations[repo] = {
            "recommended_model": best_model["model_name"],
            "total_score": best_model["total_score"],
            "performance_score": best_model["weighted_score"],
            "estimated_monthly_cost": estimate_monthly_cost(best_model["model_name"], cost_data),
            "confidence_level": calculate_confidence_level(repo_data),
            "alternative_models": get_alternative_models(repo_data, best_model["model_name"])
        }
    
    return recommendations

def generate_actionable_insights(performance_df: pd.DataFrame, 
                               failure_patterns: Dict[str, Any],
                               prompt_comparison: Dict[str, Any]) -> List[str]:
    """실제 의사결정을 위한 구체적인 액션 아이템을 생성한다."""
    insights = []
    
    # 성능 기반 인사이트
    best_overall_model = performance_df.loc[performance_df["weighted_score"].idxmax(), "model_name"]
    insights.append(f"🏆 전체 최고 성능 모델: {best_overall_model} (종합 점수: {performance_df['weighted_score'].max():.3f})")
    
    # 일관성 기반 인사이트
    most_consistent_model = performance_df.loc[performance_df["consistency_score"].idxmax(), "model_name"]
    insights.append(f"🎯 가장 일관성 있는 모델: {most_consistent_model}")
    
    # 개선 우선순위
    for metric, patterns in failure_patterns.items():
        if patterns["total_failures"] > 10:
            worst_model = patterns["worst_performing_models"][0][0]
            insights.append(f"⚠️ {metric} 개선 필요: {worst_model} 모델의 실패율이 높음")
    
    # 프롬프트 개선 효과
    for comparison, result in prompt_comparison.items():
        for metric, data in result.items():
            if data["is_significant"] and data["improvement"] > 0.1:
                insights.append(f"📈 {comparison} 비교: {metric}에서 {data['improvement_percentage']:.1f}% 유의미한 개선")
    
    return insights
```

#### 4.6 자동화된 보고서 생성 및 시각화

**종합 분석 보고서 스키마**
```json
{
  "evaluation_session": {
    "id": "eval_20240622_143022_a1b2c3d",
    "date": "2024-06-22T14:30:22Z",
    "repositories_analyzed": ["cline", "ecommerce-microservices", "kotlin-realworld"],
    "models_compared": ["gemini-2.5-pro", "claude-sonnet-4", "claude-sonnet-4-thinking"],
    "total_test_cases": 450,
    "evaluation_duration_minutes": 127
  },
  "executive_summary": {
    "key_findings": [
      "gemini-2.5-pro가 TypeScript 코드에서 최고 성능 (0.847 종합 점수)",
      "claude-sonnet-4가 가장 일관성 있는 결과 제공 (표준편차 0.045)",
      "프롬프트 v3가 v2 대비 Correctness에서 평균 15.3% 개선"
    ],
    "recommended_actions": [
      "TypeScript 프로젝트에는 gemini-2.5-pro 우선 사용",
      "프롬프트 v3를 모든 모델에 적용",
      "Java 코드 리뷰 품질 개선을 위한 specialized prompt 개발"
    ],
    "estimated_cost_impact": {
      "current_monthly_cost": 245.67,
      "optimized_monthly_cost": 189.23,
      "savings_percentage": 23.0
    }
  },
  "detailed_analysis": {
    "model_performance_matrix": {
      "gemini-2.5-pro": {
        "overall_score": 0.847,
        "correctness": {"mean": 0.832, "std": 0.067, "pass_rate": 0.89},
        "clarity": {"mean": 0.798, "std": 0.054, "pass_rate": 0.92},
        "actionability": {"mean": 0.756, "std": 0.089, "pass_rate": 0.85},
        "json_correctness": {"mean": 0.945, "std": 0.023, "pass_rate": 0.98},
        "strengths": ["높은 정확성", "우수한 JSON 형식 준수"],
        "weaknesses": ["실행가능성 점수가 상대적으로 낮음"],
        "best_for": ["typescript", "javascript", "python"]
      },
      "claude-sonnet-4": {
        "overall_score": 0.789,
        "correctness": {"mean": 0.778, "std": 0.045, "pass_rate": 0.82},
        "clarity": {"mean": 0.823, "std": 0.041, "pass_rate": 0.94},
        "actionability": {"mean": 0.801, "std": 0.052, "pass_rate": 0.88},
        "json_correctness": {"mean": 0.934, "std": 0.031, "pass_rate": 0.96},
        "strengths": ["가장 일관성 있는 성능", "우수한 명확성"],
        "weaknesses": ["정확성에서 상대적 열세"],
        "best_for": ["java", "kotlin", "enterprise_code"]
      }
    },
    "failure_pattern_analysis": {
      "correctness_failures": {
        "total": 23,
        "categories": {
          "missing_issues": 8,
          "incorrect_line_numbers": 5,
          "inappropriate_severity": 6,
          "other": 4
        },
        "worst_performing_repos": ["ecommerce-microservices", "kotlin-realworld"],
        "improvement_suggestions": [
          "Java/Kotlin 코드를 위한 specialized prompt 개발",
          "라인 번호 매핑 정확도 개선 알고리즘 구현"
        ]
      }
    },
    "prompt_version_comparison": {
      "v2_vs_v3": {
        "sample_size": 45,
        "correctness": {
          "improvement": 0.153,
          "p_value": 0.003,
          "effect_size": 0.712,
          "significance": "통계적으로 유의미한 개선"
        },
        "clarity": {
          "improvement": 0.087,
          "p_value": 0.021,
          "effect_size": 0.423,
          "significance": "중간 정도의 개선"
        }
      }
    },
    "optimization_recommendations": {
      "model_assignments": {
        "cline": "gemini-2.5-pro",
        "ecommerce-microservices": "claude-sonnet-4", 
        "kotlin-realworld": "claude-sonnet-4",
        "selvage-deprecated": "gemini-2.5-pro"
      },
      "prompt_versions": {
        "current_best": "v3",
        "next_iteration_focus": ["actionability 개선", "Java/Kotlin 특화"]
      },
      "cost_optimization": {
        "high_confidence_cases": "gemini-2.5-flash 사용으로 30% 비용 절감 가능",
        "critical_cases": "claude-sonnet-4-thinking 사용으로 품질 보장"
      }
    }
  },
  "monitoring_alerts": [
    {
      "type": "performance_degradation",
      "model": "gemini-2.5-pro",
      "repo": "ecommerce-microservices", 
      "metric": "correctness",
      "threshold": 0.7,
      "current_value": 0.647,
      "recommendation": "프롬프트 재조정 또는 모델 교체 검토"
    }
  ],
  "next_evaluation_recommendations": {
    "focus_areas": ["Java 코드 리뷰 품질", "비용 최적화", "실시간 모니터링"],
    "new_test_cases": ["복잡한 비즈니스 로직", "보안 취약점 탐지"],
    "experiment_ideas": ["모델 앙상블", "dynamic prompt selection"]
  }
}
```

**시각화 자동 생성**
```python
def generate_performance_visualizations(performance_df: pd.DataFrame, 
                                      output_dir: str) -> List[str]:
    """성능 분석 결과를 시각화하여 차트로 생성한다."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    output_paths = []
    
    # 1. 모델별 종합 성능 레이더 차트
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    metrics = ['correctness_mean', 'clarity_mean', 'actionability_mean', 'json_correctness_mean']
    for _, row in performance_df.iterrows():
        values = [row[metric] for metric in metrics]
        values += values[:1]  # 차트를 닫기 위해 첫 값을 마지막에 추가
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model_name'])
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Correctness', 'Clarity', 'Actionability', 'JSON Correctness'])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title('모델별 종합 성능 비교')
    
    radar_path = f"{output_dir}/model_performance_radar.png"
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    output_paths.append(radar_path)
    plt.close()
    
    # 2. 저장소별 모델 성능 히트맵
    pivot_data = performance_df.pivot(index='model_name', columns='repo_name', values='weighted_score')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0.7, 
                fmt='.3f', cbar_kws={'label': '종합 성능 점수'})
    plt.title('저장소별 모델 성능 히트맵')
    plt.xlabel('저장소')
    plt.ylabel('모델')
    
    heatmap_path = f"{output_dir}/repo_model_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    output_paths.append(heatmap_path)
    plt.close()
    
    # 3. 성능 vs 일관성 산점도
    plt.figure(figsize=(10, 8))
    
    for repo in performance_df['repo_name'].unique():
        repo_data = performance_df[performance_df['repo_name'] == repo]
        plt.scatter(repo_data['weighted_score'], repo_data['consistency_score'], 
                   label=repo, s=100, alpha=0.7)
        
        # 모델명 라벨 추가
        for _, row in repo_data.iterrows():
            plt.annotate(row['model_name'], 
                        (row['weighted_score'], row['consistency_score']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('종합 성능 점수')
    plt.ylabel('일관성 점수')
    plt.title('성능 vs 일관성 분석')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    scatter_path = f"{output_dir}/performance_consistency_scatter.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    output_paths.append(scatter_path)
    plt.close()
    
    return output_paths
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