# Selvage 평가 에이전트 - 구현 가이드 (2단계)

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

## Phase 1-2 Tool 구현

### Phase 1 Tools: Commit Collection

#### GitLogTool - Git 로그 조회

**LLM Tool Definition (OpenAI Function Calling)**
```javascript
{
  "type": "function",
  "function": {
    "name": "git_log",
    "description": "지정된 저장소에서 커밋 로그를 조회합니다",
    "parameters": {
      "type": "object",
      "properties": {
        "repo_path": {
          "type": "string",
          "description": "저장소 경로"
        },
        "since": {
          "type": "string",
          "description": "시작 날짜 (YYYY-MM-DD 형식)"
        },
        "until": {
          "type": "string",
          "description": "종료 날짜 (YYYY-MM-DD 형식)"
        },
        "grep": {
          "type": "string",
          "description": "커밋 메시지에서 검색할 키워드"
        },
        "max_count": {
          "type": "integer",
          "description": "조회할 최대 커밋 수"
        }
      },
      "required": ["repo_path"]
    }
  }
}
```

**LLM Tool Definition (Anthropic Claude)**
```javascript
{
  "name": "git_log",
  "description": "지정된 저장소에서 커밋 로그를 조회합니다",
  "input_schema": {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "저장소 경로"
      },
      "since": {
        "type": "string",
        "description": "시작 날짜 (YYYY-MM-DD 형식)"
      },
      "until": {
        "type": "string",
        "description": "종료 날짜 (YYYY-MM-DD 형식)"
      },
      "grep": {
        "type": "string",
        "description": "커밋 메시지에서 검색할 키워드"
      },
      "max_count": {
        "type": "integer",
        "description": "조회할 최대 커밋 수"
      }
    },
    "required": ["repo_path"]
  }
}
```

**Python Implementation**
```python
class GitLogTool(Tool):
    """Git 커밋 로그를 조회하는 도구"""
    
    @property
    def name(self) -> str:
        return "git_log"
    
    @property
    def description(self) -> str:
        return "지정된 저장소에서 커밋 로그를 조회합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "저장소 경로"},
                "since": {"type": "string", "description": "시작 날짜 (YYYY-MM-DD)"},
                "until": {"type": "string", "description": "종료 날짜 (YYYY-MM-DD)"},
                "grep": {"type": "string", "description": "커밋 메시지 필터"},
                "max_count": {"type": "integer", "description": "최대 커밋 수"}
            },
            "required": ["repo_path"]
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Git 로그 실행"""
        repo_path = kwargs["repo_path"]
        
        cmd = ["git", "log", "--oneline", "--no-merges"]
        
        if kwargs.get("since"):
            cmd.extend(["--since", kwargs["since"]])
        if kwargs.get("until"):
            cmd.extend(["--until", kwargs["until"]])
        if kwargs.get("grep"):
            cmd.extend(["--grep", kwargs["grep"]])
        if kwargs.get("max_count"):
            cmd.extend(["-n", str(kwargs["max_count"])])
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                commits = self._parse_git_log(stdout.decode())
                return ToolResult(success=True, data=commits)
            else:
                return ToolResult(
                    success=False, 
                    data=None,
                    error_message=stderr.decode()
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                data=None, 
                error_message=str(e)
            )
    
    def _parse_git_log(self, log_output: str) -> List[Dict[str, str]]:
        """Git 로그 출력 파싱"""
        commits = []
        for line in log_output.strip().split('\n'):
            if line:
                parts = line.split(' ', 1)
                commits.append({
                    "hash": parts[0],
                    "message": parts[1] if len(parts) > 1 else ""
                })
        return commits
```

#### CommitScoringTool - 커밋 배점 도구

**LLM Tool Definition (OpenAI Function Calling)**
```javascript
{
  "type": "function",
  "function": {
    "name": "commit_scoring",
    "description": "커밋의 평가 가치를 배점하여 의미있는 커밋을 선별합니다",
    "parameters": {
      "type": "object",
      "properties": {
        "commit_hash": {
          "type": "string",
          "description": "평가할 커밋의 해시값"
        },
        "repo_path": {
          "type": "string",
          "description": "Git 저장소 경로"
        },
        "scoring_config": {
          "type": "object",
          "description": "배점 설정 (선택사항)",
          "properties": {
            "file_type_weight": {"type": "number", "description": "파일 타입 가중치"},
            "scale_weight": {"type": "number", "description": "변경 규모 가중치"},
            "characteristic_weight": {"type": "number", "description": "커밋 특성 가중치"}
          }
        }
      },
      "required": ["commit_hash", "repo_path"]
    }
  }
}
```

**LLM Tool Definition (Anthropic Claude)**
```javascript
{
  "name": "commit_scoring",
  "description": "커밋의 평가 가치를 배점하여 의미있는 커밋을 선별합니다",
  "input_schema": {
    "type": "object",
    "properties": {
      "commit_hash": {
        "type": "string",
        "description": "평가할 커밋의 해시값"
      },
      "repo_path": {
        "type": "string",
        "description": "Git 저장소 경로"
      },
      "scoring_config": {
        "type": "object",
        "description": "배점 설정 (선택사항)",
        "properties": {
          "file_type_weight": {"type": "number", "description": "파일 타입 가중치"},
          "scale_weight": {"type": "number", "description": "변경 규모 가중치"},
          "characteristic_weight": {"type": "number", "description": "커밋 특성 가중치"}
        }
      }
    },
    "required": ["commit_hash", "repo_path"]
  }
}
```

**Python Implementation**
```python
@dataclass
class CommitScore:
    """커밋 점수 데이터 클래스"""
    total: int
    breakdown: Dict[str, int]

class CommitScoringTool(Tool):
    """커밋 배점 및 필터링 도구"""
    
    @property
    def name(self) -> str:
        return "commit_scoring"
        
    @property
    def description(self) -> str:
        return "커밋의 평가 가치를 배점하여 의미있는 커밋을 선별합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "commit_hash": {"type": "string", "description": "커밋 해시"},
                "repo_path": {"type": "string", "description": "저장소 경로"},
                "scoring_config": {"type": "object", "description": "배점 설정"}
            },
            "required": ["commit_hash", "repo_path"]
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        commit_hash = kwargs["commit_hash"]
        repo_path = kwargs["repo_path"]
        scoring_config = kwargs.get("scoring_config", {})
        
        # 커밋 정보 수집
        commit_info = await self._get_commit_info(commit_hash, repo_path)
        
        # 배점 계산
        score = self._calculate_score(commit_info, scoring_config)
        
        return ToolResult(
            success=True,
            data={
                "commit_hash": commit_hash,
                "score": score,
                "breakdown": score.breakdown,
                "commit_info": commit_info
            }
        )
    
    async def _get_commit_info(self, commit_hash: str, repo_path: str) -> Dict[str, Any]:
        """커밋 상세 정보 수집"""
        try:
            # 커밋 메시지 및 기본 정보
            log_result = await asyncio.create_subprocess_exec(
                "git", "log", "--format=%H|%s|%an|%ad", "-1", commit_hash,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            log_stdout, _ = await log_result.communicate()
            
            # 변경 통계
            stat_result = await asyncio.create_subprocess_exec(
                "git", "show", "--stat", "--format=", commit_hash,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stat_stdout, _ = await stat_result.communicate()
            
            # 변경된 파일 목록
            files_result = await asyncio.create_subprocess_exec(
                "git", "show", "--name-only", "--format=", commit_hash,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            files_stdout, _ = await files_result.communicate()
            
            # 파싱
            log_parts = log_stdout.decode().strip().split('|')
            files = [f.strip() for f in files_stdout.decode().strip().split('\n') if f.strip()]
            
            # 통계 정보 파싱
            stats = self._parse_stat_output(stat_stdout.decode())
            
            return {
                "hash": log_parts[0],
                "message": log_parts[1],
                "author": log_parts[2],
                "date": log_parts[3],
                "files": files,
                "stats": stats
            }
            
        except Exception as e:
            raise Exception(f"Failed to get commit info: {str(e)}")
    
    def _parse_stat_output(self, stat_output: str) -> Dict[str, int]:
        """git show --stat 출력 파싱"""
        lines = stat_output.strip().split('\n')
        
        # 마지막 줄에서 통계 정보 추출 (예: " 3 files changed, 45 insertions(+), 12 deletions(-)")
        summary_line = lines[-1] if lines else ""
        
        import re
        files_match = re.search(r'(\d+) files? changed', summary_line)
        insertions_match = re.search(r'(\d+) insertions?', summary_line)
        deletions_match = re.search(r'(\d+) deletions?', summary_line)
        
        return {
            "files_changed": int(files_match.group(1)) if files_match else 0,
            "lines_added": int(insertions_match.group(1)) if insertions_match else 0,
            "lines_deleted": int(deletions_match.group(1)) if deletions_match else 0
        }
    
    def _calculate_score(self, commit_info: Dict, config: Dict) -> CommitScore:
        """커밋 배점 계산"""
        # A. 파일 타입 감점 조정
        file_type_score = self._calculate_file_type_score(commit_info["files"])
        
        # B. 변경 규모 적정성
        scale_score = self._calculate_scale_score(commit_info["stats"])
        
        # C. 커밋 특성
        characteristic_score = self._calculate_characteristic_score(
            commit_info["message"], 
            commit_info["files"]
        )
        
        # D. 시간 가중치
        time_score = self._calculate_time_score(commit_info["date"])
        
        # E. 추가 조정사항
        adjustment_score = self._calculate_adjustments(commit_info)
        
        total_score = (
            file_type_score + scale_score + 
            characteristic_score + time_score + adjustment_score
        )
        
        return CommitScore(
            total=max(0, min(100, total_score)),
            breakdown={
                "file_type": file_type_score,
                "scale": scale_score, 
                "characteristic": characteristic_score,
                "time": time_score,
                "adjustment": adjustment_score
            }
        )
    
    def _calculate_file_type_score(self, files: List[str]) -> int:
        """파일 타입 기반 점수 계산"""
        base_score = 100
        
        # 비-코드 파일 감점 (-5점)
        non_code_extensions = {
            '.txt', '.rst', '.adoc', '.doc', '.docx', '.pdf',
            '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.mp4', '.mp3', '.wav',
            '.zip', '.tar', '.gz', '.exe', '.dll', '.so', '.dylib', '.bin'
        }
        
        # 경미한 감점 파일 (-2점)
        minor_deduction_extensions = {
            '.json', '.yaml', '.yml', '.toml', '.ini', '.env', '.md', '.mdc'
        }
        
        for file in files:
            file_lower = file.lower()
            
            # 자동생성 파일 체크
            if any(pattern in file_lower for pattern in ['package-lock.json', '.lock', '.cache']):
                base_score -= 5
            # 비-코드 파일 체크
            elif any(file_lower.endswith(ext) for ext in non_code_extensions):
                base_score -= 5
            # 경미한 감점 파일 체크
            elif any(file_lower.endswith(ext) for ext in minor_deduction_extensions):
                base_score -= 2
        
        return base_score
    
    def _calculate_scale_score(self, stats: Dict[str, int]) -> int:
        """변경 규모 적정성 점수 계산"""
        files_changed = stats["files_changed"]
        lines_total = stats["lines_added"] + stats["lines_deleted"]
        
        # 파일 수 적정성 (10점)
        if 2 <= files_changed <= 4:
            file_score = 10
        elif 5 <= files_changed <= 7:
            file_score = 7
        elif 8 <= files_changed <= 10:
            file_score = 4
        else:
            file_score = 0
        
        # 변경 라인 수 밸런스 (15점)
        if 50 <= lines_total <= 200:
            line_score = 15
        elif 201 <= lines_total <= 400:
            line_score = 10
        elif 401 <= lines_total <= 600:
            line_score = 5
        else:
            line_score = 0
        
        # 추가/삭제 비율 체크
        if stats["lines_added"] > 0 and stats["lines_deleted"] > 0:
            ratio = max(stats["lines_added"], stats["lines_deleted"]) / min(stats["lines_added"], stats["lines_deleted"])
            if ratio > 10:  # 극단적인 비율
                line_score -= 5
        
        return file_score + line_score
    
    def _calculate_characteristic_score(self, message: str, files: List[str]) -> int:
        """커밋 특성 점수 계산"""
        score = 0
        message_lower = message.lower()
        
        # 긍정 키워드 (각 +5점, 최대 15점)
        positive_keywords = ['fix', 'refactor', 'improve', 'optimize', 'enhance', 
                           'feature', 'implement', 'add', 'update', 'security', 'performance', 'bug']
        positive_matches = sum(1 for keyword in positive_keywords if keyword in message_lower)
        score += min(positive_matches * 5, 15)
        
        # 부정 키워드 (각 -3점)
        negative_keywords = ['typo', 'format', 'style', 'whitespace', 'lint', 'merge', 'revert', 'backup']
        negative_matches = sum(1 for keyword in negative_keywords if keyword in message_lower)
        score -= negative_matches * 3
        
        # 경로 패턴 가중치 (10점)
        path_score = 0
        for file in files:
            file_lower = file.lower()
            if any(pattern in file_lower for pattern in ['src/', 'lib/', 'core/']):
                path_score = 10
                break
            elif any(pattern in file_lower for pattern in ['utils/', 'helpers/']):
                path_score = max(path_score, 7)
            elif any(pattern in file_lower for pattern in ['config/', 'build/', '.github/']):
                path_score = -5
                break
        
        score += path_score
        return score
    
    def _calculate_time_score(self, date_str: str) -> int:
        """시간 가중치 점수 계산"""
        from datetime import datetime, timedelta
        
        try:
            # Git 날짜 파싱 (다양한 형식 지원)
            commit_date = datetime.strptime(date_str.split()[0], '%Y-%m-%d')
            now = datetime.now()
            days_diff = (now - commit_date).days
            
            if days_diff <= 30:
                return 20
            elif days_diff <= 90:
                return 15
            elif days_diff <= 180:
                return 10
            elif days_diff <= 365:
                return 5
            else:
                return 2
        except:
            return 2  # 파싱 실패 시 최소 점수
    
    def _calculate_adjustments(self, commit_info: Dict) -> int:
        """추가 조정사항 점수 계산"""
        score = 0
        
        # 머지 커밋 처리는 별도 로직에서 처리 (현재는 --no-merges 옵션 사용)
        # 작성자 다양성 처리는 전체 커밋 리스트에서 처리
        
        return score
```

### Phase 2 Tools: Review Execution

#### SelvageExecutorTool - Selvage 리뷰 실행

**LLM Tool Definition (OpenAI Function Calling)**
```javascript
{
  "type": "function",
  "function": {
    "name": "selvage_executor",
    "description": "Selvage를 실행하여 코드 리뷰를 수행합니다",
    "parameters": {
      "type": "object",
      "properties": {
        "repo_path": {
          "type": "string",
          "description": "Git 저장소 경로"
        },
        "commit_hash": {
          "type": "string",
          "description": "리뷰할 커밋의 해시값"
        },
        "model": {
          "type": "string",
          "description": "사용할 AI 모델 (예: gpt-4, claude-3-sonnet)",
          "enum": ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "claude-3-haiku"]
        },
        "log_dir": {
          "type": "string",
          "description": "리뷰 로그를 저장할 디렉토리 경로 (기본값: ~/Library/selvage-eval-agent/review_logs)"
        }
      },
      "required": ["repo_path", "commit_hash", "model"]
    }
  }
}
```

**LLM Tool Definition (Anthropic Claude)**
```javascript
{
  "name": "selvage_executor",
  "description": "Selvage를 실행하여 코드 리뷰를 수행합니다",
  "input_schema": {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "Git 저장소 경로"
      },
      "commit_hash": {
        "type": "string",
        "description": "리뷰할 커밋의 해시값"
      },
      "model": {
        "type": "string",
        "description": "사용할 AI 모델 (예: gpt-4, claude-3-sonnet)",
        "enum": ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "claude-3-haiku"]
      },
      "log_dir": {
        "type": "string",
        "description": "리뷰 로그를 저장할 디렉토리 경로 (기본값: ~/Library/selvage-eval-agent/review_logs)"
      }
    },
    "required": ["repo_path", "commit_hash", "model"]
  }
}
```

**Python Implementation**
```python
class SelvageExecutorTool(Tool):
    """Selvage 코드 리뷰 실행 도구"""
    
    @property
    def name(self) -> str:
        return "selvage_executor"
    
    @property
    def description(self) -> str:
        return "Selvage를 실행하여 코드 리뷰를 수행합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "저장소 경로"},
                "commit_hash": {"type": "string", "description": "리뷰할 커밋 해시"},
                "model": {"type": "string", "description": "사용할 모델"},
                "log_dir": {"type": "string", "description": "로그 저장 디렉토리"}
            },
            "required": ["repo_path", "commit_hash", "model"]
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        repo_path = kwargs["repo_path"]
        commit_hash = kwargs["commit_hash"]
        model = kwargs["model"]
        log_dir = kwargs.get("log_dir", "~/Library/selvage-eval-agent/review_logs")
        
        # 1. 현재 브랜치 저장
        current_branch = await self._get_current_branch(repo_path)
        
        # 2. 커밋 체크아웃
        checkout_result = await self._checkout_commit(repo_path, commit_hash)
        if not checkout_result.success:
            return checkout_result
        
        try:
            # 3. 부모 커밋 ID 획득
            parent_hash = await self._get_parent_commit(repo_path)
            
            # 4. Selvage 실행
            review_result = await self._execute_selvage_review(
                repo_path=repo_path,
                target_commit=parent_hash,
                model=model,
                log_dir=log_dir
            )
            
            return review_result
            
        finally:
            # 5. 원래 브랜치로 복원
            await self._checkout_branch(repo_path, current_branch)
    
    async def _get_current_branch(self, repo_path: str) -> str:
        """현재 브랜치 이름 가져오기"""
        try:
            result = await asyncio.create_subprocess_exec(
                "git", "branch", "--show-current",
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            return stdout.decode().strip() or "main"
        except:
            return "main"
    
    async def _checkout_commit(self, repo_path: str, commit_hash: str) -> ToolResult:
        """특정 커밋으로 체크아웃"""
        try:
            result = await asyncio.create_subprocess_exec(
                "git", "checkout", commit_hash,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return ToolResult(success=True, data={"output": stdout.decode()})
            else:
                return ToolResult(
                    success=False,
                    error_message=f"Failed to checkout commit {commit_hash}: {stderr.decode()}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Error checking out commit: {str(e)}"
            )
    
    async def _checkout_branch(self, repo_path: str, branch: str) -> ToolResult:
        """특정 브랜치로 체크아웃"""
        try:
            result = await asyncio.create_subprocess_exec(
                "git", "checkout", branch,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.communicate()
            return ToolResult(success=result.returncode == 0)
        except:
            return ToolResult(success=False)
    
    async def _get_parent_commit(self, repo_path: str) -> str:
        """현재 커밋의 부모 커밋 해시 가져오기"""
        try:
            result = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "HEAD^",
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            return stdout.decode().strip()
        except Exception as e:
            raise Exception(f"Failed to get parent commit: {str(e)}")
    
    async def _execute_selvage_review(self, **params) -> ToolResult:
        """Selvage 리뷰 실제 실행"""
        repo_path = params["repo_path"]
        target_commit = params["target_commit"]
        model = params["model"]
        log_dir = params["log_dir"]
        
        # 로그 디렉토리 생성
        os.makedirs(log_dir, exist_ok=True)
        
        cmd = [
            "selvage", "review",
            "--target-commit", target_commit,
            "--model", model,
            "--log-dir", log_dir
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300  # 5분 타임아웃
            )
            
            if process.returncode == 0:
                # 리뷰 로그 파일 위치 파악 및 반환
                log_file = await self._find_review_log_file(log_dir)
                return ToolResult(
                    success=True,
                    data={
                        "log_file": log_file, 
                        "stdout": stdout.decode(),
                        "stderr": stderr.decode()
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    error_message=f"Selvage execution failed: {stderr.decode()}"
                )
                
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error_message="Selvage execution timed out after 5 minutes"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Selvage execution error: {str(e)}"
            )
    
    async def _find_review_log_file(self, log_dir: str) -> Optional[str]:
        """리뷰 로그 파일 찾기"""
        try:
            import glob
            pattern = os.path.join(log_dir, "*.json")
            files = glob.glob(pattern)
            
            if files:
                # 가장 최근 파일 반환
                return max(files, key=os.path.getctime)
            return None
        except:
            return None
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