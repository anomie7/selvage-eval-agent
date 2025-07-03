# Selvage 평가 에이전트 - 구현 가이드 (2단계)

### 2단계: Selvage 리뷰 실행

#### 목표
선별된 커밋들에 대해 다중 모델로 Selvage 리뷰 실행 및 결과 수집

#### 구체적인 명령어 패턴
```python
from selvage_eval.commit_collection import MeaningfulCommitsData

# JSON 파일을 MeaningfulCommitsData로 변환하여 읽기
meaningful_commits = MeaningfulCommitsData.from_json('meaningful_commits.json')

# repo_name 별 커밋 목록 순회   
for repo in meaningful_commits.repositories:
    repo_path = repo.repo_path
    repo_name = repo.repo_name
    for commit in repo.commits:
        TODO("문서 아래에 명시된 프로세스를 구현할 것")
```

```python
from selvage_eval.tools.tool_executor import ToolExecutor

# ToolExecutor 인스턴스 생성
tool_executor = ToolExecutor()

# commit checkout
checkout_result = tool_executor.execute_tool_call(
    "execute_safe_command",
    {
        "command": f"git checkout {commit_id}",
        "cwd": repo_path,
        "capture_output": True,
        "timeout": 60
    }
)

# get parent commit id
parent_commit_result = tool_executor.execute_tool_call(
    "execute_safe_command",
    {
        "command": "git rev-parse HEAD^",
        "cwd": repo_path,
        "capture_output": True,
        "timeout": 60
    }
)

if parent_commit_result.success:
    parent_commit_id = parent_commit_result.data["stdout"].strip()
    
    # 리뷰 실행
    review_result = tool_executor.execute_tool_call(
        "execute_safe_command",
        {
            "command": f"selvage review --no-print --target-commit {parent_commit_id} --model {model_name} --log-dir ~/Library/selvage-eval-agent/review_logs/{repo_name}/{commit_id}/{model_name}",
            "cwd": repo_path,
            "capture_output": True,
            "timeout": 300
        }
    )

# checkout HEAD로 초기화 (커밋 체크아웃 후 초기화)
reset_result = tool_executor.execute_tool_call(
    "execute_safe_command",
    {
        "command": "git checkout main",
        "cwd": repo_path,
        "capture_output": True,
        "timeout": 60
    }
)
```

#### 다중 모델 병렬 처리
```python
# 설정 파일에서 review_models 로딩
review_models = config.review_models  # ["gemini-2.5-pro", "claude-sonnet-4", "claude-sonnet-4-thinking"]

# 각 모델별로 execute_reviews Tool을 병렬 실행
tasks = [
    tool_executor.execute_tool_call(
        "execute_reviews",
        {
            "meaningful_commits_path": "/path/to/meaningful_commits.json",
            "output_dir": "~/Library/selvage-eval-agent/review_logs",
            "model": model
        }
    )
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

## ReviewExecutor Tool 구현 명세

### Tool 클래스 인터페이스

```python
class ReviewExecutorTool(Tool):
    """Selvage 리뷰 실행 도구"""
    
    @property
    def name(self) -> str:
        return "execute_reviews"
    
    @property  
    def description(self) -> str:
        return "선별된 커밋들에 대해 다중 모델로 Selvage 리뷰를 실행하고 결과를 수집합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return generate_parameters_schema_from_hints(self.execute)
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """파라미터 유효성 검증
        
        Args:
            params: 검증할 파라미터 딕셔너리
            
        Returns:
            bool: 유효성 검증 결과
        """
        # 필수 파라미터 확인
        required_params = ['meaningful_commits_path', 'output_dir', 'model']
        for param in required_params:
            if param not in params:
                return False
            if not isinstance(params[param], str) or not params[param].strip():
                return False
                
        return True
    
    def execute(self, meaningful_commits_path: str, output_dir: str = '~/Library/selvage-eval-agent/review_logs', 
                model: str) -> ToolResult:
        """리뷰 실행
        
        Args:
            meaningful_commits_path: meaningful_commits.json 파일 경로
            output_dir: 리뷰 결과 저장 디렉토리 경로 (예: ~/Library/selvage-eval-agent/review_logs)
            model: 사용할 리뷰 모델
            
        Returns:
            ToolResult: 리뷰 실행 결과 (ReviewExecutionSummary 포함)
        """
```

### 파라미터 명세

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `meaningful_commits_path` | str | O | meaningful_commits.json 파일의 절대 경로 |
| `output_dir` | str | O | 리뷰 결과를 저장할 디렉토리 경로 |
| `model` | str | O | 사용할 모델 |

### 반환값 명세

**성공 시 (ToolResult)**:
```python
ToolResult(
    success=True,
    data=ReviewExecutionSummary(
        total_commits_reviewed=15,
        total_reviews_executed=15,  # 15 커밋 × 1 모델
        total_successes=15,
        total_failures=0,
        execution_time_seconds=120.5,
        output_directory="~/Library/selvage-eval-agent/review_logs",
        success_rate=1.0
    ),
    error_message=None
)
```

**실패 시 (ToolResult)**:
```python
ToolResult(
    success=False,
    data=None,
    error_message="Failed to load meaningful_commits.json: file not found"
)
```

### 에러 처리

1. **파일 관련 에러**:
   - meaningful_commits.json 파일이 존재하지 않음
   - output_dir 디렉토리 생성 실패
   - 권한 부족으로 파일 쓰기 실패

2. **Git 관련 에러**:
   - 저장소 경로가 유효하지 않음
   - git checkout 실패 (커밋이 존재하지 않음)
   - git rev-parse 실패 (parent 커밋 없음)

3. **Selvage 실행 에러**:
   - selvage 바이너리를 찾을 수 없음
   - selvage review 명령어 실행 실패
   - 타임아웃 발생

4. **설정 관련 에러**:
   - 잘못된 모델명 지정
   - API 키 누락 또는 잘못됨

## 필요한 dataclass 명세

### 리뷰 로그 관련 dataclass

```python
@dataclass
class ModelInfo:
    """모델 정보"""
    provider: str
    name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        return cls(
            provider=data['provider'],
            name=data['name']
        )

@dataclass
class PromptMessage:
    """프롬프트 메시지"""
    role: str  # "system" | "user"
    content: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptMessage':
        return cls(
            role=data['role'],
            content=data['content']
        )

@dataclass
class DiffHunk:
    """diff 블록"""
    header: str
    content: str
    before_code: str
    after_code: str
    start_line_original: int
    line_count_original: int
    start_line_modified: int
    line_count_modified: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiffHunk':
        return cls(
            header=data['header'],
            content=data['content'],
            before_code=data['before_code'],
            after_code=data['after_code'],
            start_line_original=data['start_line_original'],
            line_count_original=data['line_count_original'],
            start_line_modified=data['start_line_modified'],
            line_count_modified=data['line_count_modified']
        )

@dataclass
class ProcessedFile:
    """처리된 파일 정보"""
    filename: str
    file_content: str
    language: str
    additions: int
    deletions: int
    hunks: List[DiffHunk]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'filename': self.filename,
            'file_content': self.file_content,
            'language': self.language,
            'additions': self.additions,
            'deletions': self.deletions,
            'hunks': [hunk.to_dict() for hunk in self.hunks]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedFile':
        return cls(
            filename=data['filename'],
            file_content=data['file_content'],
            language=data['language'],
            additions=data['additions'],
            deletions=data['deletions'],
            hunks=[DiffHunk.from_dict(hunk_data) for hunk_data in data['hunks']]
        )

@dataclass
class ProcessedDiff:
    """처리된 diff 정보"""
    files: List[ProcessedFile]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'files': [file.to_dict() for file in self.files]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedDiff':
        return cls(
            files=[ProcessedFile.from_dict(file_data) for file_data in data['files']]
        )

@dataclass
class ReviewIssue:
    """리뷰 이슈"""
    type: str
    line_number: int
    file: str
    description: str
    suggestion: str
    severity: str  # "info" | "warning" | "error"
    target_code: str
    suggested_code: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewIssue':
        return cls(
            type=data['type'],
            line_number=data['line_number'],
            file=data['file'],
            description=data['description'],
            suggestion=data['suggestion'],
            severity=data['severity'],
            target_code=data['target_code'],
            suggested_code=data['suggested_code']
        )

@dataclass
class ReviewResponse:
    """LLM 리뷰 응답"""
    issues: List[ReviewIssue]
    summary: str
    score: float
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'issues': [issue.to_dict() for issue in self.issues],
            'summary': self.summary,
            'score': self.score,
            'recommendations': self.recommendations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewResponse':
        return cls(
            issues=[ReviewIssue.from_dict(issue_data) for issue_data in data['issues']],
            summary=data['summary'],
            score=data['score'],
            recommendations=data['recommendations']
        )

@dataclass
class ReviewRequest:
    """리뷰 요청 데이터"""
    diff_content: str
    processed_diff: ProcessedDiff
    file_paths: List[str]
    use_full_context: bool
    model: str
    repo_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'diff_content': self.diff_content,
            'processed_diff': self.processed_diff.to_dict(),
            'file_paths': self.file_paths,
            'use_full_context': self.use_full_context,
            'model': self.model,
            'repo_path': self.repo_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewRequest':
        return cls(
            diff_content=data['diff_content'],
            processed_diff=ProcessedDiff.from_dict(data['processed_diff']),
            file_paths=data['file_paths'],
            use_full_context=data['use_full_context'],
            model=data['model'],
            repo_path=data['repo_path']
        )

@dataclass
class ReviewLog:
    """리뷰 로그 (review_log.json 구조)"""
    id: str
    model: ModelInfo
    created_at: datetime
    prompt: List[PromptMessage]
    review_request: ReviewRequest
    review_response: Optional[ReviewResponse]
    status: str  # "SUCCESS" | "FAILED"
    error: Optional[str]
    prompt_version: str
    repo_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            'id': self.id,
            'model': self.model.to_dict(),
            'created_at': self.created_at.isoformat(),
            'prompt': [msg.to_dict() for msg in self.prompt],
            'review_request': self.review_request.to_dict(),
            'review_response': self.review_response.to_dict() if self.review_response else None,
            'status': self.status,
            'error': self.error,
            'prompt_version': self.prompt_version,
            'repo_path': self.repo_path
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewLog':
        return cls(
            id=data['id'],
            model=ModelInfo.from_dict(data['model']),
            created_at=datetime.fromisoformat(data['created_at']),
            prompt=[PromptMessage.from_dict(msg_data) for msg_data in data['prompt']],
            review_request=ReviewRequest.from_dict(data['review_request']),
            review_response=ReviewResponse.from_dict(data['review_response']) if data['review_response'] else None,
            status=data['status'],
            error=data['error'],
            prompt_version=data['prompt_version'],
            repo_path=data['repo_path']
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> 'ReviewLog':
        """JSON 파일에서 ReviewLog 객체 생성"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_to_json(self, filepath: str) -> None:
        """JSON 파일로 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
```

### Tool 결과용 dataclass

```python
@dataclass
class ReviewExecutionSummary:
    """리뷰 실행 요약 (ToolResult.data 용)"""
    total_commits_reviewed: int
    total_reviews_executed: int
    total_successes: int
    total_failures: int
    execution_time_seconds: float
    output_directory: str
    success_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewExecutionSummary':
        return cls(**data)
    
    @property
    def summary_message(self) -> str:
        """실행 요약 메시지"""
        return (f"리뷰 완료: {self.total_commits_reviewed}개 커밋, "
                f"{self.total_reviews_executed}개 리뷰 ({self.success_rate:.1%} 성공)")
```

## Tool 등록 및 사용 방법

### ToolGenerator에 등록

```python
# src/selvage_eval/tools/tool_generator.py
from .review_executor_tool import ReviewExecutorTool

class ToolGenerator:
    def generate_tool(self, tool_name: str, params: Dict[str, Any]) -> Tool:
        if tool_name == "read_file":
            return ReadFileTool()
        elif tool_name == "write_file":
            return WriteFileTool()
        elif tool_name == "file_exists":
            return FileExistsTool()
        elif tool_name == "execute_safe_command":
            return ExecuteSafeCommandTool()
        elif tool_name == "list_directory":
            return ListDirectoryTool()
        elif tool_name == "execute_reviews":  # 새로 추가
            return ReviewExecutorTool()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
```

### 사용 예시

```python
from selvage_eval.tools.tool_executor import ToolExecutor

# ToolExecutor 인스턴스 생성
tool_executor = ToolExecutor()

# 리뷰 실행
result = tool_executor.execute_tool_call(
    "execute_reviews",
    {
        "meaningful_commits_path": "/path/to/meaningful_commits.json",
        "output_dir": "~/Library/selvage-eval-agent/review_logs",
        "model": "gemini-2.5-pro"
    }
)

if result.success:
    summary = result.data  # ReviewExecutionSummary 객체
    print(summary.summary_message)
    print(f"성공률: {summary.success_rate:.1%}")
else:
    print(f"리뷰 실행 실패: {result.error_message}")
```

### 파일 구조

구현 완료 후 예상되는 파일 구조:

```
src/selvage_eval/
├── tools/
│   ├── __init__.py
│   ├── tool.py
│   ├── tool_executor.py
│   ├── tool_generator.py           # ReviewExecutorTool 등록
│   ├── tool_result.py
│   ├── execute_safe_command_tool.py
│   ├── file_exists_tool.py
│   ├── list_directory_tool.py
│   ├── read_file_tool.py
│   ├── write_file_tool.py
│   └── review_executor_tool.py     # 새로 구현할 파일
└── review_execution.py             # dataclass들 정의
```

### 구현 시 주의사항

1. **비동기 처리**: 다중 모델 병렬 실행을 위해 asyncio 사용
2. **에러 복구**: Git checkout 실패 시 원래 브랜치로 복원
3. **타임아웃 처리**: 각 명령어별 적절한 타임아웃 설정
4. **로그 저장**: 실행 과정의 상세 로그 저장
5. **진행률 표시**: 대량 커밋 처리 시 진행률 출력
6. **메모리 관리**: 대용량 diff 처리 시 메모리 사용량 최적화
