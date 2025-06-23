# 현대적 에이전트 도구 체계

Claude Code, Cursor와 같은 현대적 에이전트 패턴을 적용하여 **범용 도구 + 적절한 제약** 방식을 사용합니다.

## 동기 처리 기반 도구 체계

**동기적 실행의 장점:**
- **단순성**: async/await 복잡성 제거
- **디버깅**: 명확한 스택 트레이스
- **성능**: I/O 병목이 없는 환경에서 오버헤드 감소

## 핵심 범용 도구

**[TOOLS] 핵심 범용 도구 (모든 작업에 사용)**
- `read_file`: 안전한 파일 읽기 (평가 결과 디렉토리 내에서만)
- `write_file`: 안전한 파일 쓰기 (결과 저장용)
- `execute_safe_command`: 제한된 안전 명령어 실행
- `list_directory`: 디렉토리 탐색 (허용된 경로 내에서만)

## 프로젝트 파일 구조

**[STRUCTURE] 프로젝트 파일 구조 (LLM이 숙지해야 할 컨텍스트)**
```
selvage-eval-results/
├── session_metadata.json          # 세션 정보 및 설정
├── meaningful_commits.json        # Phase 1: 선별된 커밋 목록
├── review_logs/                   # Phase 2: 리뷰 실행 결과
│   ├── {repo_name}/
│   │   ├── {commit_hash}/
│   │   │   ├── {model_name}_review.json
│   │   │   └── {model_name}_error.log
├── evaluations/                   # Phase 3: DeepEval 결과
│   ├── deepeval_testcases.json   # 변환된 테스트케이스
│   ├── evaluation_results.json   # 평가 결과
│   └── metrics_breakdown.json    # 메트릭별 상세 분석
└── analysis/                     # Phase 4: 최종 분석
    ├── statistical_summary.json  # 통계 요약
    ├── model_comparison.json     # 모델별 성능 비교
    └── insights_report.json      # 도출된 인사이트
```

## 안전 제약사항

**[SECURITY] 안전 제약사항 (execute_safe_command용)**

### 허용된 명령어:
```bash
# 데이터 조회 및 분석
jq, grep, find, ls, cat, head, tail, wc
git log, git show, git diff (읽기 전용)

# 파일 처리
cp, mv (결과 디렉토리 내에서만)
mkdir, touch (결과 디렉토리 내에서만)

# Selvage 실행
/Users/demin_coder/.local/bin/selvage (subprocess로만)
```

### 금지된 작업:
```bash
# 절대 금지
rm, rmdir, delete (원본 저장소 손상 방지)
chmod, chown (권한 변경 금지)
curl, wget (외부 네트워크 금지)
sudo, su (권한 상승 금지)

# 원본 저장소 쓰기 금지
git commit, git push, git merge
echo >, sed -i, awk (파일 수정 명령)
```

## 실제 사용 예시

**[EXAMPLE] 실제 사용 예시**

사용자: "cline 저장소에서 최근 일주일 내 fix 관련 커밋만 보여줘"

LLM 계획:
```json
{
  "tool_calls": [
    {
      "tool": "read_file",
      "params": {"file_path": "./selvage-eval-results/meaningful_commits.json"},
      "rationale": "저장된 커밋 데이터 읽기"
    },
    {
      "tool": "execute_safe_command", 
      "params": {
        "command": "jq '.commits[] | select(.repository==\"cline\" and (.message | contains(\"fix\")) and (.date | fromdateiso8601 > (now - 7*24*3600)))' ./selvage-eval-results/meaningful_commits.json"
      },
      "rationale": "cline 저장소에서 최근 일주일 내 fix 관련 커밋 필터링"
    }
  ]
}
```

이 방식으로 특수한 도구 없이도 복잡한 쿼리를 유연하게 처리할 수 있습니다.

## 표준화된 도구 인터페이스

모든 도구는 단일 에이전트가 사용하는 유틸리티로서 표준화된 인터페이스를 구현합니다:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ToolResult:
    """도구 실행 결과"""
    success: bool
    data: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

class Tool(ABC):
    """
    모든 도구의 기본 인터페이스
    각 도구는 고유한 명시적 파라미터 시그니처를 가집니다.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """도구 이름"""
        pass
    
    @property 
    @abstractmethod
    def description(self) -> str:
        """도구 설명"""
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """매개변수 스키마 (타입 힌트로부터 자동 생성)"""
        pass
    
    # execute 메서드는 각 도구별로 고유한 파라미터 시그니처로 구현
    # 예: def execute(self, command: str, timeout: int = 60) -> ToolResult
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """매개변수 유효성 검증"""
        # 타입 힌트 기반 검증으로 대체됨
        return params is not None
```

## 타입 힌트 기반 자동 스키마 생성

### 명시적 파라미터의 장점

**기존 방식 (문제점):**
```python
def execute(self, **kwargs) -> ToolResult:
    command = kwargs["command"]  # 타입 불명, 오타 가능
    timeout = kwargs.get("timeout", 60)  # IDE 지원 없음
```

**현재 방식 (개선됨):**
```python  
def execute(self, command: str, cwd: Optional[str] = None, 
           timeout: int = 60, capture_output: bool = True) -> ToolResult:
    """제한된 안전 명령어를 실행합니다
    
    Args:
        command: 실행할 터미널 명령어
        cwd: 명령어 실행 디렉토리 (선택사항)
        timeout: 타임아웃 (초, 기본값: 60)
        capture_output: 출력 캡처 여부 (기본값: true)
    """
```

### 자동 스키마 생성

`generate_parameters_schema_from_hints()` 함수가 타입 힌트로부터 JSON Schema를 자동 생성:

```python
@property
def parameters_schema(self) -> Dict[str, Any]:
    return generate_parameters_schema_from_hints(self.execute)

# 자동 생성 결과:
# {
#   "type": "object",
#   "properties": {
#     "command": {"type": "string", "description": "command 파라미터"},
#     "cwd": {"type": "string", "description": "cwd 파라미터", "default": None},
#     "timeout": {"type": "integer", "description": "timeout 파라미터", "default": 60},
#     "capture_output": {"type": "boolean", "description": "capture_output 파라미터", "default": True}
#   },
#   "required": ["command"]
# }
```

### LLM Tool Calls 실행

`ToolExecutor` 클래스가 LLM tool_calls를 파싱하여 명시적 파라미터로 변환:

```python
# LLM이 반환한 tool_calls
tool_calls = [
    {
        "tool": "execute_safe_command",
        "parameters": {
            "command": "jq '.commits[] | select(.repository==\"cline\")' data.json",
            "timeout": 30
        }
    }
]

# ToolExecutor가 자동으로 타입 체크 및 변환 후 실행
executor = ToolExecutor()
results = executor.execute_tool_call(tool_calls)
```

## 범용 도구 방식

```json
{
  "tool_calls": [
    {
      "tool": "read_file", 
      "params": {"file_path": "./selvage-eval-results/meaningful_commits.json"},
      "rationale": "커밋 데이터 파일을 읽어서 분석"
    },
    {
      "tool": "execute_safe_command",
      "params": {"command": "jq '.commits[] | select(.repository==\"cline\")' meaningful_commits.json"},
      "rationale": "cline 저장소 커밋만 필터링"
    }
  ]
}
```

### 장점:
1. **극도의 유연성**: 예상치 못한 요청도 처리 가능
2. **LLM 창의성 활용**: 새로운 방식으로 문제 해결
3. **확장성**: 새 도구 개발 없이 기능 확장