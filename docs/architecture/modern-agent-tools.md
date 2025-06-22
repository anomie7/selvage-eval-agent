# 현대적 에이전트 도구 체계

Claude Code, Cursor와 같은 현대적 에이전트 패턴을 적용하여 **범용 도구 + 적절한 제약** 방식을 사용합니다.

## 핵심 범용 도구

**🔧 핵심 범용 도구 (모든 작업에 사용)**
- `read_file`: 안전한 파일 읽기 (평가 결과 디렉토리 내에서만)
- `write_file`: 안전한 파일 쓰기 (결과 저장용)
- `execute_safe_command`: 제한된 안전 명령어 실행
- `list_directory`: 디렉토리 탐색 (허용된 경로 내에서만)

## 프로젝트 파일 구조

**📂 프로젝트 파일 구조 (LLM이 숙지해야 할 컨텍스트)**
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

**🛡️ 안전 제약사항 (execute_safe_command용)**

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

**🎯 실제 사용 예시**

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
        """매개변수 스키마 (JSON Schema 형식)"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """도구 실행"""
        pass
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """매개변수 유효성 검증"""
        # JSON Schema 기반 검증 구현
        pass
```

## 범용 도구 vs 특수 도구

### ❌ 특수 도구 방식 (기존):
```python
# 매번 새 도구 필요
commit_data_query()
commit_data_query_with_filters()
commit_data_query_by_date()
review_result_query()
review_result_by_model()
```

### ✅ 범용 도구 방식 (현대적):
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