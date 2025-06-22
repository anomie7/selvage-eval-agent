# Selvage 평가 에이전트 - 핵심 명세

## 개요

AI 기반 코드 리뷰 도구인 Selvage를 평가하는 현대적 에이전트입니다. LLM 기반 쿼리 분석과 범용 도구 체계를 통해 유연하고 안전한 평가 환경을 제공합니다.

## 구현 참조 문서

이 명세서의 구체적인 구현은 다음 문서들을 참조하세요:

### 아키텍처 및 설계
- [에이전트 아키텍처](../architecture/agent-architecture.md) - ReAct 패턴, 대화형 인터페이스
- [현대적 도구 체계](../architecture/modern-agent-tools.md) - 범용 도구, 안전 제약
- [프롬프트 및 컨텍스트](../architecture/prompts-and-context.md) - LLM 프롬프트, 쿼리 분석

### 구현 가이드
- [핵심 구현](../implementation/core-implementation.md) - SelvageEvaluationAgent 클래스
- [도구 구현](../implementation/tool-implementations.md) - 범용 도구 클래스들
- [상태 관리](../implementation/state-management.md) - WorkingMemory, SessionState

### 배포 및 설정
- [설정 및 배포](../deployment/configuration-deployment.md) - 설정 파일, 환경 설정

## 핵심 특징

### 1. 현대적 에이전트 패턴
- **LLM 기반 쿼리 분석**: regex 패턴 대신 LLM이 사용자 의도 파악
- **범용 도구 체계**: 특수 도구 대신 `read_file`, `execute_safe_command` 활용
- **구조화된 응답**: ExecutionPlan으로 체계적인 계획 관리

### 2. 대화형 인터페이스
- Phase 관련 작업: "Phase 1 상태 확인해줘", "Phase 2 실행해줘"
- 데이터 조회: "cline 저장소 commit 목록 보여줘"
- 실행 요청: "deepeval 실행해줘"
- 결과 분석: "모델별 성능 비교해줘"

### 3. 안전성 보장
- 명령어 화이트리스트: jq, grep, git (읽기 전용) 등만 허용
- 경로 제한: 평가 결과 디렉토리와 지정된 저장소만 접근
- 금지 패턴: rm, chmod, curl 등 위험한 명령어 차단

## 핵심 클래스 구조

```python
class SelvageEvaluationAgent:
    """단일 에이전트로 대화형/자동 실행 모드 지원"""
    
    # 대화형 모드
    async def handle_user_message(self, message: str) -> str
    async def plan_execution(self, user_query: str) -> ExecutionPlan
    async def generate_response(self, user_query: str, plan: ExecutionPlan, tool_results: List[Dict]) -> str
    
    # 자동 실행 모드  
    async def execute_evaluation(self) -> EvaluationReport
    async def _analyze_current_state(self) -> Dict[str, Any]
    async def _decide_next_action(self, current_state: Dict[str, Any]) -> str
    async def _execute_action(self, action: str, current_state: Dict[str, Any]) -> Dict[str, Any]
```

## 도구 체계

### 범용 도구 (Claude Code 패턴)
- `read_file`: JSON, 로그, 설정 파일 읽기
- `write_file`: 결과 저장, 임시 파일 생성
- `execute_safe_command`: 제한된 터미널 명령 실행
- `list_directory`: 안전한 디렉토리 탐색

### 사용 예시
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
        "command": "jq '.commits[] | select(.repository==\"cline\")' meaningful_commits.json"
      },
      "rationale": "cline 저장소 커밋만 필터링"
    }
  ]
}
```

## 4단계 워크플로우

### Phase 1 - Commit Collection
- **목적**: 평가 가치가 높은 커밋 식별 및 선별
- **구현**: [02-commit-collection-and-review-execution.md](02-commit-collection-and-review-execution.md)

### Phase 2 - Review Execution  
- **목적**: 선별된 커밋에 대해 다중 모델로 Selvage 리뷰 실행
- **구현**: [02-commit-collection-and-review-execution.md](02-commit-collection-and-review-execution.md)

### Phase 3 - DeepEval Conversion
- **목적**: 리뷰 결과를 DeepEval 형식으로 변환 및 평가
- **구현**: [03-evaluation-conversion-and-analysis.md](03-evaluation-conversion-and-analysis.md)

### Phase 4 - Analysis & Insights
- **목적**: 통계 분석을 통한 actionable insights 도출
- **구현**: [03-evaluation-conversion-and-analysis.md](03-evaluation-conversion-and-analysis.md)

## 프로젝트 파일 구조

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

## 구현 요구사항

### 기본 요구사항
- **Python 3.10+** (타입 힌팅 필수)
- **Google 스타일 독스트링** (한국어 주석)
- **PEP 8 준수**
- **비동기 처리** (다중 모델 병렬 평가)

### 핵심 의존성
- `deepeval` - LLM 평가 프레임워크
- `pytest` - 테스트 프레임워크
- subprocess 실행 및 데이터 처리를 위한 표준 라이브러리

### 환경 설정
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key" 
export GEMINI_API_KEY="your-key"
```

## 다음 단계

1. **Phase 1-2 구현**: [02-commit-collection-and-review-execution.md](02-commit-collection-and-review-execution.md)
2. **Phase 3-4 구현**: [03-evaluation-conversion-and-analysis.md](03-evaluation-conversion-and-analysis.md)
3. **설정 및 배포**: [../deployment/configuration-deployment.md](../deployment/configuration-deployment.md)