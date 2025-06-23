# Selvage 평가 에이전트 - 아키텍처 및 설정

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

## Single Agent 아키텍처 패러다임

### ReAct (Reasoning + Acting) 패턴
Selvage 평가 에이전트는 단일 에이전트가 ReAct 패턴으로 두 가지 모드를 지원합니다:
1. **자동 실행 모드**: 4단계 워크플로우 순차 실행
2. **대화형 모드**: 사용자 요청에 따른 동적 액션 실행

### Interactive Agent Interface

에이전트는 터미널에서 사용자와 실시간으로 상호작용하며 다음과 같은 요청을 처리합니다:

#### 지원하는 상호작용 유형
1. **Phase 관련 작업**
   - "Phase 1 상태 확인해줘"
   - "Phase 2 실행해줘" 
   - "어떤 단계까지 완료됐어?"

2. **저장된 Commit 관련 질문**
   - "cline 저장소 commit 목록 보여줘"
   - "선별된 commit들의 상세 정보는?"
   - "commit scoring 결과는?"

3. **리뷰 결과 데이터 질문**
   - "gemini-2.5-pro 리뷰 결과 보여줘"
   - "실패한 리뷰들은 어떤 것들이야?"
   - "모델별 리뷰 완료 현황은?"

4. **LLM Eval 실행 요청**
   - "deepeval 실행해줘"
   - "특정 모델 결과만 평가해줘"
   - "평가 재실행해줘"

5. **LLM Eval Result 분석**
   - "평가 결과 요약해줘"
   - "모델별 성능 비교해줘"
   - "어떤 모델이 가장 좋아?"

#### LLM-Based Query Analysis System

현대적 에이전트 패턴을 적용하여 LLM이 사용자 쿼리를 분석하고 실행 계획을 수립합니다:

```python
QUERY_ANALYSIS_PROMPT = """
# ROLE
당신은 Selvage 평가 에이전트의 Query Planner입니다.
사용자 쿼리를 분석하여 실행 계획을 수립하고 필요한 도구들을 식별합니다.

# QUERY EXAMPLES
다음과 같은 다양한 사용자 질문들을 처리할 수 있습니다:

**상태 조회:**
- "Phase 1 완료됐어?"
- "현재 어떤 상황이야?"
- "cline 저장소 진행 상황은?"

**데이터 조회:**
- "선별된 커밋 목록 보여줘"
- "gemini 모델 리뷰 결과는?"
- "실패한 평가들 알려줘"

**실행 요청:**
- "Phase 2 실행해줘"
- "deepeval 돌려줘"
- "특정 저장소만 다시 평가해줘"

**분석 요청:**
- "모델별 성능 비교해줘"
- "어떤 에러가 많이 발생했어?"
- "결과를 차트로 보여줄 수 있어?"

# STRICT CONSTRAINTS
다음 작업들은 절대 수행하지 마세요:

**[FORBIDDEN] 절대 금지:**
- 원본 저장소 파일 수정/삭제
- selvage-deprecated 저장소 쓰기 작업
- 시스템 파일 접근
- API 키나 민감한 정보 노출
- 평가 결과 데이터 조작
- 네트워크 외부 연결 (승인되지 않은)

# PROJECT FILE STRUCTURE
현재 작업 디렉토리 구조를 숙지하고 적절한 파일 경로를 사용하세요:

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

# AVAILABLE TOOLS
{available_tools}

# COMMON COMMANDS FOR DATA ANALYSIS
다음과 같은 명령어들을 활용하여 데이터를 분석할 수 있습니다:

```bash
# JSON 데이터 쿼리
jq '.commits[] | select(.repository=="cline")' meaningful_commits.json
jq '.evaluations | group_by(.model) | map({model: .[0].model, avg_score: (map(.score) | add/length)})' evaluation_results.json

# 파일 검색 및 분석
find ./review_logs -name "*_error.log" -exec wc -l {} +
grep -r "success.*true" ./review_logs/ | wc -l

# 로그 분석
cat ./review_logs/cline/abc123/gemini-2.5-pro_review.json | jq '.review_content'
tail -f ./review_logs/*/*/error.log  # 실시간 에러 모니터링
```

# TASK
사용자 쿼리를 분석하고 안전하고 효과적인 실행 계획을 JSON으로 제공하세요.

Response format:
{{
  "intent_summary": "사용자 의도 요약",
  "confidence": 0.0-1.0,
  "parameters": {{}},
  "tool_calls": [
    {{"tool": "tool_name", "params": {{}}, "rationale": "이 도구를 선택한 이유"}}
  ],
  "safety_check": "안전성 검토 결과",
  "expected_outcome": "예상 결과"
}}
"""

@dataclass
class ExecutionPlan:
    """LLM이 생성한 실행 계획"""
    intent_summary: str
    confidence: float
    parameters: Dict[str, Any]
    tool_calls: List[ToolCall]
    safety_check: str
    expected_outcome: str
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ExecutionPlan':
        data = json.loads(json_str)
        return cls(
            intent_summary=data["intent_summary"],
            confidence=data["confidence"],
            parameters=data["parameters"],
            tool_calls=[ToolCall(**tc) for tc in data["tool_calls"]],
            safety_check=data["safety_check"],
            expected_outcome=data["expected_outcome"]
        )

@dataclass
class ToolCall:
    tool: str
    params: Dict[str, Any]
    rationale: str

```python
class SelvageEvaluationAgent:
    """
    단일 에이전트로 전체 평가 프로세스를 관리하는 Selvage 평가 에이전트
    대화형 모드와 자동 실행 모드를 모두 지원
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.tools = self._initialize_tools()
        self.working_memory = WorkingMemory()
        self.session_state = SessionState()
        self.current_phase = None
        self.llm = self._initialize_llm()  # Query Planning용 LLM
        self.is_interactive_mode = False
    
    async def handle_user_message(self, message: str) -> str:
        """
        현대적 에이전트 패턴으로 사용자 메시지 처리
        
        Flow:
        1. LLM이 쿼리 분석 및 실행 계획 수립
        2. 계획에 따라 도구들 실행  
        3. 도구 결과를 바탕으로 LLM이 최종 응답 생성
        """
        try:
            # 1. LLM 기반 쿼리 분석 및 실행 계획 수립
            plan = await self.plan_execution(message)
            
            # 2. 안전성 검증
            if not self._validate_plan_safety(plan):
                return f"요청하신 작업은 보안상 실행할 수 없습니다: {plan.safety_check}"
            
            # 3. 계획에 따라 도구들 실행
            tool_results = []
            for tool_call in plan.tool_calls:
                result = await self.execute_tool(tool_call.tool, tool_call.params)
                tool_results.append({
                    "tool": tool_call.tool,
                    "result": result,
                    "rationale": tool_call.rationale
                })
            
            # 4. 도구 결과를 바탕으로 LLM이 최종 응답 생성
            return await self.generate_response(message, plan, tool_results)
            
        except Exception as e:
            return f"메시지 처리 중 오류가 발생했습니다: {str(e)}"
    
    async def plan_execution(self, user_query: str) -> ExecutionPlan:
        """LLM을 통한 쿼리 분석 및 실행 계획 수립"""
        
        # 현재 상태 정보 수집
        current_state = await self._analyze_current_state()
        
        prompt = QUERY_ANALYSIS_PROMPT.format(
            available_tools=self._get_available_tools_description()
        )
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"""
현재 상태: {json.dumps(current_state, ensure_ascii=False, indent=2)}

사용자 쿼리: {user_query}
            """}
        ]
        
        response = await self.llm.query(
            messages=messages,
            response_format="json",
            max_tokens=1000
        )
        
        return ExecutionPlan.from_json(response)
    
    async def generate_response(self, user_query: str, plan: ExecutionPlan, tool_results: List[Dict]) -> str:
        """도구 실행 결과를 바탕으로 사용자에게 제공할 최종 응답 생성"""
        
        response_prompt = f"""
# ROLE  
사용자에게 도구 실행 결과를 바탕으로 명확하고 유용한 답변을 제공하는 어시스턴트입니다.

# CONTEXT
사용자 질문: {user_query}
의도 분석: {plan.intent_summary}
예상 결과: {plan.expected_outcome}

# TOOL EXECUTION RESULTS
{json.dumps(tool_results, ensure_ascii=False, indent=2)}

# TASK
도구 실행 결과를 바탕으로 사용자가 이해하기 쉬운 답변을 생성하세요.
- 핵심 정보를 명확히 전달
- 필요시 표나 리스트 형태로 구조화  
- 다음 단계 제안 (해당되는 경우)
        """
        
        response = await self.llm.query(
            messages=[{"role": "user", "content": response_prompt}],
            max_tokens=1500
        )
        
        return response
    
    def _validate_plan_safety(self, plan: ExecutionPlan) -> bool:
        """실행 계획의 안전성 검증"""
        
        # 금지된 도구 확인
        forbidden_tools = ["delete_file", "modify_repository", "system_command"]
        for tool_call in plan.tool_calls:
            if tool_call.tool in forbidden_tools:
                return False
        
        # selvage-deprecated 쓰기 작업 확인
        for tool_call in plan.tool_calls:
            if "selvage-deprecated" in str(tool_call.params) and tool_call.tool.startswith("write"):
                return False
        
        return True
    
    def _get_available_tools_description(self) -> str:
        """사용 가능한 도구들의 설명 반환"""
        descriptions = []
        for tool_name, tool in self.tools.items():
            descriptions.append(f"- {tool_name}: {tool.description}")
        return "\n".join(descriptions)
    
    async def execute_evaluation(self) -> EvaluationReport:
        """
        에이전트 방식으로 평가 프로세스 실행
        상태를 파악하고 동적으로 다음 행동 결정
        """
        session_id = self._generate_session_id()
        
        while True:
            # 현재 상태 분석
            current_state = await self._analyze_current_state()
            
            # 다음 행동 결정
            next_action = await self._decide_next_action(current_state)
            
            if next_action == "COMPLETE":
                break
                
            # 행동 실행
            action_result = await self._execute_action(next_action, current_state)
            
            # 결과 저장 및 상태 업데이트
            await self._update_state(action_result)
        
        # 최종 보고서 생성
        return await self._generate_final_report(session_id)
    
    async def _analyze_current_state(self) -> Dict[str, Any]:
        """
        현재 상태를 분석하여 어떤 단계까지 완료되었는지 파악
        """
        state = {
            "session_id": self.session_state.session_id,
            "completed_phases": [],
            "available_data": {},
            "next_required_phase": None
        }
        
        # Phase 1: 커밋 수집 상태 확인
        commits_file = f"{self.config.output_dir}/meaningful_commits.json"
        if await self.tools["file_exists"].execute(file_path=commits_file):
            state["completed_phases"].append("commit_collection")
            commit_data = await self.tools["read_file"].execute(
                file_path=commits_file, as_json=True
            )
            state["available_data"]["commits"] = commit_data.data["content"]
        
        # Phase 2: 리뷰 실행 상태 확인
        review_logs_exist = await self._check_review_logs_exist()
        if review_logs_exist:
            state["completed_phases"].append("review_execution")
            state["available_data"]["reviews"] = await self._scan_review_logs()
        
        # Phase 3: DeepEval 결과 확인
        eval_results_exist = await self._check_evaluation_results_exist()
        if eval_results_exist:
            state["completed_phases"].append("deepeval_conversion")
            state["available_data"]["evaluations"] = await self._load_evaluation_results()
        
        # 다음 필요한 단계 결정
        if "commit_collection" not in state["completed_phases"]:
            state["next_required_phase"] = "commit_collection"
        elif "review_execution" not in state["completed_phases"]:
            state["next_required_phase"] = "review_execution"
        elif "deepeval_conversion" not in state["completed_phases"]:
            state["next_required_phase"] = "deepeval_conversion"
        elif "analysis" not in state["completed_phases"]:
            state["next_required_phase"] = "analysis"
        else:
            state["next_required_phase"] = "complete"
        
        return state
    
    async def _decide_next_action(self, current_state: Dict[str, Any]) -> str:
        """
        현재 상태를 기반으로 다음 행동을 결정
        """
        next_phase = current_state["next_required_phase"]
        
        if next_phase == "complete":
            return "COMPLETE"
        
        # skip 로직 확인
        if self.config.workflow.skip_existing:
            if next_phase == "commit_collection" and current_state["available_data"].get("commits"):
                return "SKIP_TO_REVIEW"
            elif next_phase == "review_execution" and current_state["available_data"].get("reviews"):
                return "SKIP_TO_EVALUATION"
            elif next_phase == "deepeval_conversion" and current_state["available_data"].get("evaluations"):
                return "SKIP_TO_ANALYSIS"
        
        return f"EXECUTE_{next_phase.upper()}"
    
    async def _execute_action(self, action: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        결정된 행동을 실행
        
        구체적인 Phase별 구현은 다음 문서들을 참조:
        - Phase 1-2: docs/specs/02-commit-collection-and-review-execution.md
        - Phase 3-4: docs/specs/03-evaluation-conversion-and-analysis.md
        """
        if action == "EXECUTE_COMMIT_COLLECTION":
            # Phase 1 구현은 02-commit-collection-and-review-execution.md 참조
            return await self._execute_phase1_commit_collection()
        elif action == "EXECUTE_REVIEW_EXECUTION":
            commits = current_state["available_data"]["commits"]
            # Phase 2 구현은 02-commit-collection-and-review-execution.md 참조
            return await self._execute_phase2_review_execution(commits)
        elif action == "EXECUTE_DEEPEVAL_CONVERSION":
            reviews = current_state["available_data"]["reviews"]
            # Phase 3 구현은 03-evaluation-conversion-and-analysis.md 참조
            return await self._execute_phase3_deepeval_conversion(reviews)
        elif action == "EXECUTE_ANALYSIS":
            evaluations = current_state["available_data"]["evaluations"]
            # Phase 4 구현은 03-evaluation-conversion-and-analysis.md 참조
            return await self._execute_phase4_analysis(evaluations)
        elif action.startswith("SKIP_TO_"):
            # 스킵 액션 처리
            return {"action": action, "skipped": True}
        else:
            raise ValueError(f"Unknown action: {action}")
    
    # Phase별 구체적인 구현 메서드들은 해당 문서에서 구현:
    # - Phase 1-2: docs/specs/02-commit-collection-and-review-execution.md
    # - Phase 3-4: docs/specs/03-evaluation-conversion-and-analysis.md

## Tool 정의 및 분류

### Tool 분류 및 Interface 정의

도구는 크게 **기본 유틸리티 도구**와 **Phase별 전용 도구**로 분류됩니다:

#### 현대적 에이전트 도구 체계

Claude Code, Cursor와 같은 현대적 에이전트 패턴을 적용하여 **범용 도구 + 적절한 제약** 방식을 사용합니다:

**[TOOLS] 핵심 범용 도구 (모든 작업에 사용)**
- `read_file`: 안전한 파일 읽기 (평가 결과 디렉토리 내에서만)
- `write_file`: 안전한 파일 쓰기 (결과 저장용)
- `execute_safe_command`: 제한된 안전 명령어 실행
- `list_directory`: 디렉토리 탐색 (허용된 경로 내에서만)

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

**[SECURITY] 안전 제약사항 (execute_safe_command용)**

허용된 명령어:
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

금지된 작업:
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

### 기본 유틸리티 도구 구현

**ExecuteSafeCommandTool** - 제한된 안전 명령어 실행
```python
class ExecuteSafeCommandTool(Tool):
    """제한된 안전 명령어 실행 도구 (현대적 에이전트 패턴)"""
    
    def __init__(self):
        self.allowed_commands = {
            'jq', 'grep', 'find', 'ls', 'cat', 'head', 'tail', 'wc',
            'git', 'cp', 'mv', 'mkdir', 'touch'
        }
        self.allowed_paths = [
            './selvage-eval-results/',
            '/Users/demin_coder/Dev/cline',
            '/Users/demin_coder/Dev/selvage-deprecated',
            '/Users/demin_coder/Dev/ecommerce-microservices', 
            '/Users/demin_coder/Dev/kotlin-realworld'
        ]
        self.forbidden_patterns = [
            r'rm\s+', r'rmdir\s+', r'delete\s+',
            r'chmod\s+', r'chown\s+',
            r'curl\s+', r'wget\s+',
            r'sudo\s+', r'su\s+',
            r'echo\s+.*>', r'sed\s+-i', r'>\s*'
        ]
    
    @property
    def name(self) -> str:
        return "execute_safe_command"
    
    @property
    def description(self) -> str:
        return "제한된 안전 명령어를 실행합니다. 데이터 조회, 분석, 읽기 전용 Git 작업만 허용"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string", 
                    "description": "실행할 터미널 명령어"
                },
                "cwd": {
                    "type": "string", 
                    "description": "명령어 실행 디렉토리 (선택사항)"
                },
                "timeout": {
                    "type": "integer", 
                    "description": "타임아웃 (초, 기본값: 60)"
                },
                "capture_output": {
                    "type": "boolean", 
                    "description": "출력 캡처 여부 (기본값: true)"
                }
            },
            "required": ["command"]
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        command = kwargs["command"]
        cwd = kwargs.get("cwd", None)
        timeout = kwargs.get("timeout", 60)
        capture_output = kwargs.get("capture_output", True)
        
        try:
            # 보안을 위한 명령어 검증
            if not self._validate_command_safety(command):
                return ToolResult(
                    success=False,
                    error_message=f"Command blocked by safety filters: {command}"
                )
            
            # 명령어 실행
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
            
            return ToolResult(
                success=process.returncode == 0,
                data={
                    "returncode": process.returncode,
                    "stdout": stdout.decode() if stdout else "",
                    "stderr": stderr.decode() if stderr else "",
                    "command": command
                },
                error_message=stderr.decode() if process.returncode != 0 and stderr else None
            )
            
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error_message=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to execute command: {str(e)}"
            )
    
    def _validate_command_safety(self, command: str) -> bool:
        """현대적 에이전트 패턴의 안전성 검증"""
        import re
        import shlex
        
        # 금지된 패턴 확인
        for pattern in self.forbidden_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False
        
        # 명령어 파싱 및 허용 목록 확인
        try:
            tokens = shlex.split(command)
            if not tokens:
                return False
                
            base_command = tokens[0].split('/')[-1]  # 경로에서 명령어만 추출
            
            if base_command not in self.allowed_commands:
                return False
            
            # 특별 처리: git 명령어는 읽기 전용만 허용
            if base_command == 'git':
                if len(tokens) < 2:
                    return False
                git_subcommand = tokens[1]
                allowed_git_commands = {'log', 'show', 'diff', 'status', 'branch'}
                if git_subcommand not in allowed_git_commands:
                    return False
            
            return True
            
        except ValueError:  # shlex.split 실패
            return False
    
    def _validate_path_access(self, path: str) -> bool:
        """경로 접근 권한 검증"""
        import os
        abs_path = os.path.abspath(path)
        
        for allowed_path in self.allowed_paths:
            if abs_path.startswith(os.path.abspath(allowed_path)):
                return True
        return False
```

**ReadFileTool** - 파일 읽기
```python
class ReadFileTool(Tool):
    """파일 내용 읽기 도구 (모든 Phase에서 사용)"""
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "지정된 파일의 내용을 읽어서 반환합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string", 
                    "description": "읽을 파일의 경로"
                },
                "encoding": {
                    "type": "string", 
                    "description": "파일 인코딩 (기본값: utf-8)"
                },
                "max_size_mb": {
                    "type": "integer", 
                    "description": "최대 파일 크기 (MB, 기본값: 10)"
                },
                "as_json": {
                    "type": "boolean", 
                    "description": "JSON으로 파싱 여부 (기본값: false)"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        file_path = kwargs["file_path"]
        encoding = kwargs.get("encoding", "utf-8")
        max_size_mb = kwargs.get("max_size_mb", 10)
        as_json = kwargs.get("as_json", False)
        
        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                return ToolResult(
                    success=False,
                    error_message=f"File not found: {file_path}"
                )
            
            # 파일 크기 확인
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return ToolResult(
                    success=False,
                    error_message=f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB"
                )
            
            # 파일 읽기
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # JSON 파싱 (필요시)
            if as_json:
                try:
                    content = json.loads(content)
                except json.JSONDecodeError as e:
                    return ToolResult(
                        success=False,
                        error_message=f"Invalid JSON format: {str(e)}"
                    )
            
            return ToolResult(
                success=True,
                data={
                    "content": content,
                    "file_path": file_path,
                    "file_size_bytes": os.path.getsize(file_path),
                    "encoding": encoding
                }
            )
            
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                error_message=f"Unable to decode file with encoding: {encoding}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to read file: {str(e)}"
            )
```

**WriteFileTool** - 파일 쓰기
```python
class WriteFileTool(Tool):
    """파일 쓰기 도구 (결과 저장용)"""
    
    @property
    def name(self) -> str:
        return "write_file"
    
    async def execute(self, **kwargs) -> ToolResult:
        file_path = kwargs["file_path"]
        content = kwargs["content"]
        encoding = kwargs.get("encoding", "utf-8")
        create_dirs = kwargs.get("create_dirs", True)
        as_json = kwargs.get("as_json", False)
        
        try:
            # 디렉토리 생성 (필요시)
            if create_dirs:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # JSON 직렬화 (필요시)
            if as_json:
                content = json.dumps(content, indent=2, ensure_ascii=False)
            
            # 파일 쓰기
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                data={
                    "file_path": file_path,
                    "bytes_written": len(content.encode(encoding)),
                    "encoding": encoding
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to write file: {str(e)}"
            )
```

### Phase별 Tool 구현

**Phase별 Tool들은 각각의 전용 문서로 이동되었습니다:**
- **Phase 1-2 Tools**: `docs/specs/02-commit-collection-and-review-execution.md` 참조
- **Phase 3-4 Tools**: `docs/specs/03-evaluation-conversion-and-analysis.md` 참조



## 단일 에이전트 프롬프트 설계

### Master Agent Prompt
```python
SINGLE_AGENT_PROMPT = """
# ROLE
당신은 Selvage 코드 리뷰 도구를 평가하는 전문 AI 에이전트입니다.
단일 에이전트로서 4단계 워크플로우를 순차적으로 실행하여 체계적이고 정량적인 평가를 수행합니다.

# CAPABILITIES
- 다양한 도구를 사용하여 Git 저장소 분석, 코드 리뷰 실행, 결과 평가 수행
- 통계적 분석을 통한 모델 성능 비교 및 인사이트 도출
- 재현 가능한 평가 환경 구축 및 결과 문서화

# WORKFLOW PHASES
당신은 다음 4단계를 순차적으로 실행합니다:

1. **Phase 1 - Commit Collection**: 
   - 목적: meaningful한 커밋들을 자동 식별 및 배점
   - 사용 도구: git_log, commit_scoring
   - 결과: 평가 가치가 높은 커밋 리스트

2. **Phase 2 - Review Execution**: 
   - 목적: 선별된 커밋에 대해 다중 모델로 Selvage 리뷰 실행
   - 사용 도구: selvage_executor
   - 결과: 모델별 리뷰 결과 로그

3. **Phase 3 - DeepEval Conversion**: 
   - 목적: 리뷰 결과를 DeepEval 형식으로 변환 및 평가
   - 사용 도구: review_log_scanner, deepeval_converter, metric_evaluator
   - 결과: 정량화된 평가 메트릭

4. **Phase 4 - Analysis & Insights**: 
   - 목적: 통계 분석을 통한 actionable insights 도출 (복잡한 추론 필요)
   - 사용 도구: statistical_analysis + AI 추론
   - 결과: 실행 가능한 권장사항 및 인사이트

# PHASE EXECUTION STRATEGY
- Phase 1-3: 주로 도구 호출과 데이터 처리 중심
- Phase 4: AI 추론을 통한 패턴 분석 및 인사이트 도출
- 각 단계의 결과는 다음 단계의 입력으로 사용
- 실패 시 재시도 로직 내장

# DECISION MAKING PRINCIPLES
- **데이터 기반**: 모든 결정은 정량적 데이터에 근거
- **재현성**: 동일 조건에서 동일 결과 보장
- **효율성**: 적절한 도구 선택 및 캐싱 활용
- **신뢰성**: 에러 처리 및 복구 메커니즘 내장

# ERROR HANDLING
- 각 단계에서 실패 시 자동 재시도 (최대 3회)
- 부분 실패 시에도 가능한 결과 수집 및 분석
- 상세한 에러 로깅 및 디버깅 정보 제공

# OUTPUT FORMAT
모든 결과는 JSON 형식으로 구조화하여 제공하며, 
사람이 읽기 쉬운 요약과 함께 제공합니다.

당신의 목표는 Selvage의 성능을 정확하고 공정하게 평가하여 
실제 의사결정에 도움이 되는 인사이트를 제공하는 것입니다.
"""
```

### Phase-Specific Context (프롬프트에 포함될 단계별 컨텍스트)

단일 에이전트가 현재 실행 중인 Phase를 이해할 수 있도록 각 단계별 세부 컨텍스트를 제공합니다:

**Phase 1 Context: Commit Collection**
```python
PHASE1_CONTEXT = """
현재 단계: Phase 1 - Commit Collection

목적: 평가 가치가 높은 의미있는 커밋들을 식별하고 선별

전략:
1. 키워드 기반 1차 필터링 (fix, feature, refactor 포함 / typo, format 제외)
2. 통계 기반 2차 필터링 (파일 수 2-10개, 변경 라인 50+ 기준)
3. 배점 기반 최종 선별 (파일 타입, 변경 규모, 커밋 특성 종합 고려)

사용할 도구: git_log, commit_scoring
예상 결과: commits_per_repo 개수만큼 선별된 고품질 커밋 리스트

실행 단계:
1. 각 저장소별 git_log로 후보 커밋 수집
2. commit_scoring으로 평가 가치 배점
3. 상위 점수 커밋 선별
"""

PHASE2_CONTEXT = """
현재 단계: Phase 2 - Review Execution

목적: 선별된 커밋들에 대해 다중 모델로 Selvage 리뷰 실행

전략:
1. 안전한 커밋 체크아웃 (실행 후 HEAD 복원)
2. 모델별 순차 실행 (동시성 제한)
3. 체계적 결과 저장 (repo/commit/model 구조)

사용할 도구: selvage_executor
예상 결과: 모델별 리뷰 결과 로그 파일들

실행 단계:
1. Phase 1 결과에서 커밋 목록 로드
2. 각 커밋별로 모델별 리뷰 실행
3. 결과 검증 및 구조화된 저장
"""

PHASE3_CONTEXT = """
현재 단계: Phase 3 - DeepEval Conversion

목적: 리뷰 결과를 DeepEval 테스트 케이스로 변환 및 평가

전략:
1. 리뷰 로그 파일 전체 스캔
2. prompt/response 데이터 추출
3. DeepEval 형식 변환
4. 4개 메트릭으로 평가 실행

사용할 도구: review_log_scanner, deepeval_converter, metric_evaluator
평가 메트릭: Correctness, Clarity, Actionability, JsonCorrectness
예상 결과: 정량화된 평가 점수 데이터

실행 단계:
1. 저장된 리뷰 로그 스캔
2. 데이터 추출 및 형식 변환
3. DeepEval 평가 실행
"""

PHASE4_CONTEXT = """
현재 단계: Phase 4 - Analysis & Insights (복잡한 추론 단계)

목적: 평가 결과 종합 분석 및 actionable insights 도출

전략:
1. 통계적 분석으로 기본 패턴 파악
2. AI 추론을 통한 깊이 있는 패턴 분석
3. 실행 가능한 권장사항 생성
4. 의사결정 지원 인사이트 도출

사용할 도구: statistical_analysis + AI 추론 능력
분석 차원: 모델별 성능, 기술스택별 특화, 실패 패턴, 비용 효율성
예상 결과: Executive Summary, 상세 성능 매트릭스, 개선 권장사항

주의: 이 단계는 단순한 도구 호출이 아닌 복잡한 추론과 인사이트 도출이 필요
"""
```

## 단일 에이전트의 Tool 실행 전략

### Phase-Sequential Tool Execution
단일 에이전트가 각 Phase 내에서 도구들을 순차적으로 실행하는 전략:

```python
class SingleAgentToolExecutor:
    """단일 에이전트의 도구 실행 관리"""
    
    def __init__(self, agent: SelvageEvaluationAgent):
        self.agent = agent
        self.retry_count = 3
        self.timeout_seconds = 300
    
    async def execute_phase_tools(self, phase: str, tool_sequence: List[Dict]) -> List[ToolResult]:
        """Phase 내 도구들을 순차 실행"""
        results = []
        
        for tool_config in tool_sequence:
            tool_name = tool_config["name"]
            tool_params = tool_config["params"]
            
            # 재시도 로직 포함 도구 실행
            result = await self._execute_with_retry(
                tool_name=tool_name,
                params=tool_params,
                max_retries=self.retry_count
            )
            
            results.append(result)
            
            # 중요한 도구 실패 시 Phase 중단
            if not result.success and tool_config.get("critical", False):
                raise PhaseExecutionError(f"Critical tool {tool_name} failed in {phase}")
        
        return results
    
    async def _execute_with_retry(self, tool_name: str, params: Dict, max_retries: int) -> ToolResult:
        """재시도 로직이 포함된 도구 실행"""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                tool = self.agent.tools[tool_name]
                result = await asyncio.wait_for(
                    tool.execute(**params),
                    timeout=self.timeout_seconds
                )
                
                if result.success:
                    return result
                    
                last_error = result.error_message
                
            except asyncio.TimeoutError:
                last_error = f"Tool {tool_name} timed out after {self.timeout_seconds}s"
            except Exception as e:
                last_error = str(e)
            
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # 지수 백오프
        
        return ToolResult(
            success=False,
            data=None,
            error_message=f"Failed after {max_retries} retries: {last_error}"
        )
```

### Phase Transition Management
Phase 간 데이터 전달 및 상태 관리:

```python
class PhaseTransitionManager:
    """Phase 간 전환 및 데이터 전달 관리"""
    
    def __init__(self):
        self.phase_results = {}
        self.transition_rules = {
            "commit_collection": "review_execution",
            "review_execution": "deepeval_conversion", 
            "deepeval_conversion": "analysis",
            "analysis": None  # 마지막 단계
        }
    
    def store_phase_result(self, phase: str, result: Any):
        """Phase 결과 저장"""
        self.phase_results[phase] = result
    
    def get_input_for_phase(self, phase: str) -> Dict[str, Any]:
        """다음 Phase의 입력 데이터 준비"""
        if phase == "commit_collection":
            return {}  # 첫 단계는 설정에서 입력
        elif phase == "review_execution":
            return {"commits": self.phase_results["commit_collection"]}
        elif phase == "deepeval_conversion":
            return {"reviews": self.phase_results["review_execution"]}
        elif phase == "analysis":
            return {"evaluations": self.phase_results["deepeval_conversion"]}
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    def get_next_phase(self, current_phase: str) -> Optional[str]:
        """다음 실행할 Phase 반환"""
        return self.transition_rules.get(current_phase)
```

## 상태 관리 및 메모리

### Working Memory
```python
class WorkingMemory:
    """에이전트 작업 메모리"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.memory = {}
        self.access_count = {}
        self.timestamps = {}
    
    def store(self, key: str, value: Any, ttl: Optional[int] = None):
        """메모리에 저장"""
        if len(self.memory) >= self.max_size:
            self._evict_lru()
        
        self.memory[key] = value
        self.access_count[key] = 0
        self.timestamps[key] = time.time()
        
        if ttl:
            asyncio.create_task(self._schedule_cleanup(key, ttl))
    
    def retrieve(self, key: str) -> Optional[Any]:
        """메모리에서 조회"""
        if key in self.memory:
            self.access_count[key] += 1
            return self.memory[key]
        return None
    
    def _evict_lru(self):
        """LRU 정책으로 메모리 정리"""
        if not self.memory:
            return
        
        # 가장 적게 사용된 항목 제거
        lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
        self.remove(lru_key)
    
    async def _schedule_cleanup(self, key: str, ttl: int):
        """TTL 기반 자동 정리"""
        await asyncio.sleep(ttl)
        self.remove(key)
```

### Session State Management
```python
class SessionState:
    """평가 세션 상태 관리"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.current_phase = None
        self.phase_states = {}
        self.global_state = {}
        self.checkpoints = []
    
    def save_checkpoint(self, phase: str, state: Dict[str, Any]):
        """체크포인트 저장"""
        checkpoint = {
            "phase": phase,
            "timestamp": datetime.now(),
            "state": state,
            "checkpoint_id": f"{phase}_{len(self.checkpoints)}"
        }
        self.checkpoints.append(checkpoint)
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """체크포인트 복원"""
        for checkpoint in self.checkpoints:
            if checkpoint["checkpoint_id"] == checkpoint_id:
                return checkpoint["state"]
        return None
    
    def persist_to_disk(self, file_path: str):
        """디스크에 상태 저장"""
        state_data = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "current_phase": self.current_phase,
            "phase_states": self.phase_states,
            "global_state": self.global_state,
            "checkpoints": [
                {
                    **cp,
                    "timestamp": cp["timestamp"].isoformat()
                }
                for cp in self.checkpoints
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(state_data, f, indent=2)
```

## 에이전트 안전성 및 제약

### Resource Management
```python
class ResourceManager:
    """시스템 리소스 관리 및 제한"""
    
    def __init__(self, config: ResourceConfig):
        self.max_memory_mb = config.max_memory_mb
        self.max_cpu_percent = config.max_cpu_percent
        self.max_disk_gb = config.max_disk_gb
        self.max_execution_time = config.max_execution_time
        
        self.current_usage = ResourceUsage()
        self._monitoring_task = None
    
    async def start_monitoring(self):
        """리소스 모니터링 시작"""
        self._monitoring_task = asyncio.create_task(self._monitor_resources())
    
    async def _monitor_resources(self):
        """주기적 리소스 사용량 체크"""
        while True:
            try:
                usage = await self._get_current_usage()
                
                if usage.memory_mb > self.max_memory_mb:
                    await self._handle_memory_limit()
                
                if usage.cpu_percent > self.max_cpu_percent:
                    await self._handle_cpu_limit()
                
                if usage.disk_gb > self.max_disk_gb:
                    await self._handle_disk_limit()
                
                await asyncio.sleep(5)  # 5초마다 체크
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
    
    async def _handle_memory_limit(self):
        """메모리 한계 처리"""
        # 캐시 정리
        await self._clear_caches()
        
        # 가비지 컬렉션 강제 실행
        gc.collect()
        
        # 그래도 한계 초과시 예외 발생
        if await self._get_memory_usage() > self.max_memory_mb:
            raise ResourceLimitExceeded("Memory limit exceeded")
```

### Security Constraints
```python
class SecurityManager:
    """보안 제약 및 접근 제어"""
    
    def __init__(self, config: SecurityConfig):
        self.allowed_paths = config.allowed_paths
        self.forbidden_commands = config.forbidden_commands
        self.audit_log = AuditLog()
    
    def validate_file_access(self, file_path: str, operation: str) -> bool:
        """파일 접근 권한 검증"""
        abs_path = os.path.abspath(file_path)
        
        # 허용된 경로 내부인지 확인
        for allowed_path in self.allowed_paths:
            if abs_path.startswith(os.path.abspath(allowed_path)):
                self.audit_log.log_access(abs_path, operation, "ALLOWED")
                return True
        
        self.audit_log.log_access(abs_path, operation, "DENIED")
        return False
    
    def validate_command(self, command: List[str]) -> bool:
        """명령어 실행 권한 검증"""
        cmd_name = command[0] if command else ""
        
        if cmd_name in self.forbidden_commands:
            self.audit_log.log_command(command, "DENIED")
            return False
        
        # 특별 제약: selvage-deprecated는 읽기 전용
        if "selvage-deprecated" in " ".join(command):
            if any(write_op in " ".join(command) 
                   for write_op in ["commit", "push", "rm", "mv"]):
                self.audit_log.log_command(command, "DENIED - READ_ONLY")
                return False
        
        self.audit_log.log_command(command, "ALLOWED") 
        return True
```

## 사용 모델 전략
- **Primary**: `gemini-2.5-pro` (속도/비용 최적화)

## 대상 repo-path
- cline
    - path: /Users/demin_coder/Dev/cline
    - description: typescript로 구현된 coding assistant
- selvage-deprecated
    - path: /Users/demin_coder/Dev/selvage-deprecated
    - description: selvage가 정식 배포되기 전 commit history를 가지고 있는 repository (주의: 현재 selvage의 이전 작업 폴더이므로 review 대상으로서만 접근할 것)
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