# 핵심 에이전트 구현

## SelvageEvaluationAgent 클래스

```python
class SelvageEvaluationAgent:
    """
    단일 에이전트로 전체 평가 프로세스를 관리하는 Selvage 평가 에이전트
    대화형 모드와 자동 실행 모드를 모두 지원
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.tool_executor = self._initialize_tool_executor()  # ToolExecutor 사용
        self.working_memory = WorkingMemory()
        self.session_state = SessionState()
        self.current_phase = None
        self.llm = self._initialize_llm()  # Query Planning용 LLM
        self.is_interactive_mode = False
    
    def handle_user_message(self, message: str) -> str:
        """
        현대적 에이전트 패턴으로 사용자 메시지 처리
        
        Flow:
        1. LLM이 쿼리 분석 및 실행 계획 수립
        2. 계획에 따라 도구들 실행  
        3. 도구 결과를 바탕으로 LLM이 최종 응답 생성
        """
        try:
            # 1. LLM 기반 쿼리 분석 및 실행 계획 수립
            plan = self.plan_execution(message)
            
            # 2. 안전성 검증
            if not self._validate_plan_safety(plan):
                return f"요청하신 작업은 보안상 실행할 수 없습니다: {plan.safety_check}"
            
            # 3. 계획에 따라 도구들 실행 (ToolExecutor 사용)
            tool_results = []
            for tool_call in plan.tool_calls:
                result = self.tool_executor.execute_tool_call(tool_call.tool, tool_call.params)
                tool_results.append({
                    "tool": tool_call.tool,
                    "result": result,
                    "rationale": tool_call.rationale
                })
            
            # 4. 도구 결과를 바탕으로 LLM이 최종 응답 생성
            return self.generate_response(message, plan, tool_results)
            
        except Exception as e:
            return f"메시지 처리 중 오류가 발생했습니다: {str(e)}"
    
    def plan_execution(self, user_query: str) -> ExecutionPlan:
        """LLM을 통한 쿼리 분석 및 실행 계획 수립"""
        
        # 현재 상태 정보 수집
        current_state = self._analyze_current_state()
        
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
        
        response = self.llm.query(
            messages=messages,
            response_format="json",
            max_tokens=1000
        )
        
        return ExecutionPlan.from_json(response)
    
    def generate_response(self, user_query: str, plan: ExecutionPlan, tool_results: List[Dict]) -> str:
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
        
        response = self.llm.query(
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
    
    def _initialize_tool_executor(self) -> ToolExecutor:
        """ToolExecutor 초기화 및 도구 등록"""
        from .tools.tool_executor import ToolExecutor
        from .tools.command_tools import ExecuteSafeCommandTool, ListDirectoryTool
        from .tools.file_tools import ReadFileTool, WriteFileTool, FileExistsTool
        
        executor = ToolExecutor()
        
        # 모든 도구 등록
        executor.register_tool(ExecuteSafeCommandTool())
        executor.register_tool(ListDirectoryTool())
        executor.register_tool(ReadFileTool())
        executor.register_tool(WriteFileTool())
        executor.register_tool(FileExistsTool())
        
        return executor
    
    def _get_available_tools_description(self) -> str:
        """사용 가능한 도구들의 설명 반환"""
        tools_info = self.tool_executor.get_available_tools()
        descriptions = []
        for tool_name, tool_info in tools_info.items():
            descriptions.append(f"- {tool_name}: {tool_info['description']}")
        return "\\n".join(descriptions)
```

## 자동 실행 모드 구현

```python
    def execute_evaluation(self) -> EvaluationReport:
        """
        에이전트 방식으로 평가 프로세스 실행
        상태를 파악하고 동적으로 다음 행동 결정
        """
        session_id = self._generate_session_id()
        
        while True:
            # 현재 상태 분석
            current_state = self._analyze_current_state()
            
            # 다음 행동 결정
            next_action = self._decide_next_action(current_state)
            
            if next_action == "COMPLETE":
                break
                
            # 행동 실행
            action_result = self._execute_action(next_action, current_state)
            
            # 결과 저장 및 상태 업데이트
            self._update_state(action_result)
        
        # 최종 보고서 생성
        return self._generate_final_report(session_id)
    
    def _analyze_current_state(self) -> Dict[str, Any]:
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
        if self.tools["file_exists"].execute(file_path=commits_file):
            state["completed_phases"].append("commit_collection")
            commit_data = self.tools["read_file"].execute(
                file_path=commits_file, as_json=True
            )
            state["available_data"]["commits"] = commit_data.data["content"]
        
        # Phase 2: 리뷰 실행 상태 확인
        review_logs_exist = self._check_review_logs_exist()
        if review_logs_exist:
            state["completed_phases"].append("review_execution")
            state["available_data"]["reviews"] = self._scan_review_logs()
        
        # Phase 3: DeepEval 결과 확인
        eval_results_exist = self._check_evaluation_results_exist()
        if eval_results_exist:
            state["completed_phases"].append("deepeval_conversion")
            state["available_data"]["evaluations"] = self._load_evaluation_results()
        
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
    
    def _decide_next_action(self, current_state: Dict[str, Any]) -> str:
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
    
    def _execute_action(self, action: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        결정된 행동을 실행
        
        구체적인 Phase별 구현은 다음 문서들을 참조:
        - Phase 1-2: docs/specs/02-commit-collection-and-review-execution.md
        - Phase 3-4: docs/specs/03-evaluation-conversion-and-analysis.md
        """
        if action == "EXECUTE_COMMIT_COLLECTION":
            # Phase 1 구현은 02-commit-collection-and-review-execution.md 참조
            return self._execute_phase1_commit_collection()
        elif action == "EXECUTE_REVIEW_EXECUTION":
            commits = current_state["available_data"]["commits"]
            # Phase 2 구현은 02-commit-collection-and-review-execution.md 참조
            return self._execute_phase2_review_execution(commits)
        elif action == "EXECUTE_DEEPEVAL_CONVERSION":
            reviews = current_state["available_data"]["reviews"]
            # Phase 3 구현은 03-evaluation-conversion-and-analysis.md 참조
            return self._execute_phase3_deepeval_conversion(reviews)
        elif action == "EXECUTE_ANALYSIS":
            evaluations = current_state["available_data"]["evaluations"]
            # Phase 4 구현은 03-evaluation-conversion-and-analysis.md 참조
            return self._execute_phase4_analysis(evaluations)
        elif action.startswith("SKIP_TO_"):
            # 스킵 액션 처리
            return {"action": action, "skipped": True}
        else:
            raise ValueError(f"Unknown action: {action}")
```

## 도구 실행 전략

### Phase-Sequential Tool Execution
단일 에이전트가 각 Phase 내에서 도구들을 순차적으로 실행하는 전략:

```python
class SingleAgentToolExecutor:
    """단일 에이전트의 도구 실행 관리"""
    
    def __init__(self, agent: SelvageEvaluationAgent):
        self.agent = agent
        self.retry_count = 3
        self.timeout_seconds = 300
    
    def execute_phase_tools(self, phase: str, tool_sequence: List[Dict]) -> List[ToolResult]:
        """Phase 내 도구들을 순차 실행"""
        results = []
        
        for tool_config in tool_sequence:
            tool_name = tool_config["name"]
            tool_params = tool_config["params"]
            
            # 재시도 로직 포함 도구 실행
            result = self._execute_with_retry(
                tool_name=tool_name,
                params=tool_params,
                max_retries=self.retry_count
            )
            
            results.append(result)
            
            # 중요한 도구 실패 시 Phase 중단
            if not result.success and tool_config.get("critical", False):
                raise PhaseExecutionError(f"Critical tool {tool_name} failed in {phase}")
        
        return results
    
    def _execute_with_retry(self, tool_name: str, params: Dict, max_retries: int) -> ToolResult:
        """재시도 로직이 포함된 도구 실행"""
        import time
        import signal
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                tool = self.agent.tools[tool_name]
                
                # 타임아웃 처리 (signal을 사용한 동기적 타임아웃)
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Tool {tool_name} timed out after {self.timeout_seconds}s")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout_seconds)
                
                try:
                    result = tool.execute(**params)
                    signal.alarm(0)  # 타임아웃 해제
                    
                    if result.success:
                        return result
                        
                    last_error = result.error_message
                finally:
                    signal.alarm(0)  # 확실히 타임아웃 해제
                
            except TimeoutError as e:
                last_error = str(e)
            except Exception as e:
                last_error = str(e)
            
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # 지수 백오프
        
        return ToolResult(
            success=False,
            data=None,
            error_message=f"Failed after {max_retries} retries: {last_error}"
        )
```

## Phase Transition Management
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