"""Selvage Evaluation Agent Core Class

Single agent that manages the entire evaluation process for Selvage.
Supports both interactive mode and automatic execution mode.
"""

from typing import Dict, List, Any, Optional
import logging

from selvage_eval.tools.execution_plan import ExecutionPlan
from selvage_eval.tools.tool_call import ToolCall
from selvage_eval.tools.tool_executor import ToolExecutor
from selvage_eval.tools.tool_result import ToolResult

from ..config.settings import EvaluationConfig
from ..memory.session_state import SessionState

logger = logging.getLogger(__name__)


class SelvageEvaluationAgent:
    """
    Single agent that manages the entire evaluation process for Selvage.
    Supports both interactive mode and automatic execution mode.
    """
    
    def __init__(self, config: EvaluationConfig):
        """Initialize the agent
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.session_state = SessionState()  # 즉시 초기화
        self.current_phase: Optional[str] = None
        self.is_interactive_mode = False
        
        # 초기 세션 메타데이터 저장
        self._save_session_metadata()
        
        # 자동 영속화 시작
        self.session_state.auto_persist(self.config.evaluation.output_dir)
        
        logger.info(f"Initialized SelvageEvaluationAgent with model: {config.agent_model} (session: {self.session_state.session_id})")
    
    
    def reset_session(self, session_id: Optional[str] = None) -> str:
        """Reset to new evaluation session
        
        Args:
            session_id: Session ID (auto-generated if None)
            
        Returns:
            Created session ID
        """
        old_session_id = self.session_state.session_id
        self.session_state = SessionState(session_id)
        
        # 새 세션 메타데이터 저장 및 자동 영속화 재시작
        self._save_session_metadata()
        self.session_state.auto_persist(self.config.evaluation.output_dir)
        
        logger.info(f"Reset from session {old_session_id} to new session: {self.session_state.session_id}")
        return self.session_state.session_id
    
    def handle_user_message(self, message: str) -> str:
        """
        개선된 대화형 메시지 처리
        
        Flow:
        1. 특수 명령어 처리 (/clear, /context)
        2. 대화 히스토리를 포함한 실행 계획 수립
        3. 계획에 따라 도구들 실행  
        4. 도구 결과를 바탕으로 최종 응답 생성
        5. 대화 히스토리에 추가
        
        Args:
            message: User message
            
        Returns:
            Response result
        """
        self.is_interactive_mode = True
        
        # 특수 명령어 처리
        if message.startswith('/'):
            return self._handle_special_command(message)
        
        try:
            # 1. 대화 히스토리를 포함한 실행 계획 수립
            plan = self.plan_execution(message)
            
            # 2. 안전성 검증
            if not self._validate_plan_safety(plan):
                response = f"보안상 실행할 수 없습니다: {plan.safety_check}"
                # 오류 마저 히스토리에 기록
                self.session_state.add_conversation_turn(
                    user_message=message,
                    assistant_response=response
                )
                return response
            
            # 3. 계획에 따라 도구들 실행
            tool_results = []
            for tool_call in plan.tool_calls:
                result = self.execute_tool(tool_call.tool, tool_call.params)
                tool_results.append({
                    "tool": tool_call.tool,
                    "result": result,
                    "rationale": tool_call.rationale
                })
            
            # 4. 도구 결과를 바탕으로 최종 응답 생성
            response = self.generate_response(message, plan, tool_results)
            
            # 5. 대화 히스토리에 추가
            self.session_state.add_conversation_turn(
                user_message=message,
                assistant_response=response,
                tool_results=tool_results
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling user message: {e}")
            error_response = f"메시지 처리 중 오류가 발생했습니다: {str(e)}"
            
            # 오류 상황도 히스토리에 기록
            self.session_state.add_conversation_turn(
                user_message=message,
                assistant_response=error_response
            )
            
            return error_response
    
    def plan_execution(self, user_query: str) -> ExecutionPlan:
        """Query analysis and execution planning via LLM
        
        Args:
            user_query: User query
            
        Returns:
            Execution plan
        """
        # 현재 상태 정보 수집
        current_state = self._analyze_current_state()
        
        # 대화 히스토리 컨텍스트 수집
        conversation_context = self.session_state.get_conversation_context()
        
        # TODO: Implement actual LLM call with conversation history
        # For now, use simple rule-based planning with history awareness
        plan = self._create_simple_plan(user_query, current_state, conversation_context)
        
        logger.debug(f"Created execution plan for query: {user_query[:50]}... (with {len(conversation_context)} context turns)")
        return plan
    
    def _create_simple_plan(self, user_query: str, current_state: Dict[str, Any], 
                          conversation_context: Optional[List[Dict[str, Any]]] = None) -> ExecutionPlan:
        """버번 법칙 기반 실행 계획 생성 (임시 구현)
        
        Args:
            user_query: 사용자 질문
            current_state: 현재 상태
            conversation_context: 대화 컨텍스트
            
        Returns:
            실행 계획
        """
        
        # 대화 컨텍스트 고려 (TODO: 향후 LLM에서 활용)
        context_info = ""
        if conversation_context:
            context_info = f" (이전 {len(conversation_context)}개 대화 참고)"
        
        query_lower = user_query.lower()
        
        # Status query
        if any(keyword in query_lower for keyword in ["status", "current", "progress"]):
            return ExecutionPlan(
                intent_summary="Query current status",
                confidence=0.9,
                parameters={},
                tool_calls=[
                    ToolCall(
                        tool="read_file",
                        params={"file_path": f"{self.config.evaluation.output_dir}/session_metadata.json", "as_json": True},
                        rationale="Read session metadata"
                    )
                ],
                safety_check="Read-only operation, safe",
                expected_outcome=f"Current session status information{context_info}"
            )
        
        # Commit list query
        elif any(keyword in query_lower for keyword in ["commit"]):
            return ExecutionPlan(
                intent_summary="Query commit list",
                confidence=0.8,
                parameters={},
                tool_calls=[
                    ToolCall(
                        tool="read_file",
                        params={"file_path": f"{self.config.evaluation.output_dir}/meaningful_commits.json", "as_json": True},
                        rationale="Read selected commit list"
                    )
                ],
                safety_check="Read-only operation, safe",
                expected_outcome=f"Selected commit list{context_info}"
            )
        
        # Default directory query
        else:
            return ExecutionPlan(
                intent_summary="Query output directory",
                confidence=0.5,
                parameters={},
                tool_calls=[
                    ToolCall(
                        tool="list_directory",
                        params={"directory_path": self.config.evaluation.output_dir},
                        rationale="Check output directory contents"
                    )
                ],
                safety_check="Read-only operation, safe",
                expected_outcome=f"Output directory file list{context_info}"
            )
    
    def generate_response(self, user_query: str, plan: ExecutionPlan, tool_results: List[Dict]) -> str:
        """도구 실행 결과를 바탕으로 사용자에게 제공할 최종 응답 생성
        
        Args:
            user_query: 사용자 질문
            plan: 실행 계획
            tool_results: 도구 실행 결과
            
        Returns:
            사용자 응답
        """
        # TODO: Implement actual LLM-based response generation with conversation history
        # For now, use enhanced text response with context awareness
        
        # 대화 컨텍스트 고려
        conversation_context = self.session_state.get_conversation_context()
        context_info = ""
        if len(conversation_context) > 0:
            context_info = f" (이전 대화 {len(conversation_context)}개 참고)"
        
        if not tool_results:
            return f"도구가 실행되지 않았습니다.{context_info}"
        
        response_parts = [f"**{plan.intent_summary}**{context_info}\n"]
        
        for result in tool_results:
            tool_name = result["tool"]
            tool_result = result["result"]
            
            if tool_result.success:
                if tool_name == "read_file":
                    content = tool_result.data.get("content", {})
                    if isinstance(content, dict):
                        response_parts.append(f"[FILE] 파일 내용 ({len(content)}개 항목):")
                        for key, value in list(content.items())[:3]:  # Show first 3 only
                            response_parts.append(f"  - {key}: {str(value)[:100]}...")
                    else:
                        response_parts.append(f"[FILE] 파일 내용: {str(content)[:200]}...")
                        
                elif tool_name == "list_directory":
                    files = tool_result.data.get("files", [])
                    dirs = tool_result.data.get("directories", [])
                    response_parts.append("[DIRECTORY] 디렉토리 내용:")
                    response_parts.append(f"  - 파일: {len(files)}개")
                    response_parts.append(f"  - 디렉토리: {len(dirs)}개")
                    if files:
                        response_parts.append(f"  - 주요 파일: {', '.join(files[:5])}")
                        
                else:
                    response_parts.append(f"[SUCCESS] {tool_name} 실행 완료")
            else:
                response_parts.append(f"[ERROR] {tool_name} 실행 실패: {tool_result.error_message}")
        
        return "\n".join(response_parts)
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """Execute tool
        
        Args:
            tool_name: Tool name
            params: Tool parameters
            
        Returns:
            Tool execution result
        """
        try:
            result = ToolExecutor().execute_tool_call(tool_name, params)
            
            logger.debug(f"Executed tool {tool_name} in {result.execution_time:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=str(e)
            )
    
    def _validate_plan_safety(self, plan: ExecutionPlan) -> bool:
        """Validate execution plan safety
        
        Args:
            plan: Execution plan to validate
            
        Returns:
            Safety validation result
        """
        # Check forbidden tools
        forbidden_tools = ["delete_file", "modify_repository", "system_command"]
        for tool_call in plan.tool_calls:
            if tool_call.tool in forbidden_tools:
                return False
        
        # Check selvage-deprecated write operations
        for tool_call in plan.tool_calls:
            if "selvage-deprecated" in str(tool_call.params) and tool_call.tool.startswith("write"):
                return False
        
        return True
    
    def _handle_special_command(self, command: str) -> str:
        """특수 명령어 처리
        
        Args:
            command: 입력된 명령어
            
        Returns:
            명령어 처리 결과
        """
        command = command.strip().lower()
        
        if command == '/clear':
            return self._clear_conversation()
        elif command == '/context':
            return self._show_context_info()
        else:
            return f"알 수 없는 명령어입니다: {command}\n"\
                   f"사용 가능한 명령어:\n"\
                   f"  /clear  - 대화 히스토리 초기화\n"\
                   f"  /context - 컨텍스트 정보 표시"
    
    def _clear_conversation(self) -> str:
        """대화 히스토리 초기화
        
        Returns:
            초기화 결과 메시지
        """
        # 현재 대화 수 확인
        context_stats = self.session_state.get_context_stats()
        total_turns = context_stats["total_conversation_turns"]
        
        # 대화 히스토리 초기화
        self.session_state.clear_conversation_history()
        
        return f"대화 히스토리가 초기화되었습니다. ({total_turns}개 대화 삭제)"
    
    def _show_context_info(self) -> str:
        """컨텍스트 정보 표시
        
        Returns:
            컨텍스트 정보 메시지
        """
        stats = self.session_state.get_context_stats()
        
        utilization_percent = stats["context_utilization"] * 100
        
        info_parts = [
            "**컨텍스트 사용량 정보**",
            f"전체 대화 수: {stats['total_conversation_turns']}개",
            f"현재 컨텍스트 대화 수: {stats['context_turns']}개",
            f"현재 컨텍스트 토큰 수: {stats['current_context_tokens']:,}",
            f"최대 컨텍스트 토큰 수: {stats['max_context_tokens']:,}",
            f"컨텍스트 사용률: {utilization_percent:.1f}%",
            f"최대 히스토리 보존 수: {stats['max_history_entries']}개"
        ]
        
        if utilization_percent > 80:
            info_parts.append("\n⚠️  컨텍스트 사용률이 높습니다. '/clear' 명령어로 초기화를 고려해보세요.")
        
        return "\n".join(info_parts)
    
    def execute_evaluation(self) -> Dict[str, Any]:
        """
        Execute evaluation process using agent approach
        Analyze state and dynamically decide next actions
        
        Returns:
            Evaluation result dictionary
        """
        # SessionState는 항상 존재함 (생성자에서 초기화됨)
        
        logger.info("Starting automatic evaluation execution")
        
        while True:
            # Analyze current state
            current_state = self._analyze_current_state()
            
            # Decide next action
            next_action = self._decide_next_action(current_state)
            
            if next_action == "COMPLETE":
                break
                
            # Execute action
            action_result = self._execute_action(next_action, current_state)
            
            # Save result and update state
            self._update_state(action_result)
        
        # Generate final report
        return self._generate_final_report()
    
    def _analyze_current_state(self) -> Dict[str, Any]:
        """
        Analyze current state to determine which phases have been completed
        
        Returns:
            Current state dictionary
        """
        # SessionState는 항상 존재함
        
        state = {
            "session_id": self.session_state.session_id,
            "completed_phases": [],
            "available_data": {},
            "next_required_phase": None
        }
        
        # Phase 1: Check commit collection status
        commits_file = self.config.get_output_path("meaningful_commits.json")
        commits_exist = self.execute_tool("file_exists", {"file_path": commits_file})
        if commits_exist.success and commits_exist.data.get("exists"):
            state["completed_phases"].append("commit_collection")
            # TODO: Load actual commit data
        
        # Phase 2: Check review execution status
        review_logs_dir = self.config.get_output_path("review_logs")
        review_logs_exist = self.execute_tool("file_exists", {"file_path": review_logs_dir})
        if review_logs_exist.success and review_logs_exist.data.get("exists"):
            state["completed_phases"].append("review_execution")
        
        # Phase 3: Check DeepEval results
        eval_results_file = self.config.get_output_path("evaluations", "evaluation_results.json")
        eval_results_exist = self.execute_tool("file_exists", {"file_path": eval_results_file})
        if eval_results_exist.success and eval_results_exist.data.get("exists"):
            state["completed_phases"].append("deepeval_conversion")
        
        # Determine next required phase
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
        Decide next action based on current state
        
        Args:
            current_state: Current state
            
        Returns:
            Next action string
        """
        next_phase = current_state["next_required_phase"]
        
        if next_phase == "complete":
            return "COMPLETE"
        
        # Check skip logic
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
        Execute decided action
        
        Args:
            action: Action to execute
            current_state: Current state
            
        Returns:
            Action execution result
        """
        logger.info(f"Executing action: {action}")
        
        # TODO: Implement concrete phase implementations
        if action == "EXECUTE_COMMIT_COLLECTION":
            return self._execute_phase1_commit_collection()
        elif action == "EXECUTE_REVIEW_EXECUTION":
            return self._execute_phase2_review_execution(current_state)
        elif action == "EXECUTE_DEEPEVAL_CONVERSION":
            return self._execute_phase3_deepeval_conversion(current_state)
        elif action == "EXECUTE_ANALYSIS":
            return self._execute_phase4_analysis(current_state)
        elif action.startswith("SKIP_TO_"):
            return {"action": action, "skipped": True}
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _execute_phase1_commit_collection(self) -> Dict[str, Any]:
        """Phase 1: Commit collection execution (placeholder implementation)"""
        logger.info("Executing Phase 1: Commit Collection")
        # TODO: Actual implementation
        return {"phase": "commit_collection", "status": "placeholder"}
    
    def _execute_phase2_review_execution(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Review execution (placeholder implementation)"""
        logger.info("Executing Phase 2: Review Execution")
        # TODO: Actual implementation
        return {"phase": "review_execution", "status": "placeholder"}
    
    def _execute_phase3_deepeval_conversion(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: DeepEval conversion (placeholder implementation)"""
        logger.info("Executing Phase 3: DeepEval Conversion")
        # TODO: Actual implementation
        return {"phase": "deepeval_conversion", "status": "placeholder"}
    
    def _execute_phase4_analysis(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Analysis (placeholder implementation)"""
        logger.info("Executing Phase 4: Analysis")
        # TODO: Actual implementation
        return {"phase": "analysis", "status": "placeholder"}
    
    def _update_state(self, action_result: Dict[str, Any]) -> None:
        """Update state with action execution result
        
        Args:
            action_result: Action execution result
        """
        if self.session_state and "phase" in action_result:
            phase = action_result["phase"]
            self.session_state.update_phase_state(phase, action_result)
            
            if action_result.get("status") == "completed":
                self.session_state.mark_phase_completed(phase)
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final evaluation report
        
        Returns:
            Final report dictionary
        """
        # SessionState는 항상 존재함
        
        return {
            "session_summary": self.session_state.get_session_summary(),
            "completed_phases": self.session_state.get_completed_phases(),
            "status": "completed"
        }
    
    def _save_session_metadata(self) -> None:
        """Save session metadata"""
        # SessionState는 항상 존재함
        
        metadata = {
            "session_id": self.session_state.session_id,
            "start_time": self.session_state.start_time.isoformat(),
            "agent_model": self.config.agent_model,
            "review_models": self.config.review_models,
            "target_repositories": [repo.model_dump() for repo in self.config.target_repositories],
            "configuration": {
                "commits_per_repo": self.config.commits_per_repo,
                "workflow": self.config.workflow.model_dump(),
                "deepeval_metrics": [metric.model_dump() for metric in self.config.deepeval.metrics]
            }
        }
        
        metadata_file = self.config.get_output_path("session_metadata.json")
        self.execute_tool("write_file", {
            "file_path": metadata_file,
            "content": metadata,
            "as_json": True
        })
        
        logger.info(f"Saved session metadata: {metadata_file}")
