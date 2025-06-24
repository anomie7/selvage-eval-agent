"""Selvage Evaluation Agent Core Class

Single agent that manages the entire evaluation process for Selvage.
Supports both interactive mode and automatic execution mode.
"""

from typing import Dict, List, Any, Optional
import json
import logging
import os
import re

from selvage_eval.tools.execution_plan import ExecutionPlan
from selvage_eval.tools.tool import Tool
from selvage_eval.tools.tool_call import ToolCall
from selvage_eval.tools.tool_executor import ToolExecutor
from selvage_eval.tools.tool_result import ToolResult
from selvage_eval.tools.read_file_tool import ReadFileTool
from selvage_eval.tools.write_file_tool import WriteFileTool
from selvage_eval.tools.file_exists_tool import FileExistsTool
from selvage_eval.tools.execute_safe_command_tool import ExecuteSafeCommandTool
from selvage_eval.tools.list_directory_tool import ListDirectoryTool

from ..config.settings import EvaluationConfig
from ..memory.session_state import SessionState
from ..llm.gemini_client import GeminiClient

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
        
        # LLM 클라이언트 초기화
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError("GEMINI_API_KEY is not set")
        try:
            self.gemini_client = GeminiClient(
                api_key=api_key,
                model_name=config.agent_model
            )
            logger.info(f"LLM integration enabled with model: {config.agent_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client, falling back to rule-based approach: {e}")
            raise e

        
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
    
    def _get_available_tools(self) -> List[Tool]:
        """사용 가능한 모든 도구들을 반환합니다
        
        Returns:
            List[Tool]: 사용 가능한 도구들의 리스트
        """
        return [
            ReadFileTool(),
            WriteFileTool(),
            FileExistsTool(),
            ExecuteSafeCommandTool(),
            ListDirectoryTool()
        ]
    
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
        
        # 프롬프트 구성
        messages = [
            {"role": "user", "content": f"""
현재 상태: {json.dumps(current_state, ensure_ascii=False, indent=2)}

사용자 쿼리: {user_query}
            """}
        ]
        
        # 사용 가능한 도구들 가져오기
        available_tools: List[Tool] = self._get_available_tools()
        
        # LLM 호출 (Function Calling 방식)
        try:
            response = self.gemini_client.query(
                messages=messages,
                system_instruction=self._build_execution_plan_prompt(),
                tools=available_tools
            )
            
            # Function call 응답 처리 및 ExecutionPlan 생성
            plan = self._parse_execution_plan(response)
            
            logger.debug(f"Created LLM-based execution plan for query: {user_query[:50]}... (with {len(conversation_context)} context turns)")
            return plan
            
        except Exception as e:
            logger.error(f"Error in plan_execution: {e}")
            raise e
    
    
    def generate_response(self, user_query: str, plan: ExecutionPlan, tool_results: List[Dict]) -> str:
        """도구 실행 결과를 바탕으로 사용자에게 제공할 최종 응답 생성
        
        Args:
            user_query: 사용자 질문
            plan: 실행 계획
            tool_results: 도구 실행 결과
            
        Returns:
            사용자 응답
        """
        # 대화 컨텍스트 수집
        conversation_context = self.session_state.get_conversation_context()
        
        # 시스템 프롬프트 구성
        system_prompt = self._build_response_system_prompt(conversation_context)
        
        # 응답 생성 프롬프트 구성
        response_prompt = self._build_response_generation_prompt(
            user_query, plan, tool_results
        )
        
        messages = [{"role": "user", "content": response_prompt}]
        
        # LLM 호출
        try:
            response = self.gemini_client.query(
                messages=messages,
                system_instruction=system_prompt,
            )
            
            logger.debug(f"Generated LLM-based response for query: {user_query[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            raise e
    
    
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
    
    def _build_execution_plan_prompt(
        self,
    ) -> str:
        """실행 계획 생성용 시스템 인스트럭션을 구성합니다"""
        
        return """당신은 Selvage 평가 에이전트입니다. 
사용자의 요청을 분석하여 적절한 도구들을 호출해주세요.

사용자의 의도를 파악하고 필요한 작업을 수행하기 위해 제공된 도구들을 사용하세요.
각 도구 호출 시 명확한 이유와 함께 적절한 파라미터를 제공해주세요.

안전성을 고려하여 파일 시스템 작업이나 명령어 실행 시 주의깊게 검토해주세요."""
    
    def _parse_execution_plan(self, response: Any) -> ExecutionPlan:
        """Function call 응답을 ExecutionPlan 객체로 변환합니다"""
        
        tool_calls = []
        
        # Gemini function calling 응답 처리
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call'):
                        function_call = part.function_call
                        tool_call = ToolCall(
                            tool=function_call.name,
                            params=dict(function_call.args) if hasattr(function_call, 'args') else {},
                            rationale=f"LLM이 {function_call.name} 도구를 선택함"
                        )
                        tool_calls.append(tool_call)
        
        # tool_calls가 없는 경우 기본 계획 생성
        if not tool_calls:
            # 응답에서 텍스트 내용 추출하여 의도 파악
            intent_summary = "사용자 요청을 처리하기 위한 계획 수립"
            if hasattr(response, 'text') and response.text:
                intent_summary = f"텍스트 응답: {response.text[:100]}..."
        else:
            intent_summary = f"{len(tool_calls)}개의 도구 호출을 통한 작업 수행"
        
        # ExecutionPlan 객체 생성
        execution_plan = ExecutionPlan(
            intent_summary=intent_summary,
            confidence=0.9,  # Function calling의 경우 높은 신뢰도
            parameters={},
            tool_calls=tool_calls,
            safety_check="Function calling 방식으로 안전성 검증됨",
            expected_outcome="도구 호출을 통한 요청 처리 완료 예상"
        )
        
        return execution_plan
    
    def _build_response_system_prompt(
        self,
        conversation_context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """응답 생성용 시스템 프롬프트를 구성합니다
        
        Args:
            conversation_context: 대화 컨텍스트 (선택사항)
            
        Returns:
            시스템 프롬프트 문자열
        """
        system_prompt_parts = [
            "# ROLE",
            "사용자에게 도구 실행 결과를 바탕으로 명확하고 유용한 답변을 제공하는 어시스턴트입니다.",
            "",
            "# GUIDELINES",
            "- 핵심 정보를 명확히 전달",
            "- 필요시 표나 리스트 형태로 구조화",
            "- 다음 단계 제안 (해당되는 경우)",
            "- 한국어로 자연스럽게 응답"
        ]
        
        # 대화 컨텍스트가 있는 경우 포함
        if conversation_context and len(conversation_context) > 0:
            system_prompt_parts.extend([
                "",
                "# CONVERSATION CONTEXT",
                "이전 대화 내용을 참고하여 문맥에 맞는 응답을 생성하세요:",
                ""
            ])
            
            for i, turn in enumerate(conversation_context, 1):
                system_prompt_parts.append(f"**대화 {i}:**")
                system_prompt_parts.append(f"사용자: {turn.get('user_message', '').strip()}")
                system_prompt_parts.append(f"어시스턴트: {turn.get('assistant_response', '').strip()}")
                system_prompt_parts.append("")
        
        return "\n".join(system_prompt_parts)

    def _build_response_generation_prompt(
        self,
        user_query: str,
        execution_plan: ExecutionPlan,
        tool_results: List[Dict[str, Any]],
    ) -> str:
        """응답 생성용 프롬프트를 구성합니다"""
        
        # tool_results를 JSON 직렬화 가능한 형태로 변환
        serializable_results = []
        for result in tool_results:
            serializable_result = {
                "tool": result["tool"],
                "rationale": result["rationale"]
            }
            
            # ToolResult 객체를 딕셔너리로 변환
            tool_result = result["result"]
            if hasattr(tool_result, '__dict__'):
                serializable_result["result"] = {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "error_message": tool_result.error_message,
                    "execution_time": tool_result.execution_time,
                    "metadata": tool_result.metadata
                }
            else:
                serializable_result["result"] = tool_result
            
            serializable_results.append(serializable_result)
        
        response_prompt = f"""
# CONTEXT
사용자 질문: {user_query}
의도 분석: {execution_plan.intent_summary}
예상 결과: {execution_plan.expected_outcome}

# TOOL EXECUTION RESULTS
{json.dumps(serializable_results, ensure_ascii=False, indent=2)}

# TASK
도구 실행 결과를 바탕으로 사용자가 이해하기 쉬운 답변을 생성하세요.
        """
        
        return response_prompt
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """LLM 응답에서 JSON을 추출하고 파싱합니다
        
        Args:
            response: LLM 응답 텍스트
            
        Returns:
            파싱된 JSON 데이터
            
        Raises:
            json.JSONDecodeError: JSON 파싱 실패
        """
        # markdown 코드 블록에서 JSON 추출 (```json ... ```)
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if json_match:
            json_content = json_match.group(1).strip()
            return json.loads(json_content)
        
        # 코드 블록이 없다면 전체 응답을 JSON으로 시도
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # 백틱이나 다른 마크다운 요소 제거 후 재시도
            cleaned_response = re.sub(r'```[a-z]*\n?|```\n?|`', '', response).strip()
            return json.loads(cleaned_response)
