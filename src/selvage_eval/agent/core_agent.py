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
        self.session_state: Optional[SessionState] = None
        self.current_phase: Optional[str] = None
        self.is_interactive_mode = False
        
        logger.info(f"Initialized SelvageEvaluationAgent with model: {config.agent_model}")
    
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start new evaluation session
        
        Args:
            session_id: Session ID (auto-generated if None)
            
        Returns:
            Created session ID
        """
        self.session_state = SessionState(session_id)
        
        # Save session metadata
        self._save_session_metadata()
        
        # Start auto-persistence
        self.session_state.auto_persist(self.config.evaluation.output_dir)
        
        logger.info(f"Started evaluation session: {self.session_state.session_id}")
        return self.session_state.session_id
    
    def handle_user_message(self, message: str) -> str:
        """
        Handle user message using modern agent pattern
        
        Flow:
        1. LLM analyzes query and creates execution plan
        2. Execute tools according to plan  
        3. LLM generates final response based on tool results
        
        Args:
            message: User message
            
        Returns:
            Response result
        """
        if not self.session_state:
            self.start_session()
        
        self.is_interactive_mode = True
        
        try:
            # 1. LLM-based query analysis and execution planning
            plan = self.plan_execution(message)
            
            # 2. Safety validation
            if not self._validate_plan_safety(plan):
                return f"Cannot execute requested task for security reasons: {plan.safety_check}"
            
            # 3. Execute tools according to plan
            tool_results = []
            for tool_call in plan.tool_calls:
                result = self.execute_tool(tool_call.tool, tool_call.params)
                tool_results.append({
                    "tool": tool_call.tool,
                    "result": result,
                    "rationale": tool_call.rationale
                })
            
            # 4. Generate final response based on tool results
            return self.generate_response(message, plan, tool_results)
            
        except Exception as e:
            logger.error(f"Error handling user message: {e}")
            return f"Error occurred while processing message: {str(e)}"
    
    def plan_execution(self, user_query: str) -> ExecutionPlan:
        """Query analysis and execution planning via LLM
        
        Args:
            user_query: User query
            
        Returns:
            Execution plan
        """
        # Collect current state information
        current_state = self._analyze_current_state()
        
        # TODO: Implement actual LLM call
        # For now, use simple rule-based planning
        plan = self._create_simple_plan(user_query, current_state)
        
        logger.debug(f"Created execution plan for query: {user_query[:50]}...")
        return plan
    
    def _create_simple_plan(self, user_query: str, current_state: Dict[str, Any]) -> ExecutionPlan:
        """Create simple rule-based execution plan (temporary implementation)
        
        Args:
            user_query: User query
            current_state: Current state
            
        Returns:
            Execution plan
        """
        
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
                expected_outcome="Current session status information"
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
                expected_outcome="Selected commit list"
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
                expected_outcome="Output directory file list"
            )
    
    def generate_response(self, user_query: str, plan: ExecutionPlan, tool_results: List[Dict]) -> str:
        """Generate final response to user based on tool execution results
        
        Args:
            user_query: User query
            plan: Execution plan
            tool_results: Tool execution results
            
        Returns:
            User response
        """
        # TODO: Implement actual LLM-based response generation
        # For now, use simple text response
        
        if not tool_results:
            return "No tools were executed."
        
        response_parts = [f"**{plan.intent_summary}**\n"]
        
        for result in tool_results:
            tool_name = result["tool"]
            tool_result = result["result"]
            
            if tool_result.success:
                if tool_name == "read_file":
                    content = tool_result.data.get("content", {})
                    if isinstance(content, dict):
                        response_parts.append(f"[FILE] File content ({len(content)} items):")
                        for key, value in list(content.items())[:3]:  # Show first 3 only
                            response_parts.append(f"  - {key}: {str(value)[:100]}...")
                    else:
                        response_parts.append(f"[FILE] File content: {str(content)[:200]}...")
                        
                elif tool_name == "list_directory":
                    files = tool_result.data.get("files", [])
                    dirs = tool_result.data.get("directories", [])
                    response_parts.append("[DIRECTORY] Directory contents:")
                    response_parts.append(f"  - Files: {len(files)} items")
                    response_parts.append(f"  - Directories: {len(dirs)} items")
                    if files:
                        response_parts.append(f"  - Main files: {', '.join(files[:5])}")
                        
                else:
                    response_parts.append(f"[SUCCESS] {tool_name} execution completed")
            else:
                response_parts.append(f"[ERROR] {tool_name} execution failed: {tool_result.error_message}")
        
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
    
    def execute_evaluation(self) -> Dict[str, Any]:
        """
        Execute evaluation process using agent approach
        Analyze state and dynamically decide next actions
        
        Returns:
            Evaluation result dictionary
        """
        if not self.session_state:
            self.start_session()
        
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
        if not self.session_state:
            return {"error": "No active session"}
        
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
        if not self.session_state:
            return {"error": "No session state"}
        
        return {
            "session_summary": self.session_state.get_session_summary(),
            "completed_phases": self.session_state.get_completed_phases(),
            "status": "completed"
        }
    
    def _save_session_metadata(self) -> None:
        """Save session metadata"""
        if not self.session_state:
            return
        
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
