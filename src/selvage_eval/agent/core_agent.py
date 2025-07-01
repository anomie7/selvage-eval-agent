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
from .react_types import ToolCallModel, WorkingContext, IterationEntry, ReActDecision, ToolExecutionResult

logger = logging.getLogger(__name__)


class SelvageEvaluationAgent:
    """
    Single agent that manages the entire evaluation process for Selvage.
    Supports both interactive mode and automatic execution mode.
    """
    
    def __init__(self, config: EvaluationConfig, work_dir: str = "."):
        """Initialize the agent
        
        Args:
            config: Evaluation configuration
            work_dir: Working directory for agent operations (default: current directory)
        """
        self.config = config
        self.work_dir = work_dir
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
        
        logger.info(f"Initialized SelvageEvaluationAgent with model: {config.agent_model} (session: {self.session_state.session_id}, work_dir: {self.work_dir})")
    
    
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
    
    def _analyze_security_intent(self, message: str) -> Dict[str, Any]:
        """사용자 메시지의 보안 의도를 분석합니다
        
        Args:
            message: 사용자 메시지
            
        Returns:
            보안 분석 결과 {'is_safe': bool, 'risk_level': str, 'reason': str}
        """
        # 확장된 보안 키워드 리스트
        high_risk_keywords = [
            # 직접적 위험 명령어
            "rm -rf", "sudo", "delete", "삭제", "제거",
            # 시스템 파일 관련
            "/etc/passwd", "/etc/shadow", "시스템 파일",
            # 인증 정보 관련
            "패스워드", "비밀번호", "password", "credential", "secret", "token", "key", "api키",
            # 권한 관련
            "chmod", "chown", "권한 변경",
            # 네트워크 관련
            "curl", "wget", "download", "다운로드"
        ]
        
        medium_risk_keywords = [
            "파일 읽기", "내용 확인", "접근", "실행"
        ]
        
        message_lower = message.lower()
        
        # 고위험 키워드 검사
        for keyword in high_risk_keywords:
            if keyword.lower() in message_lower:
                return {
                    "is_safe": False,
                    "risk_level": "high",
                    "reason": f"보안 위험 키워드 감지: '{keyword}'"
                }
        
        # 의심스러운 패턴 검사 (더 구체적으로)
        suspicious_patterns = [
            r'(?:패스워드|비밀번호|password)\s*(?:파일|file)',
            r'/etc/\w+',
            r'sudo\s+\w+',
            r'rm\s+-[rf]+',
            r'(?:패스워드|비밀번호|credential|secret|api키|token)\s*(?:읽어|보여|알려)',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    "is_safe": False,
                    "risk_level": "high",
                    "reason": f"의심스러운 패턴 감지: {pattern}"
                }
        
        # 중위험 키워드 검사
        for keyword in medium_risk_keywords:
            if keyword.lower() in message_lower:
                return {
                    "is_safe": True,
                    "risk_level": "medium",
                    "reason": f"주의 필요: '{keyword}'"
                }
        
        return {
            "is_safe": True,
            "risk_level": "low",
            "reason": "안전한 요청으로 판단됨"
        }

    def handle_user_message(self, message: str) -> str:
        """
        ReAct 패턴 기반 대화형 메시지 처리
        
        Flow:
        1. 특수 명령어 처리 (/clear, /context)
        2. 보안 의도 분석
        3. ReAct 루프를 통한 전체 작업 처리 (plan_execution_loop)
        
        Args:
            message: User message
            
        Returns:
            Response result
        """
        self.is_interactive_mode = True
        
        # 1. 특수 명령어 처리 (단순히 /로 시작하는 게 아니라 실제 명령어인지 확인)
        if message.strip().startswith('/') and message.strip().split()[0] in ['/clear', '/context']:
            return self._handle_special_command(message)
        
        try:
            # 2. 보안 의도 분석
            security_analysis = self._analyze_security_intent(message)
            if not security_analysis["is_safe"]:
                response = f"보안상 요청을 처리할 수 없습니다. {security_analysis['reason']}"
                logger.warning(f"Security risk detected: {security_analysis['reason']} for message: {message}")
                
                # 보안 위험 응답도 히스토리에 기록
                self.session_state.add_conversation_turn(
                    user_message=message,
                    assistant_response=response
                )
                return response
            
            # 3. ReAct 루프를 통한 전체 작업 처리
            # plan_execution_loop가 내부적으로 히스토리 관리까지 처리함
            response = self.plan_execution_loop(message)
            
            return response
            
        except Exception as e:
            import traceback
            logger.error(f"Error handling user message: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
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
    
    def plan_execution_loop(self, user_query: str, max_iterations: int = 25) -> str:
        """ReAct 패턴 기반 계획 실행 및 응답 생성
        
        Think-Act-Observe 사이클을 반복하여 복잡한 작업을 단계별로 완료합니다.
        
        Args:
            user_query: 사용자 질문
            max_iterations: 최대 반복 횟수 (기본값: 25)
            
        Returns:
            최종 응답 문자열
        """
        working_context = WorkingContext(
            original_query=user_query,
            iteration_history=[],
            accumulated_tool_results=[]
        )
        
        for iteration in range(max_iterations):
            # Think: 현재 상황 분석 및 다음 행동 결정
            agent_decision = self._think_and_decide(working_context)
            
            if agent_decision.status == "TASK_COMPLETE":
                final_response = agent_decision.final_response or "작업이 완료되었습니다."
                # 대화 히스토리에 추가
                self.session_state.add_conversation_turn(
                    user_message=user_query,
                    assistant_response=final_response,
                    tool_results=working_context.accumulated_tool_results
                )
                return final_response
                
            elif agent_decision.status == "NEED_MORE_WORK":
                # Act: 도구 실행
                if agent_decision.tool_calls:
                    tool_results = self._execute_planned_tools(agent_decision.tool_calls)
                
                # Observe: 결과 반영
                iteration_entry = IterationEntry(
                    iteration=iteration + 1,
                    thinking=agent_decision.thinking,
                    actions=agent_decision.tool_calls,
                    observations=tool_results
                )
                working_context.iteration_history.append(iteration_entry)
                working_context.accumulated_tool_results.extend(tool_results)
                
            elif agent_decision.status == "NEED_USER_HELP":
                feedback_response = agent_decision.user_feedback_request or "도움이 필요합니다."
                self.session_state.add_conversation_turn(
                    user_message=user_query,
                    assistant_response=feedback_response,
                    tool_results=working_context.accumulated_tool_results
                )
                return feedback_response
        
        # 최대 반복 도달
        max_iterations_response = self._handle_max_iterations_exceeded(working_context)
        self.session_state.add_conversation_turn(
            user_message=user_query,
            assistant_response=max_iterations_response,
            tool_results=working_context.accumulated_tool_results
        )
        return max_iterations_response

    def _think_and_decide(self, working_context: WorkingContext) -> ReActDecision:
        """현재 상황을 분석하고 다음 행동을 결정합니다
        
        Args:
            working_context: 작업 컨텍스트
            
        Returns:
            ReAct 결정 데이터 클래스
        """
        # 현재 상태 정보 수집
        current_state = self._analyze_current_state()
        
        # 대화 히스토리 컨텍스트 수집
        conversation_context = self.session_state.get_conversation_context()
        
        # ReAct 프롬프트 구성
        react_prompt = self._build_react_prompt(working_context, current_state, conversation_context)
        
        messages = [{"role": "user", "content": react_prompt}]
        
        # LLM 호출 (Structured output으로 JSON 응답 요청)
        try:
            system_instruction = self._build_react_system_prompt()
            response = self.gemini_client.query(
                messages=messages,
                system_instruction=system_instruction,
                tools=None,
                response_schema=ReActDecision
            )
            
            # Pydantic model_validate_json으로 직접 파싱
            response_text = response if isinstance(response, str) else str(response)
            decision = ReActDecision.model_validate_json(response_text)
            
            logger.debug(f"ReAct decision for iteration: {decision.status}")
            return decision
            
        except Exception as e:
            logger.error(f"Error in _think_and_decide: {e}")
            # Pydantic 파싱 실패 시 fallback으로 기존 방식 시도
            try:
                if 'response_text' in locals():
                    decision_dict = self._parse_structured_decision(response_text)
                    return ReActDecision.from_dict(decision_dict)
            except Exception:
                pass
            
            # 모든 파싱 실패 시 안전한 기본값 반환
            return ReActDecision(
                thinking=f"오류가 발생했습니다: {str(e)}",
                status="NEED_USER_HELP",
                final_response=None,
                tool_calls=None,
                user_feedback_request=f"분석 중 오류가 발생했습니다: {str(e)}. 다시 시도하거나 더 간단한 요청을 해주세요."
            )

    def _execute_planned_tools(self, tool_calls: List[ToolCallModel]) -> List[ToolExecutionResult]:
        """계획된 도구들을 실행합니다
        
        Args:
            tool_calls: 실행할 도구 호출 목록
            
        Returns:
            도구 실행 결과 목록 (ToolExecutionResult 객체들)
        """
        results = []
        
        for tool_call in tool_calls:
            try:
                tool_name = tool_call.tool
                params = tool_call.params
                rationale = tool_call.rationale
                
                if not tool_name:
                    logger.warning("도구 이름이 없는 tool_call이 있습니다")
                    continue
                
                # 도구 실행
                result = self.execute_tool(tool_name, params)
                
                # 결과 포맷팅
                tool_result = ToolExecutionResult(
                    tool=tool_name,
                    params=params,
                    rationale=rationale,
                    result=result
                )
                
                results.append(tool_result)
                
                logger.debug(f"도구 실행 완료: {tool_name} - 성공: {result.success}")
                
            except Exception as e:
                logger.error(f"도구 실행 중 오류 발생: {tool_call} - {e}")
                
                # 오류 결과도 기록
                error_result = ToolResult(
                    success=False,
                    data=None,
                    error_message=str(e)
                )
                
                tool_result = ToolExecutionResult(
                    tool=tool_call.tool,
                    params=tool_call.params,
                    rationale=tool_call.rationale,
                    result=error_result
                )
                
                results.append(tool_result)
        
        return results

    def _handle_max_iterations_exceeded(self, working_context: WorkingContext) -> str:
        """최대 반복 횟수 도달 시 처리합니다
        
        Args:
            working_context: 작업 컨텍스트
            
        Returns:
            최대 반복 도달 응답
        """
        original_query = working_context.original_query or "알 수 없는 요청"
        iteration_count = len(working_context.iteration_history)
        
        return f"죄송합니다. '{original_query}' 요청을 처리하는 데 최대 반복 횟수({iteration_count})에 도달했습니다. " \
               f"작업이 복잡하거나 추가 정보가 필요할 수 있습니다. 더 구체적으로 요청해 주시거나 단계별로 나누어 요청해 주세요."

    def _build_react_system_prompt(self) -> str:
        """ReAct 패턴용 시스템 프롬프트를 구성합니다"""
        
        available_tools = self._get_available_tools()
        
        # 도구별 상세 정보 구성 (파라미터 스키마 포함)
        tools_details = []
        for tool in available_tools:
            tool_detail = f"""
## {tool.name}
- **설명**: {tool.description}
- **파라미터**: {json.dumps(tool.parameters_schema, ensure_ascii=False, indent=2)}"""
            
            # 도구별 사용 예시 추가
            if tool.name == "list_directory":
                tool_detail += f"""
- **사용 예시**: 
  ```json
  {{
    "tool": "list_directory",
    "params": {{"directory_path": "{self.work_dir}"}},
    "rationale": "현재 작업 디렉토리의 파일 목록 확인"
  }}
  ```"""
            elif tool.name == "read_file":
                tool_detail += f"""
- **사용 예시**:
  ```json
  {{
    "tool": "read_file", 
    "params": {{"file_path": "{self.work_dir}/README.md"}},
    "rationale": "README 파일 내용 읽기"
  }}
  ```"""
            elif tool.name == "file_exists":
                tool_detail += f"""
- **사용 예시**:
  ```json
  {{
    "tool": "file_exists",
    "params": {{"file_path": "{self.work_dir}/config.json"}},
    "rationale": "설정 파일 존재 여부 확인"
  }}
  ```"""
            elif tool.name == "execute_safe_command":
                tool_detail += f"""
- **사용 예시**:
  ```json
  {{
    "tool": "execute_safe_command",
    "params": {{"command": "git status", "working_directory": "{self.work_dir}"}},
    "rationale": "Git 상태 확인"
  }}
  ```"""
            
            tools_details.append(tool_detail)
        
        tools_text = "\n".join(tools_details)
        
        return f"""당신은 ReAct 패턴을 사용하는 Selvage 평가 에이전트입니다.

# 작업 환경
- **현재 작업 디렉토리**: {self.work_dir}
- **중요**: 파일이나 디렉토리를 다룰 때는 반드시 절대 경로를 사용하세요
- **예시**: "프로젝트 파일 목록"을 요청받으면 `{self.work_dir}`를 직접 사용하세요
- **금지**: 상대 경로 '.' 사용 금지 - 항상 {self.work_dir} 사용

# 사용 가능한 도구들
{tools_text}

# ReAct 패턴 지침
사용자의 요청을 분석하고 다음 JSON 형태로 응답하세요:

1. 작업이 완료된 경우:
```json
{{
  "thinking": "상황 분석 및 추론 과정",
  "status": "TASK_COMPLETE",
  "final_response": "사용자에게 제공할 최종 응답"
}}
```

2. 더 많은 작업이 필요한 경우:
```json
{{
  "thinking": "상황 분석 및 추론 과정",
  "status": "NEED_MORE_WORK",
  "tool_calls": [
    {{
      "tool": "도구_이름",
      "params": {{"매개변수": "값"}},
      "rationale": "도구를 사용하는 이유"
    }}
  ]
}}
```

3. 사용자 도움이 필요한 경우:
```json
{{
  "thinking": "상황 분석 및 추론 과정",
  "status": "NEED_USER_HELP",
  "user_feedback_request": "사용자에게 요청할 도움 메시지"
}}
```

# 중요 원칙
- 단계별로 생각하고 행동하세요
- 이전 단계의 결과를 고려하여 다음 단계를 결정하세요
- **도구 실행 결과의 실제 데이터를 활용하여 구체적인 응답을 생성하세요**
- 도구 실행이 성공했다면 "실제 결과 데이터" 섹션의 정보를 바탕으로 정확한 내용을 제공하세요
- 추측이나 일반적인 예시가 아닌 실제 데이터를 기반으로 응답하세요
- 작업이 완료되었다고 확신할 때만 TASK_COMPLETE를 사용하세요
- 안전성을 고려하여 위험한 작업은 거부하세요"""

    def _build_react_prompt(
        self, 
        working_context: WorkingContext, 
        current_state: Dict[str, Any], 
        conversation_context: List[Dict[str, Any]]
    ) -> str:
        """ReAct 프롬프트를 구성합니다"""
        
        prompt_parts = []
        
        # 사용자 원본 질문
        prompt_parts.append(f"# 사용자 질문\n{working_context.original_query}")
        
        # 현재 상태 정보
        try:
            current_state_json = json.dumps(current_state, ensure_ascii=False, indent=2)
        except TypeError:
            current_state_json = str(current_state)
        prompt_parts.append(f"\n# 현재 상태\n{current_state_json}")
        
        # 이전 대화 컨텍스트
        if conversation_context:
            prompt_parts.append("\n# 이전 대화 맥락")
            for i, context in enumerate(conversation_context[-3:], 1):  # 최근 3개만
                prompt_parts.append(f"대화 {i}:")
                prompt_parts.append(f"  사용자: {context.get('user_message', '').strip()}")
                prompt_parts.append(f"  어시스턴트: {context.get('assistant_response', '').strip()}")
        
        # 현재 반복의 히스토리
        if working_context.iteration_history:
            prompt_parts.append("\n# 이번 요청의 진행 과정")
            for entry in working_context.iteration_history[-5:]:  # 최근 5개 반복만
                iteration = entry.iteration
                thinking = entry.thinking
                actions = entry.actions
                observations = entry.observations
                
                prompt_parts.append(f"반복 {iteration}:")
                prompt_parts.append(f"  생각: {thinking}")
                prompt_parts.append(f"  실행한 도구: {len(actions or [])}개")
                prompt_parts.append(f"  관찰 결과: {len(observations)}개")
                
                # 도구 실행 결과 상세
                for obs in observations:
                    tool_name = obs.tool
                    result = obs.result
                    if hasattr(result, 'success'):
                        status = "성공" if result.success else "실패"
                        prompt_parts.append(f"    {tool_name}: {status}")
                        
                        # 성공한 경우 실제 데이터도 포함
                        if result.success and hasattr(result, 'data') and result.data:
                            try:
                                # 데이터를 JSON으로 직렬화 시도
                                data_str = json.dumps(result.data, ensure_ascii=False, indent=6)
                            except (TypeError, ValueError):
                                # 직렬화 실패 시 문자열로 변환
                                data_str = str(result.data)
                            
                            # 데이터 크기 제한 (너무 큰 경우 요약)
                            if len(data_str) > 2000:
                                data_str = data_str[:2000] + "... (데이터가 길어서 생략됨)"
                            
                            prompt_parts.append(f"      실제 결과 데이터:")
                            prompt_parts.append(f"      {data_str}")
                        
                        # 실패한 경우 오류 메시지 포함
                        elif not result.success and hasattr(result, 'error_message') and result.error_message:
                            prompt_parts.append(f"      오류 메시지: {result.error_message}")
                    else:
                        prompt_parts.append(f"    {tool_name}: 실행됨")
        
        prompt_parts.append("\n# 지시사항\n위 정보를 바탕으로 현재 상황을 분석하고 다음 행동을 JSON 형태로 결정해주세요.")
        
        return "\n".join(prompt_parts)

    def _parse_structured_decision(self, response_text: str) -> Dict[str, Any]:
        """Structured output에서 ReAct 결정을 파싱합니다"""
        
        try:
            # Structured output은 이미 유효한 JSON이어야 함
            decision = json.loads(response_text)
            
            # 필수 필드 검증
            if "status" not in decision:
                raise ValueError("status 필드가 없습니다")
            
            if "thinking" not in decision:
                decision["thinking"] = "분석 과정이 제공되지 않았습니다"
            
            # 상태별 필수 필드 검증
            status = decision["status"]
            
            if status == "TASK_COMPLETE":
                if "final_response" not in decision:
                    raise ValueError("TASK_COMPLETE 상태에는 final_response가 필요합니다")
            
            elif status == "NEED_MORE_WORK":
                if "tool_calls" not in decision:
                    raise ValueError("NEED_MORE_WORK 상태에는 tool_calls가 필요합니다")
                if not isinstance(decision["tool_calls"], list):
                    raise ValueError("tool_calls는 리스트여야 합니다")
            
            elif status == "NEED_USER_HELP":
                if "user_feedback_request" not in decision:
                    raise ValueError("NEED_USER_HELP 상태에는 user_feedback_request가 필요합니다")
            
            else:
                raise ValueError(f"알 수 없는 status: {status}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to parse structured decision: {e}")
            logger.error(f"Response text: {response_text}")
            
            # 파싱 실패 시 기존 방식으로 fallback
            return self._parse_react_decision(response_text)

    def _parse_react_decision(self, response_text: str) -> Dict[str, Any]:
        """Fallback: 기존 방식으로 ReAct 결정을 파싱합니다"""
        
        try:
            # JSON 추출 및 파싱
            decision = self._extract_json_from_response(response_text)
            
            # 필수 필드 검증
            if "status" not in decision:
                raise ValueError("status 필드가 없습니다")
            
            if "thinking" not in decision:
                decision["thinking"] = "분석 과정이 제공되지 않았습니다"
            
            # 상태별 필수 필드 검증
            status = decision["status"]
            
            if status == "TASK_COMPLETE":
                if "final_response" not in decision:
                    raise ValueError("TASK_COMPLETE 상태에는 final_response가 필요합니다")
            
            elif status == "NEED_MORE_WORK":
                if "tool_calls" not in decision:
                    raise ValueError("NEED_MORE_WORK 상태에는 tool_calls가 필요합니다")
                if not isinstance(decision["tool_calls"], list):
                    raise ValueError("tool_calls는 리스트여야 합니다")
            
            elif status == "NEED_USER_HELP":
                if "user_feedback_request" not in decision:
                    raise ValueError("NEED_USER_HELP 상태에는 user_feedback_request가 필요합니다")
            
            else:
                raise ValueError(f"알 수 없는 status: {status}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to parse ReAct decision: {e}")
            logger.error(f"Response text: {response_text}")
            
            # 파싱 실패 시 안전한 기본값 반환
            return {
                "status": "NEED_USER_HELP",
                "thinking": f"응답 파싱에 실패했습니다: {str(e)}",
                "user_feedback_request": f"죄송합니다. 요청을 처리하는 중 내부 오류가 발생했습니다. 다시 시도해 주세요."
            }
    
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
        
        return f"""당신은 Selvage 평가 에이전트입니다. 
사용자의 요청을 분석하여 적절한 도구들을 호출해주세요.

# 작업 환경
- 현재 작업 디렉토리: {self.work_dir}
- "프로젝트"는 현재 작업 디렉토리를 의미합니다
- 상대 경로는 작업 디렉토리 기준으로 해석합니다

# 대화 맥락 활용
- 이전 대화 내용과 도구 실행 결과를 참고하여 사용자 요청을 이해하세요
- "그 파일", "해당 디렉토리" 등의 참조 표현은 이전 대화에서 언급된 대상을 의미합니다
- 사용자가 명시적으로 파일/경로를 지정하지 않았다면 이전 맥락을 활용하세요

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
                    if hasattr(part, 'function_call') and part.function_call is not None:
                        function_call = part.function_call
                        if hasattr(function_call, 'name') and function_call.name:
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
