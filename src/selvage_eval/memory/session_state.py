"""세션 상태 관리

평가 세션의 상태를 관리합니다.
디스크 영속성과 복원 기능을 제공합니다.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent.react_types import ToolExecutionResult
import uuid
import os
import logging

# Gemini count_token API를 사용할 예정
# TODO: 추후 실제 Gemini API로 구현

logger = logging.getLogger(__name__)



class SessionState:
    """평가 세션 상태 관리"""
    
    def __init__(self, session_id: Optional[str] = None):
        """세션 상태 초기화
        
        Args:
            session_id: 세션 ID (None이면 자동 생성)
        """
        self.session_id = session_id or self._generate_session_id()
        self.start_time = datetime.now()
        self.current_phase: Optional[str] = None
        self.phase_states: Dict[str, Dict[str, Any]] = {}
        self.global_state: Dict[str, Any] = {}
        self._state_file: Optional[str] = None
        
        # 대화 히스토리 관리
        self.conversation_history: List[Dict[str, Any]] = []
        self.context_window_size: int = 8000  # 토큰 기준
        self.max_history_entries: int = 50    # 최대 대화 수
        
        # 이벤트 기반 영속화를 위한 설정
        self.auto_persist_dir: Optional[str] = None
        
        logger.info(f"Initialized session state: {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """세션 ID 자동 생성
        
        Returns:
            생성된 세션 ID (eval_YYYYMMDD_HHMMSS_uuid)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"eval_{timestamp}_{short_uuid}"
    
    def set_current_phase(self, phase: str) -> None:
        """현재 실행 중인 Phase 설정
        
        Args:
            phase: Phase 이름
        """
        self.current_phase = phase
        if phase not in self.phase_states:
            self.phase_states[phase] = {}
        
        logger.info(f"Set current phase: {phase}")
    
    def update_phase_state(self, phase: str, state_updates: Dict[str, Any]) -> None:
        """특정 Phase의 상태 업데이트
        
        Args:
            phase: Phase 이름
            state_updates: 업데이트할 상태 정보
        """
        if phase not in self.phase_states:
            self.phase_states[phase] = {}
        
        self.phase_states[phase].update(state_updates)
        logger.debug(f"Updated phase state: {phase} with {len(state_updates)} items")
        
        # 이벤트 기반 자동 영속화
        self._auto_persist_if_enabled()
    
    def get_phase_state(self, phase: str) -> Dict[str, Any]:
        """특정 Phase의 상태 조회
        
        Args:
            phase: Phase 이름
            
        Returns:
            Phase 상태 딕셔너리
        """
        return self.phase_states.get(phase, {})
    
    def update_global_state(self, state_updates: Dict[str, Any]) -> None:
        """전역 상태 업데이트
        
        Args:
            state_updates: 업데이트할 전역 상태 정보
        """
        self.global_state.update(state_updates)
        logger.debug(f"Updated global state with {len(state_updates)} items")
        
        # 이벤트 기반 자동 영속화
        self._auto_persist_if_enabled()
    
    def get_completed_phases(self) -> List[str]:
        """완료된 Phase 목록 반환
        
        Returns:
            완료된 Phase 이름 리스트
        """
        completed = []
        for phase in ["commit_collection", "review_execution", "deepeval_conversion", "analysis"]:
            if phase in self.phase_states and self.phase_states[phase].get("completed", False):
                completed.append(phase)
        return completed
    
    def mark_phase_completed(self, phase: str) -> None:
        """Phase 완료 표시
        
        Args:
            phase: 완료된 Phase 이름
        """
        self.update_phase_state(phase, {"completed": True, "completed_at": datetime.now().isoformat()})
        logger.info(f"Marked phase as completed: {phase}")
    
    def is_phase_completed(self, phase: str) -> bool:
        """Phase 완료 여부 확인
        
        Args:
            phase: 확인할 Phase 이름
            
        Returns:
            완료 여부
        """
        return self.phase_states.get(phase, {}).get("completed", False)
    
    def persist_to_disk(self, file_path: str) -> None:
        """디스크에 상태 저장
        
        Args:
            file_path: 저장할 파일 경로
        """
        try:
            state_data = {
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat(),
                "current_phase": self.current_phase,
                "phase_states": self.phase_states,
                "global_state": self.global_state,
                "conversation_history": self.conversation_history,
                "context_window_size": self.context_window_size,
                "max_history_entries": self.max_history_entries
            }
            
            # 디렉토리 생성 (필요시)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            self._state_file = file_path
            logger.info(f"Persisted session state to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to persist session state: {e}")
            raise
    
    @classmethod
    def load_from_disk(cls, file_path: str) -> 'SessionState':
        """디스크에서 상태 로드
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            로드된 SessionState 인스턴스
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # SessionState 인스턴스 생성
            session = cls(session_id=state_data["session_id"])
            session.start_time = datetime.fromisoformat(state_data["start_time"])
            session.current_phase = state_data["current_phase"]
            session.phase_states = state_data["phase_states"]
            session.global_state = state_data["global_state"]
            session._state_file = file_path
            
            # 대화 히스토리 복원 (이전 버전 호환성 고려)
            session.conversation_history = state_data.get("conversation_history", [])
            session.context_window_size = state_data.get("context_window_size", 8000)
            session.max_history_entries = state_data.get("max_history_entries", 50)
            
            
            logger.info(f"Loaded session state from: {file_path}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
            raise
    
    def set_auto_persist_dir(self, output_dir: str) -> None:
        """이벤트 기반 자동 영속화를 위한 출력 디렉토리 설정
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.auto_persist_dir = output_dir
        logger.info(f"Set auto-persist directory: {output_dir}")
    
    def _auto_persist_if_enabled(self) -> None:
        """자동 영속화가 활성화된 경우 디스크에 저장"""
        if self.auto_persist_dir:
            state_file = os.path.join(self.auto_persist_dir, f"session_state_{self.session_id}.json")
            try:
                self.persist_to_disk(state_file)
                logger.debug(f"Auto-persisted session state to: {state_file}")
            except Exception as e:
                logger.error(f"Auto-persist failed: {e}")
    
    def add_conversation_turn(self, user_message: str, assistant_response: str, 
                            tool_results: Optional[List["ToolExecutionResult"]] = None) -> None:
        """대화 턴 추가
        
        Args:
            user_message: 사용자 메시지
            assistant_response: 어시스턴트 응답
            tool_results: 도구 실행 결과 (ToolExecutionResult 객체들, 선택사항)
        """
        # ToolExecutionResult 객체들을 딕셔너리로 변환
        tool_results_dict = []
        if tool_results:
            for result in tool_results:
                if hasattr(result, 'to_dict'):  # ToolExecutionResult 객체인 경우
                    tool_results_dict.append(result.to_dict())
                else:  # 이미 딕셔너리인 경우 (기존 호환성)
                    tool_results_dict.append(result)

        turn = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_response": assistant_response,
            "tool_results": tool_results_dict,
            "turn_id": len(self.conversation_history) + 1
        }
        
        self.conversation_history.append(turn)
        
        # 최대 대화 수 제한
        if len(self.conversation_history) > self.max_history_entries:
            self.conversation_history = self.conversation_history[-self.max_history_entries:]
            logger.info(f"Trimmed conversation history to {self.max_history_entries} entries")
        
        logger.debug(f"Added conversation turn {turn['turn_id']}")
        
        # 이벤트 기반 자동 영속화
        self._auto_persist_if_enabled()
    
    def get_conversation_context(self) -> List[Dict[str, Any]]:
        """현재 컨텍스트 반환 (토큰 제한 고려)
        
        Args:
            include_tool_results: 도구 실행 결과 포함 여부
            
        Returns:
            컨텍스트에 포함할 대화 히스토리
        """
        if not self.conversation_history:
            return []
        
        # Gemini count_token API 사용 예정
        
        context_turns = []
        current_tokens = 0
        
        # 최신 대화부터 역순으로 추가
        for turn in reversed(self.conversation_history):
            turn_context = {
                "user_message": turn["user_message"],
                "assistant_response": turn["assistant_response"],
                "timestamp": turn["timestamp"],
                "tool_results": turn["tool_results"]
            }
            
            # 토큰 수 계산
            turn_tokens = self._count_tokens(turn_context)
            
            if current_tokens + turn_tokens > self.context_window_size:
                break
            
            context_turns.insert(0, turn_context)
            current_tokens += turn_tokens
        
        logger.debug(f"Context includes {len(context_turns)} turns ({current_tokens} tokens)")
        return context_turns
    
    def clear_conversation_history(self) -> None:
        """대화 히스토리 초기화"""
        conversation_count = len(self.conversation_history)
        self.conversation_history.clear()
        logger.info(f"Cleared conversation history ({conversation_count} turns removed)")
        
        # 이벤트 기반 자동 영속화
        self._auto_persist_if_enabled()
    
    def get_context_stats(self) -> Dict[str, Any]:
        """컨텍스트 사용량 통계
        
        Returns:
            컨텍스트 통계 정보
        """
        total_turns = len(self.conversation_history)
        context_turns = self.get_conversation_context()
        
        # 현재 컨텍스트의 토큰 수 계산
        current_tokens = sum(self._count_tokens(turn) for turn in context_turns)
        
        return {
            "total_conversation_turns": total_turns,
            "context_turns": len(context_turns),
            "current_context_tokens": current_tokens,
            "max_context_tokens": self.context_window_size,
            "context_utilization": current_tokens / self.context_window_size if self.context_window_size > 0 else 0,
            "max_history_entries": self.max_history_entries
        }
    
    def _count_tokens(self, content: Any) -> int:
        """Gemini count_token API를 사용한 토큰 수 계산
        
        Args:
            content: 토큰 수를 계산할 컨텐츠
            
        Returns:
            계산된 토큰 수
        """
        # TODO: 추후 실제 Gemini count_token API로 구현
        # 현재는 임시 구현
        return self._estimate_tokens_fallback(content)
    
    def _estimate_tokens_fallback(self, content: Any) -> int:
        """토큰 수 대략적 추정 (fallback 구현)
        
        Args:
            content: 토큰 수를 계산할 컨텐츠
            
        Returns:
            추정된 토큰 수
        """
        # JSON 직렬화 시도, 실패시 문자열 변환
        if isinstance(content, dict):
            try:
                text = json.dumps(content, ensure_ascii=False)
            except TypeError:
                # ToolResult 등 직렬화 불가능한 객체가 있는 경우
                text = str(content)
        else:
            text = str(content)
        
        # 대략적인 추정 (영어: 4글자당 1토큰, 한국어: 2글자당 1토큰)
        korean_chars = sum(1 for c in text if ord(c) > 127)
        english_chars = len(text) - korean_chars
        
        return (english_chars // 4) + (korean_chars // 2)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """세션 요약 정보 반환
        
        Returns:
            세션 요약 딕셔너리
        """
        duration = datetime.now() - self.start_time
        completed_phases = self.get_completed_phases()
        context_stats = self.get_context_stats()
        
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "current_phase": self.current_phase,
            "completed_phases": completed_phases,
            "conversation_stats": context_stats
        }