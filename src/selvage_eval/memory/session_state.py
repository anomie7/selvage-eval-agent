"""세션 상태 관리

평가 세션의 상태와 체크포인트를 관리합니다.
디스크 영속성과 복원 기능을 제공합니다.
"""

import json
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import uuid
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """체크포인트 데이터 구조"""
    checkpoint_id: str
    phase: str
    timestamp: datetime
    state: Dict[str, Any]
    metadata: Dict[str, Any]


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
        self.checkpoints: List[Checkpoint] = []
        self._state_file: Optional[str] = None
        
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
    
    def save_checkpoint(self, phase: str, state: Dict[str, Any], 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """체크포인트 저장
        
        Args:
            phase: 현재 Phase
            state: 저장할 상태 정보
            metadata: 추가 메타데이터
            
        Returns:
            생성된 체크포인트 ID
        """
        checkpoint_id = f"{phase}_{len(self.checkpoints)}"
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            phase=phase,
            timestamp=datetime.now(),
            state=state.copy(),
            metadata=metadata or {}
        )
        
        self.checkpoints.append(checkpoint)
        logger.info(f"Saved checkpoint: {checkpoint_id}")
        
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """체크포인트 복원
        
        Args:
            checkpoint_id: 복원할 체크포인트 ID
            
        Returns:
            복원된 상태 정보 또는 None
        """
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                logger.info(f"Restored checkpoint: {checkpoint_id}")
                return checkpoint.state.copy()
        
        logger.warning(f"Checkpoint not found: {checkpoint_id}")
        return None
    
    def get_latest_checkpoint(self, phase: Optional[str] = None) -> Optional[Checkpoint]:
        """최신 체크포인트 조회
        
        Args:
            phase: 특정 Phase의 체크포인트만 조회 (None이면 전체)
            
        Returns:
            최신 체크포인트 또는 None
        """
        filtered_checkpoints = self.checkpoints
        if phase:
            filtered_checkpoints = [cp for cp in self.checkpoints if cp.phase == phase]
        
        if filtered_checkpoints:
            return max(filtered_checkpoints, key=lambda cp: cp.timestamp)
        return None
    
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
    
    async def persist_to_disk(self, file_path: str) -> None:
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
                "checkpoints": [
                    {
                        "checkpoint_id": cp.checkpoint_id,
                        "phase": cp.phase,
                        "timestamp": cp.timestamp.isoformat(),
                        "state": cp.state,
                        "metadata": cp.metadata
                    }
                    for cp in self.checkpoints
                ]
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
    async def load_from_disk(cls, file_path: str) -> 'SessionState':
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
            
            # 체크포인트 복원
            session.checkpoints = [
                Checkpoint(
                    checkpoint_id=cp_data["checkpoint_id"],
                    phase=cp_data["phase"],
                    timestamp=datetime.fromisoformat(cp_data["timestamp"]),
                    state=cp_data["state"],
                    metadata=cp_data["metadata"]
                )
                for cp_data in state_data["checkpoints"]
            ]
            
            logger.info(f"Loaded session state from: {file_path}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
            raise
    
    async def auto_persist(self, output_dir: str, interval: int = 300) -> None:
        """자동 영속화 시작 (5분마다)
        
        Args:
            output_dir: 출력 디렉토리
            interval: 영속화 간격 (초)
        """
        state_file = os.path.join(output_dir, "session_state.json")
        
        async def persist_loop():
            while True:
                try:
                    await asyncio.sleep(interval)
                    await self.persist_to_disk(state_file)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Auto-persist failed: {e}")
        
        task = asyncio.create_task(persist_loop())
        logger.info(f"Started auto-persist (interval: {interval}s)")
        return task
    
    def get_session_summary(self) -> Dict[str, Any]:
        """세션 요약 정보 반환
        
        Returns:
            세션 요약 딕셔너리
        """
        duration = datetime.now() - self.start_time
        completed_phases = self.get_completed_phases()
        
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "current_phase": self.current_phase,
            "completed_phases": completed_phases,
            "total_checkpoints": len(self.checkpoints),
            "last_checkpoint": self.get_latest_checkpoint().checkpoint_id if self.checkpoints else None
        }