"""도구 시스템 기본 인터페이스

모든 도구의 기본 인터페이스와 공통 데이터 구조를 정의합니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class ToolResult:
    """도구 실행 결과"""
    success: bool
    data: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """도구 호출 정보"""
    tool: str
    params: Dict[str, Any]
    rationale: str


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
    def from_json(cls, json_data: Dict[str, Any]) -> 'ExecutionPlan':
        """JSON 데이터에서 ExecutionPlan 객체 생성"""
        return cls(
            intent_summary=json_data["intent_summary"],
            confidence=json_data["confidence"],
            parameters=json_data["parameters"],
            tool_calls=[ToolCall(**tc) for tc in json_data["tool_calls"]],
            safety_check=json_data["safety_check"],
            expected_outcome=json_data["expected_outcome"]
        )


class Tool(ABC):
    """모든 도구의 기본 인터페이스"""
    
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
        """매개변수 유효성 검증
        
        Args:
            params: 검증할 매개변수 딕셔너리
            
        Returns:
            bool: 유효성 검증 결과
        """
        # TODO: JSON Schema 기반 검증 구현
        return True
    
    async def execute_with_timing(self, **kwargs) -> ToolResult:
        """실행 시간 측정을 포함한 도구 실행
        
        Args:
            **kwargs: 도구 실행에 필요한 매개변수
            
        Returns:
            ToolResult: 실행 시간 정보가 포함된 결과
        """
        start_time = time.time()
        
        try:
            result = await self.execute(**kwargs)
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=str(e),
                execution_time=time.time() - start_time
            )


class PhaseExecutionError(Exception):
    """Phase 실행 중 발생하는 에러"""
    pass


class ResourceLimitExceeded(Exception):
    """리소스 한계 초과 에러"""  
    pass