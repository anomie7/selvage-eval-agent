"""LLM이 생성한 실행 계획

ExecutionPlan 클래스 정의입니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from .tool_call import ToolCall


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