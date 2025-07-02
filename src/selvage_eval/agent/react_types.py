"""ReAct 패턴을 위한 데이터 타입 정의

ReAct (Reasoning and Acting) 패턴에서 사용되는 구조화된 데이터 타입들을 정의합니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

from ..tools.tool_result import ToolResult


class ToolExecutionResult(BaseModel):
    """도구 실행 결과"""
    model_config = ConfigDict(arbitrary_types_allowed=True)  # ToolResult 객체 허용
    
    tool: str = Field(description="실행된 도구 이름")
    params: Any = Field(description="도구 실행 매개변수")
    rationale: str = Field(description="도구를 사용한 이유")
    result: ToolResult = Field(description="도구 실행 결과")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (JSON 직렬화 가능)"""
        return {
            "tool": self.tool,
            "params": self.params,
            "rationale": self.rationale,
            "result": {
                "success": self.result.success,
                "data": self.result.data,
                "error_message": self.result.error_message,
                "execution_time": getattr(self.result, 'execution_time', 0.0),
                "metadata": getattr(self.result, 'metadata', {})
            }
        }


class ToolCallModel(BaseModel):
    """도구 호출 모델"""
    tool: str = Field(description="실행할 도구 이름")
    params: Any = Field(description="도구 실행 매개변수")  # Dict[str, Any] 대신 Any 사용
    rationale: str = Field(description="도구를 사용하는 이유")


class ReActDecision(BaseModel):
    """ReAct 패턴의 의사결정 결과"""
    model_config = {"extra": "forbid"}  # 예상치 못한 필드 방지
    
    thinking: str = Field(description="상황 분석 및 추론 과정")
    status: Literal["TASK_COMPLETE", "NEED_MORE_WORK", "NEED_USER_HELP"] = Field(description="현재 작업 상태")
    final_response: Optional[str] = Field(None, description="작업 완료 시 사용자에게 제공할 최종 응답")
    tool_calls: Optional[List[ToolCallModel]] = Field(None, description="실행할 도구 호출 목록")
    user_feedback_request: Optional[str] = Field(None, description="사용자에게 요청할 도움 메시지")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (기존 코드 호환성용)"""
        return self.model_dump(exclude_none=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReActDecision':
        """딕셔너리에서 생성"""
        return cls.model_validate(data)
    
    def validate(self) -> bool:
        """상태별 필수 필드 검증"""
        if self.status == "TASK_COMPLETE":
            return self.final_response is not None
        elif self.status == "NEED_MORE_WORK":
            return self.tool_calls is not None and isinstance(self.tool_calls, list)
        elif self.status == "NEED_USER_HELP":
            return self.user_feedback_request is not None
        return False


@dataclass
class IterationEntry:
    """ReAct 루프의 단일 반복 기록"""
    iteration: int
    thinking: str
    actions: List[ToolCallModel] | None
    observations: List[ToolExecutionResult]



@dataclass 
class WorkingContext:
    """ReAct 루프의 작업 컨텍스트"""
    original_query: str
    iteration_history: List[IterationEntry]
    accumulated_tool_results: List[ToolExecutionResult]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (기존 코드 호환성용)"""
        return {
            "original_query": self.original_query,
            "iteration_history": [
                {
                    "iteration": entry.iteration,
                    "thinking": entry.thinking,
                    "actions": entry.actions or [],
                    "observations": [obs.to_dict() for obs in entry.observations]
                }
                for entry in self.iteration_history
            ],
            "accumulated_tool_results": [result.to_dict() for result in self.accumulated_tool_results]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkingContext':
        """딕셔너리에서 생성 (기존 코드 호환성용)"""
        iteration_history = [
            IterationEntry(
                iteration=entry["iteration"],
                thinking=entry["thinking"],
                actions=entry["actions"],
                observations=[]  # 딕셔너리에서 ToolExecutionResult로 변환은 복잡하므로 빈 리스트로 처리
            )
            for entry in data.get("iteration_history", [])
        ]
        
        return cls(
            original_query=data.get("original_query", ""),
            iteration_history=iteration_history,
            accumulated_tool_results=[]  # 딕셔너리에서 ToolExecutionResult로 변환은 복잡하므로 빈 리스트로 처리
        )