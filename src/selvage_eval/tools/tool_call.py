"""도구 호출 정보

ToolCall 클래스 정의입니다.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ToolCall:
    """도구 호출 정보"""
    tool: str
    params: Dict[str, Any]
    rationale: str 