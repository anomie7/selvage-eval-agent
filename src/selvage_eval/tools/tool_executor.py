"""LLM tool_calls 파싱 및 실행을 위한 도구 실행기

LLM이 반환한 tool_calls를 파싱하여 타입 체크 후 실제 도구 함수를 호출합니다.
"""

import inspect
import time
from typing import Any, Dict, List, Type, get_type_hints, Union

from selvage_eval.tools.command_tools import ExecuteSafeCommandTool, ListDirectoryTool
from selvage_eval.tools.file_tools import FileExistsTool, ReadFileTool, WriteFileTool
from .base import Tool, ToolResult


class ToolExecutor:
    """도구 실행기 - LLM tool_calls를 파싱하여 실제 도구 함수 호출"""
    
    def execute_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """LLM tool_call을 실행합니다
        
        Args:
            tool_name: 실행할 도구 이름
            parameters: 도구 실행에 필요한 파라미터
            
        Returns:
            ToolResult: 도구 실행 결과
            
        Raises:
            ValueError: 등록되지 않은 도구이거나 파라미터 검증 실패
        """
        start_time = time.time()
        try:
            tool = ToolGenerator().generate_tool(tool_name, parameters)
            
            # 파라미터 검증
            if not tool.validate_parameters(parameters):
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"Invalid parameters for tool '{tool_name}': {parameters}",
                )
            
            result = tool.execute(**parameters)
            result.execution_time = time.time() - start_time
            return result
        
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Tool execution failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def execute_multiple_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """여러 tool_calls를 순차적으로 실행합니다
        
        Args:
            tool_calls: LLM이 반환한 tool_calls 리스트
                      각 항목은 {"tool": "tool_name", "parameters": {...}} 형식
                      
        Returns:
            List[ToolResult]: 각 도구 실행 결과 리스트
        """
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool", "")
            parameters = tool_call.get("parameters", {})
            
            result = self.execute_tool_call(tool_name, parameters)
            results.append(result)
            
        return results
    


class ToolGenerator:
    """ToolGenerator class"""

    def generate_tool(self, tool_name: str, params: Dict[str, Any]) -> Tool:
        """Generate a tool"""
        if tool_name == "read_file":
            return ReadFileTool()
        elif tool_name == "write_file":
            return WriteFileTool()
        elif tool_name == "file_exists":
            return FileExistsTool()
        elif tool_name == "execute_safe_command":
            return ExecuteSafeCommandTool()
        elif tool_name == "list_directory":
            return ListDirectoryTool()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")