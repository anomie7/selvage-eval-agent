"""도구 생성기

ToolGenerator 클래스 정의입니다.
"""

from typing import Any, Dict

from .tool import Tool
from .read_file_tool import ReadFileTool
from .write_file_tool import WriteFileTool
from .file_exists_tool import FileExistsTool
from .execute_safe_command_tool import ExecuteSafeCommandTool
from .list_directory_tool import ListDirectoryTool
from .review_executor_tool import ReviewExecutorTool
from .deepeval_test_case_converter_tool import DeepEvalTestCaseConverterTool
from .deepeval_executor_tool import DeepEvalExecutorTool


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
        elif tool_name == "execute_reviews":
            return ReviewExecutorTool()
        elif tool_name == "deepeval_test_case_converter":
            return DeepEvalTestCaseConverterTool()
        elif tool_name == "deepeval_executor":
            return DeepEvalExecutorTool()
        else:
            raise ValueError(f"Unknown tool: {tool_name}") 