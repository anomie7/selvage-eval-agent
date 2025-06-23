"""tool_executor.py 단위 테스트

도구 실행기와 도구 생성기의 단위 테스트입니다.
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from selvage_eval.tools.tool_executor import ToolExecutor
from selvage_eval.tools.tool_generator import ToolGenerator
from selvage_eval.tools.tool_result import ToolResult
from selvage_eval.tools.tool import Tool
from selvage_eval.tools.read_file_tool import ReadFileTool
from selvage_eval.tools.write_file_tool import WriteFileTool
from selvage_eval.tools.file_exists_tool import FileExistsTool
from selvage_eval.tools.execute_safe_command_tool import ExecuteSafeCommandTool
from selvage_eval.tools.list_directory_tool import ListDirectoryTool


@pytest.mark.unit
class TestToolExecutor:
    """ToolExecutor 단위 테스트"""
    
    def test_execute_tool_call_success(self):
        """도구 호출 성공 테스트"""
        # ToolGenerator를 모킹하여 성공하는 도구 반환
        mock_tool = MagicMock(spec=Tool)
        mock_tool.validate_parameters.return_value = True
        mock_tool.execute.return_value = ToolResult(
            success=True,
            data={"content": "test file content"},
            execution_time=0.1
        )
        
        with patch.object(ToolGenerator, 'generate_tool', return_value=mock_tool):
            executor = ToolExecutor()
            result = executor.execute_tool_call(
                tool_name="read_file",
                parameters={"file_path": "/test/file.txt"}
            )
        
        assert result.success is True
        assert result.data["content"] == "test file content"
        assert result.execution_time > 0  # 실행 시간이 설정되었는지 확인
        
        # 도구가 올바르게 호출되었는지 확인
        mock_tool.validate_parameters.assert_called_once_with({"file_path": "/test/file.txt"})
        mock_tool.execute.assert_called_once_with(file_path="/test/file.txt")
    
    def test_execute_tool_call_validation_failure(self):
        """파라미터 검증 실패 테스트"""
        mock_tool = MagicMock(spec=Tool)
        mock_tool.validate_parameters.return_value = False
        
        with patch.object(ToolGenerator, 'generate_tool', return_value=mock_tool):
            executor = ToolExecutor()
            result = executor.execute_tool_call(
                tool_name="read_file",
                parameters={"invalid_param": "value"}
            )
        
        assert result.success is False
        assert result.error_message is not None
        assert "Invalid parameters for tool 'read_file'" in result.error_message
        # execution_time은 0일 수도 있음 (빠른 실행)
        
        # execute가 호출되지 않았는지 확인
        mock_tool.execute.assert_not_called()
    
    def test_execute_tool_call_tool_execution_failure(self):
        """도구 실행 실패 테스트"""
        mock_tool = MagicMock(spec=Tool)
        mock_tool.validate_parameters.return_value = True
        mock_tool.execute.return_value = ToolResult(
            success=False,
            data=None,
            error_message="파일을 찾을 수 없습니다"
        )
        
        with patch.object(ToolGenerator, 'generate_tool', return_value=mock_tool):
            executor = ToolExecutor()
            result = executor.execute_tool_call(
                tool_name="read_file",
                parameters={"file_path": "/nonexistent/file.txt"}
            )
        
        assert result.success is False
        assert result.error_message == "파일을 찾을 수 없습니다"
        assert result.execution_time > 0
    
    def test_execute_tool_call_exception_handling(self):
        """예외 처리 테스트"""
        with patch.object(ToolGenerator, 'generate_tool', side_effect=ValueError("Unknown tool")):
            executor = ToolExecutor()
            result = executor.execute_tool_call(
                tool_name="unknown_tool",
                parameters={}
            )
        
        assert result.success is False
        assert result.error_message is not None
        assert "Tool execution failed" in result.error_message
        assert "Unknown tool" in result.error_message
        assert result.execution_time > 0
    
    def test_execute_multiple_tool_calls_success(self):
        """다중 도구 호출 성공 테스트"""
        # 여러 도구 호출 데이터 준비
        tool_calls = [
            {
                "tool": "read_file",
                "parameters": {"file_path": "/test/file1.txt"}
            },
            {
                "tool": "write_file",
                "parameters": {"file_path": "/test/output.txt", "content": "test"}
            },
            {
                "tool": "file_exists",
                "parameters": {"file_path": "/test/check.txt"}
            }
        ]
        
        # 각 도구 호출에 대한 결과 모킹
        expected_results = [
            ToolResult(success=True, data={"content": "file1 content"}),
            ToolResult(success=True, data={"bytes_written": 4}),
            ToolResult(success=True, data={"exists": True, "is_file": True})
        ]
        
        with patch.object(ToolExecutor, 'execute_tool_call', side_effect=expected_results):
            executor = ToolExecutor()
            results = executor.execute_multiple_tool_calls(tool_calls)
        
        assert len(results) == 3
        assert all(result.success for result in results)
        assert results[0].data["content"] == "file1 content"
        assert results[1].data["bytes_written"] == 4
        assert results[2].data["exists"] is True
    
    def test_execute_multiple_tool_calls_partial_failure(self):
        """다중 도구 호출 부분 실패 테스트"""
        tool_calls = [
            {"tool": "read_file", "parameters": {"file_path": "/test/file1.txt"}},
            {"tool": "read_file", "parameters": {"file_path": "/nonexistent.txt"}},
            {"tool": "file_exists", "parameters": {"file_path": "/test/check.txt"}}
        ]
        
        results_mock = [
            ToolResult(success=True, data={"content": "file content"}),
            ToolResult(success=False, data=None, error_message="File not found"),
            ToolResult(success=True, data={"exists": False})
        ]
        
        with patch.object(ToolExecutor, 'execute_tool_call', side_effect=results_mock):
            executor = ToolExecutor()
            results = executor.execute_multiple_tool_calls(tool_calls)
        
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True
        assert results[1].error_message == "File not found"
    
    def test_execute_multiple_tool_calls_empty_list(self):
        """빈 도구 호출 리스트 테스트"""
        executor = ToolExecutor()
        results = executor.execute_multiple_tool_calls([])
        
        assert len(results) == 0
        assert isinstance(results, list)
    
    def test_execute_multiple_tool_calls_malformed_data(self):
        """잘못된 형식의 도구 호출 데이터 테스트"""
        tool_calls = [
            {"tool": "read_file"},  # parameters 누락
            {"parameters": {"file_path": "/test/file.txt"}},  # tool 누락
            {}  # 모든 필드 누락
        ]
        
        # execute_tool_call이 각각에 대해 적절히 처리하는지 확인
        expected_calls = [
            ("read_file", {}),
            ("", {"file_path": "/test/file.txt"}),
            ("", {})
        ]
        
        with patch.object(ToolExecutor, 'execute_tool_call') as mock_execute:
            mock_execute.return_value = ToolResult(success=False, data=None)
            
            executor = ToolExecutor()
            results = executor.execute_multiple_tool_calls(tool_calls)
            
            assert len(results) == 3
            assert mock_execute.call_count == 3
            
            # 호출된 인자들 확인
            for i, (expected_tool, expected_params) in enumerate(expected_calls):
                actual_call = mock_execute.call_args_list[i]
                assert actual_call[0] == (expected_tool, expected_params)


@pytest.mark.unit
class TestToolGenerator:
    """ToolGenerator 단위 테스트"""
    
    def test_generate_read_file_tool(self):
        """ReadFileTool 생성 테스트"""
        generator = ToolGenerator()
        tool = generator.generate_tool("read_file", {})
        
        assert isinstance(tool, ReadFileTool)
        assert tool.name == "read_file"
    
    def test_generate_write_file_tool(self):
        """WriteFileTool 생성 테스트"""
        generator = ToolGenerator()
        tool = generator.generate_tool("write_file", {})
        
        assert isinstance(tool, WriteFileTool)
        assert tool.name == "write_file"
    
    def test_generate_file_exists_tool(self):
        """FileExistsTool 생성 테스트"""
        generator = ToolGenerator()
        tool = generator.generate_tool("file_exists", {})
        
        assert isinstance(tool, FileExistsTool)
        assert tool.name == "file_exists"
    
    def test_generate_execute_safe_command_tool(self):
        """ExecuteSafeCommandTool 생성 테스트"""
        generator = ToolGenerator()
        tool = generator.generate_tool("execute_safe_command", {})
        
        assert isinstance(tool, ExecuteSafeCommandTool)
        assert tool.name == "execute_safe_command"
    
    def test_generate_list_directory_tool(self):
        """ListDirectoryTool 생성 테스트"""
        generator = ToolGenerator()
        tool = generator.generate_tool("list_directory", {})
        
        assert isinstance(tool, ListDirectoryTool)
        assert tool.name == "list_directory"
    
    def test_generate_unknown_tool(self):
        """알 수 없는 도구 생성 테스트"""
        generator = ToolGenerator()
        
        with pytest.raises(ValueError) as exc_info:
            generator.generate_tool("unknown_tool", {})
        
        assert "Unknown tool: unknown_tool" in str(exc_info.value)
    
    @pytest.mark.parametrize("tool_name,expected_class", [
        ("read_file", ReadFileTool),
        ("write_file", WriteFileTool),
        ("file_exists", FileExistsTool),
        ("execute_safe_command", ExecuteSafeCommandTool),
        ("list_directory", ListDirectoryTool),
    ])
    def test_generate_all_supported_tools(self, tool_name, expected_class):
        """지원되는 모든 도구 생성 테스트"""
        generator = ToolGenerator()
        tool = generator.generate_tool(tool_name, {})
        
        assert isinstance(tool, expected_class)
        assert tool.name == tool_name
    
    def test_generate_tool_with_parameters(self):
        """파라미터와 함께 도구 생성 테스트"""
        generator = ToolGenerator()
        params = {"file_path": "/test/file.txt", "encoding": "utf-8"}
        
        # 파라미터는 도구 생성에 영향을 주지 않아야 함
        tool = generator.generate_tool("read_file", params)
        
        assert isinstance(tool, ReadFileTool)
        assert tool.name == "read_file"
    
    def test_generator_is_stateless(self):
        """ToolGenerator가 상태를 유지하지 않는지 테스트"""
        generator = ToolGenerator()
        
        # 같은 도구를 여러 번 생성
        tool1 = generator.generate_tool("read_file", {})
        tool2 = generator.generate_tool("read_file", {})
        
        # 서로 다른 인스턴스여야 함
        assert tool1 is not tool2
        assert isinstance(tool1, ReadFileTool)
        assert isinstance(tool2, ReadFileTool)


@pytest.mark.unit
class TestToolExecutorIntegration:
    """ToolExecutor 통합 테스트"""
    
    def test_end_to_end_file_operations(self, temp_dir):
        """파일 작업 종단간 테스트"""
        executor = ToolExecutor()
        test_file = temp_dir / "integration_test.txt"
        test_content = "통합 테스트 내용"
        
        # 1. 파일이 존재하지 않는지 확인
        result1 = executor.execute_tool_call(
            "file_exists",
            {"file_path": str(test_file)}
        )
        assert result1.success is True
        assert result1.data["exists"] is False
        
        # 2. 파일 쓰기
        result2 = executor.execute_tool_call(
            "write_file",
            {"file_path": str(test_file), "content": test_content}
        )
        assert result2.success is True
        
        # 3. 파일이 존재하는지 확인
        result3 = executor.execute_tool_call(
            "file_exists",
            {"file_path": str(test_file)}
        )
        assert result3.success is True
        assert result3.data["exists"] is True
        assert result3.data["is_file"] is True
        
        # 4. 파일 읽기
        result4 = executor.execute_tool_call(
            "read_file",
            {"file_path": str(test_file)}
        )
        assert result4.success is True
        assert result4.data["content"] == test_content
    
    def test_end_to_end_directory_operations(self, temp_dir):
        """디렉토리 작업 종단간 테스트"""
        executor = ToolExecutor()
        
        # 테스트 파일들 생성
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.py").write_text("content2")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested_file.txt").write_text("nested content")
        
        # ListDirectoryTool 인스턴스의 allowed_paths 수정
        with patch.object(ListDirectoryTool, '__init__', lambda self: setattr(self, 'allowed_paths', [str(temp_dir)])):
                # 1. 디렉토리 존재 확인
                result1 = executor.execute_tool_call(
                    "file_exists",
                    {"file_path": str(temp_dir)}
                )
                assert result1.success is True
                assert result1.data["exists"] is True
                assert result1.data["is_directory"] is True
                
                # 2. 디렉토리 목록 조회
                result2 = executor.execute_tool_call(
                    "list_directory",
                    {"directory_path": str(temp_dir)}
                )
                assert result2.success is True
                assert "file1.txt" in result2.data["files"]
                assert "file2.py" in result2.data["files"]
                assert "subdir" in result2.data["directories"]
    
    def test_error_handling_chain(self):
        """오류 처리 연쇄 테스트"""
        executor = ToolExecutor()
        
        # 1. 존재하지 않는 파일 읽기 시도
        result1 = executor.execute_tool_call(
            "read_file",
            {"file_path": "/nonexistent/file.txt"}
        )
        assert result1.success is False
        assert result1.error_message is not None
        assert "File not found" in result1.error_message
        
        # 2. 잘못된 파라미터로 도구 호출
        result2 = executor.execute_tool_call(
            "read_file",
            {"invalid_param": "value"}
        )
        assert result2.success is False
        assert result2.error_message is not None
        assert "Invalid parameters" in result2.error_message
        
        # 3. 알 수 없는 도구 호출
        result3 = executor.execute_tool_call(
            "unknown_tool",
            {"param": "value"}
        )
        assert result3.success is False
        assert result3.error_message is not None
        assert "Tool execution failed" in result3.error_message