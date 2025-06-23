"""file_tools.py 단위 테스트

파일 작업 관련 도구들의 단위 테스트입니다.
"""

import pytest
import json
from unittest.mock import patch

from selvage_eval.tools.read_file_tool import ReadFileTool
from selvage_eval.tools.write_file_tool import WriteFileTool
from selvage_eval.tools.file_exists_tool import FileExistsTool


@pytest.mark.unit
class TestReadFileTool:
    """ReadFileTool 단위 테스트"""
    
    def test_tool_properties(self):
        """도구 기본 속성 테스트"""
        tool = ReadFileTool()
        
        assert tool.name == "read_file"
        assert "파일의 내용을 읽어서" in tool.description
        assert "properties" in tool.parameters_schema
        assert "file_path" in tool.parameters_schema["properties"]
    
    @pytest.mark.parametrize("params,expected", [
        ({"file_path": "/test/file.txt"}, True),
        ({"file_path": "/test/file.txt", "encoding": "utf-8"}, True),
        ({"file_path": "/test/file.txt", "max_size_mb": 5}, True),
        ({"file_path": "/test/file.txt", "as_json": True}, True),
        ({}, False),  # file_path 누락
        ({"file_path": ""}, False),  # 빈 경로
        ({"file_path": 123}, False),  # 잘못된 타입
        ({"file_path": "/test/file.txt", "encoding": 123}, False),  # 잘못된 encoding 타입
        ({"file_path": "/test/file.txt", "max_size_mb": "5"}, False),  # 잘못된 max_size_mb 타입
        ({"file_path": "/test/file.txt", "as_json": "true"}, False),  # 잘못된 as_json 타입
    ])
    def test_validate_parameters(self, params, expected):
        """파라미터 유효성 검증 테스트"""
        tool = ReadFileTool()
        assert tool.validate_parameters(params) == expected
    
    def test_execute_success(self, temp_file, sample_file_content):
        """파일 읽기 성공 테스트"""
        # 임시 파일에 내용 작성
        temp_file.write_text(sample_file_content, encoding='utf-8')
        
        tool = ReadFileTool()
        result = tool.execute(file_path=str(temp_file))
        
        assert result.success is True
        assert result.data["content"] == sample_file_content
        assert result.data["file_path"] == str(temp_file)
        assert result.data["encoding"] == "utf-8"
        assert "file_size_bytes" in result.data
    
    def test_execute_file_not_found(self):
        """존재하지 않는 파일 읽기 테스트"""
        tool = ReadFileTool()
        result = tool.execute(file_path="/nonexistent/file.txt")
        
        assert result.success is False
        assert result.error_message is not None
        assert "File not found" in result.error_message
    
    def test_execute_json_parsing_success(self, temp_json_file):
        """JSON 파싱 성공 테스트"""
        tool = ReadFileTool()
        result = tool.execute(file_path=str(temp_json_file), as_json=True)
        
        assert result.success is True
        assert isinstance(result.data["content"], dict)
        assert result.data["content"]["test"] == "data"
        assert result.data["content"]["number"] == 42
    
    def test_execute_json_parsing_failure(self, temp_file):
        """JSON 파싱 실패 테스트"""
        # 잘못된 JSON 내용 작성
        temp_file.write_text('{"invalid": json}', encoding='utf-8')
        
        tool = ReadFileTool()
        result = tool.execute(file_path=str(temp_file), as_json=True)
        
        assert result.success is False
        assert result.error_message is not None
        assert "Invalid JSON format" in result.error_message
    
    def test_execute_file_too_large(self, temp_file):
        """파일 크기 제한 테스트"""
        # 1KB 파일 생성
        large_content = "a" * 1024
        temp_file.write_text(large_content, encoding='utf-8')
        
        tool = ReadFileTool()
        # 최대 크기를 0MB로 설정하여 파일 크기 제한 테스트
        result = tool.execute(file_path=str(temp_file), max_size_mb=0)
        
        assert result.success is False
        assert result.error_message is not None
        assert "File too large" in result.error_message
    
    @patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"))
    def test_execute_unicode_decode_error(self, mock_open_func):
        """유니코드 디코딩 오류 테스트"""
        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=100):
            
            tool = ReadFileTool()
            result = tool.execute(file_path="/test/file.txt")
            
            assert result.success is False
            assert result.error_message is not None
            assert "Unable to decode file" in result.error_message
    
    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_execute_permission_error(self, _mock_open_func):
        """권한 오류 테스트"""
        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=100):
            
            tool = ReadFileTool()
            result = tool.execute(file_path="/test/file.txt")
            
            assert result.success is False
            assert result.error_message is not None
            assert "Failed to read file" in result.error_message


@pytest.mark.unit
class TestWriteFileTool:
    """WriteFileTool 단위 테스트"""
    
    def test_tool_properties(self):
        """도구 기본 속성 테스트"""
        tool = WriteFileTool()
        
        assert tool.name == "write_file"
        assert "파일에 내용을 쓰고" in tool.description
        assert "properties" in tool.parameters_schema
        assert "file_path" in tool.parameters_schema["properties"]
        assert "content" in tool.parameters_schema["properties"]
    
    @pytest.mark.parametrize("params,expected", [
        ({"file_path": "/test/file.txt", "content": "test"}, True),
        ({"file_path": "/test/file.txt", "content": {"key": "value"}}, True),
        ({"file_path": "/test/file.txt", "content": "test", "encoding": "utf-8"}, True),
        ({"file_path": "/test/file.txt", "content": "test", "create_dirs": False}, True),
        ({"file_path": "/test/file.txt", "content": "test", "as_json": True}, True),
        ({}, False),  # 필수 파라미터 누락
        ({"file_path": "/test/file.txt"}, False),  # content 누락
        ({"content": "test"}, False),  # file_path 누락
        ({"file_path": "", "content": "test"}, False),  # 빈 경로
        ({"file_path": 123, "content": "test"}, False),  # 잘못된 file_path 타입
        ({"file_path": "/test/file.txt", "content": None}, False),  # None content
        ({"file_path": "/test/file.txt", "content": "test", "encoding": 123}, False),  # 잘못된 encoding 타입
    ])
    def test_validate_parameters(self, params, expected):
        """파라미터 유효성 검증 테스트"""
        tool = WriteFileTool()
        assert tool.validate_parameters(params) == expected
    
    def test_execute_success(self, temp_dir):
        """파일 쓰기 성공 테스트"""
        test_file = temp_dir / "test_output.txt"
        test_content = "테스트 내용입니다\n한글도 잘 써집니다"
        
        tool = WriteFileTool()
        result = tool.execute(file_path=str(test_file), content=test_content)
        
        assert result.success is True
        assert result.data["file_path"] == str(test_file)
        assert result.data["encoding"] == "utf-8"
        assert "bytes_written" in result.data
        
        # 실제 파일이 생성되고 내용이 맞는지 확인
        assert test_file.exists()
        assert test_file.read_text(encoding='utf-8') == test_content
    
    def test_execute_create_directories(self, temp_dir):
        """디렉토리 자동 생성 테스트"""
        test_file = temp_dir / "subdir" / "nested" / "test.txt"
        test_content = "nested directory test"
        
        tool = WriteFileTool()
        result = tool.execute(file_path=str(test_file), content=test_content, create_dirs=True)
        
        assert result.success is True
        assert test_file.exists()
        assert test_file.read_text() == test_content
    
    def test_execute_json_serialization(self, temp_dir, sample_json_data):
        """JSON 직렬화 테스트"""
        test_file = temp_dir / "test_data.json"
        
        tool = WriteFileTool()
        result = tool.execute(file_path=str(test_file), content=sample_json_data, as_json=True)
        
        assert result.success is True
        assert test_file.exists()
        
        # JSON 파일이 올바르게 저장되었는지 확인
        with open(test_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        assert loaded_data == sample_json_data
    
    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_execute_permission_error(self, _mock_open_func):
        """권한 오류 테스트"""
        with patch("os.makedirs"):
            tool = WriteFileTool()
            result = tool.execute(file_path="/test/file.txt", content="test")
            
            assert result.success is False
            assert result.error_message is not None
            assert "Failed to write file" in result.error_message
    
    @patch("os.makedirs", side_effect=OSError("Cannot create directory"))
    def test_execute_directory_creation_error(self, _mock_makedirs):
        """디렉토리 생성 오류 테스트"""
        tool = WriteFileTool()
        result = tool.execute(file_path="/test/subdir/file.txt", content="test", create_dirs=True)
        
        assert result.success is False
        assert result.error_message is not None
        assert "Failed to write file" in result.error_message


@pytest.mark.unit
class TestFileExistsTool:
    """FileExistsTool 단위 테스트"""
    
    def test_tool_properties(self):
        """도구 기본 속성 테스트"""
        tool = FileExistsTool()
        
        assert tool.name == "file_exists"
        assert "파일 또는 디렉토리의 존재 여부" in tool.description
        assert "properties" in tool.parameters_schema
        assert "file_path" in tool.parameters_schema["properties"]
    
    @pytest.mark.parametrize("params,expected", [
        ({"file_path": "/test/file.txt"}, True),
        ({"file_path": "/test/directory/"}, True),
        ({}, False),  # file_path 누락
        ({"file_path": ""}, False),  # 빈 경로
        ({"file_path": 123}, False),  # 잘못된 타입
        ({"file_path": "   "}, False),  # 공백만 있는 경로
    ])
    def test_validate_parameters(self, params, expected):
        """파라미터 유효성 검증 테스트"""
        tool = FileExistsTool()
        assert tool.validate_parameters(params) == expected
    
    def test_execute_file_exists(self, temp_file):
        """존재하는 파일 확인 테스트"""
        tool = FileExistsTool()
        result = tool.execute(file_path=str(temp_file))
        
        assert result.success is True
        assert result.data["exists"] is True
        assert result.data["is_file"] is True
        assert result.data["is_directory"] is False
        assert result.data["file_path"] == str(temp_file)
    
    def test_execute_directory_exists(self, temp_dir):
        """존재하는 디렉토리 확인 테스트"""
        tool = FileExistsTool()
        result = tool.execute(file_path=str(temp_dir))
        
        assert result.success is True
        assert result.data["exists"] is True
        assert result.data["is_file"] is False
        assert result.data["is_directory"] is True
        assert result.data["file_path"] == str(temp_dir)
    
    def test_execute_file_not_exists(self):
        """존재하지 않는 파일/디렉토리 확인 테스트"""
        tool = FileExistsTool()
        result = tool.execute(file_path="/nonexistent/path")
        
        assert result.success is True
        assert result.data["exists"] is False
        assert result.data["is_file"] is False
        assert result.data["is_directory"] is False
        assert result.data["file_path"] == "/nonexistent/path"
    
    @patch("os.path.exists", side_effect=OSError("System error"))
    def test_execute_system_error(self, _mock_exists):
        """시스템 오류 테스트"""
        tool = FileExistsTool()
        result = tool.execute(file_path="/test/file.txt")
        
        assert result.success is False
        assert result.error_message is not None
        assert "Failed to check file existence" in result.error_message
    
    def test_execute_special_characters(self, temp_dir):
        """특수 문자가 포함된 경로 테스트"""
        # 한글과 특수 문자가 포함된 파일명
        special_file = temp_dir / "테스트 파일 (특수).txt"
        special_file.write_text("한글 내용", encoding='utf-8')
        
        tool = FileExistsTool()
        result = tool.execute(file_path=str(special_file))
        
        assert result.success is True
        assert result.data["exists"] is True
        assert result.data["is_file"] is True