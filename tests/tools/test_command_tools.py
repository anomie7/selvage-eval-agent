"""command_tools.py 단위 테스트

명령어 실행 관련 도구들의 단위 테스트입니다.
"""

import pytest
import subprocess
from unittest.mock import patch, MagicMock

from selvage_eval.tools.execute_safe_command_tool import ExecuteSafeCommandTool
from selvage_eval.tools.list_directory_tool import ListDirectoryTool


@pytest.mark.unit
class TestExecuteSafeCommandTool:
    """ExecuteSafeCommandTool 단위 테스트"""
    
    def test_tool_properties(self):
        """도구 기본 속성 테스트"""
        tool = ExecuteSafeCommandTool()
        
        assert tool.name == "execute_safe_command"
        assert "제한된 안전 명령어를 실행" in tool.description
        assert "properties" in tool.parameters_schema
        assert "command" in tool.parameters_schema["properties"]
    
    @pytest.mark.parametrize("params,expected", [
        ({"command": "ls -la"}, True),
        ({"command": "git status", "cwd": "/tmp"}, True),
        ({"command": "find . -name '*.py'", "timeout": 30}, True),
        ({"command": "grep 'pattern' file.txt", "capture_output": False}, True),
        ({}, False),  # command 누락
        ({"command": ""}, False),  # 빈 명령어
        ({"command": 123}, False),  # 잘못된 타입
        ({"command": "ls", "cwd": 123}, False),  # 잘못된 cwd 타입
        ({"command": "ls", "timeout": "30"}, False),  # 잘못된 timeout 타입
        ({"command": "ls", "capture_output": "true"}, False),  # 잘못된 capture_output 타입
    ])
    def test_validate_parameters(self, params, expected):
        """파라미터 유효성 검증 테스트"""
        tool = ExecuteSafeCommandTool()
        assert tool.validate_parameters(params) == expected
    
    @pytest.mark.parametrize("command,expected", [
        ("ls -la", True),  # 허용된 명령어
        ("git status", True),  # 허용된 git 명령어
        ("git log --oneline", True),  # 허용된 git 명령어
        ("grep 'pattern' file.txt", True),  # 허용된 명령어
        ("find . -name '*.py'", True),  # 허용된 명령어
        ("jq '.key' data.json", True),  # 허용된 명령어
        ("rm file.txt", False),  # 금지된 명령어
        ("sudo ls", False),  # 금지된 명령어
        ("curl http://example.com", False),  # 금지된 명령어
        ("git push origin main", False),  # 금지된 git 명령어
        ("echo 'test' > file.txt", False),  # 출력 리다이렉션
        ("sed -i 's/old/new/' file.txt", False),  # 파일 수정
        ("python", False),  # 허용되지 않은 명령어
        ("", False),  # 빈 명령어
    ])
    def test_validate_command_safety(self, command, expected):
        """명령어 안전성 검증 테스트"""
        tool = ExecuteSafeCommandTool()
        assert tool._validate_command_safety(command) == expected
    
    @patch('subprocess.run')
    def test_execute_success(self, mock_run):
        """명령어 실행 성공 테스트"""
        # subprocess.run 모킹
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "file1.txt\nfile2.txt\n"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        tool = ExecuteSafeCommandTool()
        result = tool.execute(command="ls -la")
        
        assert result.success is True
        assert result.data["returncode"] == 0
        assert result.data["stdout"] == "file1.txt\nfile2.txt\n"
        assert result.data["stderr"] == ""
        assert result.data["command"] == "ls -la"
        assert result.error_message is None
        
        # subprocess.run이 올바른 인자로 호출되었는지 확인
        mock_run.assert_called_once_with(
            "ls -la",
            shell=True,
            cwd=None,
            capture_output=True,
            text=True,
            timeout=60
        )
    
    @patch('subprocess.run')
    def test_execute_command_failure(self, mock_run):
        """명령어 실행 실패 테스트"""
        # subprocess.run 모킹 (실패)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "ls: cannot access 'nonexistent': No such file or directory"
        mock_run.return_value = mock_result
        
        tool = ExecuteSafeCommandTool()
        result = tool.execute(command="ls nonexistent")
        
        assert result.success is False
        assert result.data["returncode"] == 1
        assert result.data["stderr"] == "ls: cannot access 'nonexistent': No such file or directory"
        assert result.error_message == "ls: cannot access 'nonexistent': No such file or directory"
    
    @patch('subprocess.run')
    def test_execute_timeout(self, mock_run):
        """명령어 타임아웃 테스트"""
        # subprocess.TimeoutExpired 예외 모킹
        mock_run.side_effect = subprocess.TimeoutExpired("ls", 30)
        
        tool = ExecuteSafeCommandTool()
        result = tool.execute(command="ls", timeout=30)
        
        assert result.success is False
        assert result.error_message is not None
        assert "Command timed out after 30 seconds" in result.error_message
    
    def test_execute_unsafe_command(self):
        """안전하지 않은 명령어 실행 테스트"""
        tool = ExecuteSafeCommandTool()
        result = tool.execute(command="rm -rf /")
        
        assert result.success is False
        assert result.error_message is not None
        assert "Command blocked by safety filters" in result.error_message
    
    @patch('subprocess.run')
    def test_execute_with_custom_parameters(self, mock_run):
        """사용자 정의 파라미터로 명령어 실행 테스트"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "current directory content"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        tool = ExecuteSafeCommandTool()
        result = tool.execute(
            command="ls",
            cwd="/tmp",
            timeout=30,
            capture_output=True
        )
        
        assert result.success is True
        
        # 올바른 파라미터로 호출되었는지 확인
        mock_run.assert_called_once_with(
            "ls",
            shell=True,
            cwd="/tmp",
            capture_output=True,
            text=True,
            timeout=30
        )
    
    @patch('subprocess.run')
    def test_execute_exception_handling(self, mock_run):
        """예외 처리 테스트"""
        mock_run.side_effect = Exception("Unexpected error")
        
        tool = ExecuteSafeCommandTool()
        result = tool.execute(command="ls")
        
        assert result.success is False
        assert result.error_message is not None
        assert "Failed to execute command" in result.error_message
    
    @pytest.mark.parametrize("git_command,expected", [
        ("git status", True),
        ("git log", True),
        ("git show", True),
        ("git diff", True),
        ("git branch", True),
        ("git add .", False),  # 쓰기 작업
        ("git commit", False),  # 쓰기 작업
        ("git push", False),  # 쓰기 작업
        ("git pull", False),  # 쓰기 작업
        ("git merge", False),  # 쓰기 작업
        ("git", False),  # subcommand 없음
    ])
    def test_git_command_validation(self, git_command, expected):
        """Git 명령어 특별 검증 테스트"""
        tool = ExecuteSafeCommandTool()
        assert tool._validate_command_safety(git_command) == expected


@pytest.mark.unit
class TestListDirectoryTool:
    """ListDirectoryTool 단위 테스트"""
    
    def test_tool_properties(self):
        """도구 기본 속성 테스트"""
        tool = ListDirectoryTool()
        
        assert tool.name == "list_directory"
        assert "디렉토리의 파일과 하위 디렉토리" in tool.description
        assert "properties" in tool.parameters_schema
        assert "directory_path" in tool.parameters_schema["properties"]
    
    @pytest.mark.parametrize("params,expected", [
        ({"directory_path": "/tmp"}, True),
        ({"directory_path": "/tmp", "recursive": True}, True),
        ({"directory_path": "/tmp", "include_hidden": False}, True),
        ({"directory_path": "/tmp", "recursive": True, "include_hidden": True}, True),
        ({}, False),  # directory_path 누락
        ({"directory_path": ""}, False),  # 빈 경로
        ({"directory_path": 123}, False),  # 잘못된 타입
        ({"directory_path": "/tmp", "recursive": "true"}, False),  # 잘못된 recursive 타입
        ({"directory_path": "/tmp", "include_hidden": "false"}, False),  # 잘못된 include_hidden 타입
    ])
    def test_validate_parameters(self, params, expected):
        """파라미터 유효성 검증 테스트"""
        tool = ListDirectoryTool()
        assert tool.validate_parameters(params) == expected
    
    def test_execute_success(self, temp_dir):
        """디렉토리 리스팅 성공 테스트"""
        # 테스트용 파일과 디렉토리 생성
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.py").write_text("content2")
        (temp_dir / "subdir1").mkdir()
        (temp_dir / "subdir2").mkdir()
        (temp_dir / ".hidden_file").write_text("hidden")
        (temp_dir / ".hidden_dir").mkdir()
        
        # 허용된 경로로 설정
        tool = ListDirectoryTool()
        tool.allowed_paths = [str(temp_dir)]
        
        result = tool.execute(directory_path=str(temp_dir))
        
        assert result.success is True
        assert result.data["directory_path"] == str(temp_dir)
        assert "file1.txt" in result.data["files"]
        assert "file2.py" in result.data["files"]
        assert "subdir1" in result.data["directories"]
        assert "subdir2" in result.data["directories"]
        # 숨겨진 파일/디렉토리는 기본적으로 제외
        assert ".hidden_file" not in result.data["files"]
        assert ".hidden_dir" not in result.data["directories"]
        assert result.data["total_items"] == 4  # 2 files + 2 directories
    
    def test_execute_include_hidden(self, temp_dir):
        """숨겨진 파일 포함 테스트"""
        # 테스트용 파일과 숨겨진 파일 생성
        (temp_dir / "visible.txt").write_text("visible")
        (temp_dir / ".hidden.txt").write_text("hidden")
        (temp_dir / "visible_dir").mkdir()
        (temp_dir / ".hidden_dir").mkdir()
        
        tool = ListDirectoryTool()
        tool.allowed_paths = [str(temp_dir)]
        
        result = tool.execute(directory_path=str(temp_dir), include_hidden=True)
        
        assert result.success is True
        assert "visible.txt" in result.data["files"]
        assert ".hidden.txt" in result.data["files"]
        assert "visible_dir" in result.data["directories"]
        assert ".hidden_dir" in result.data["directories"]
    
    def test_execute_recursive(self, temp_dir):
        """재귀 탐색 테스트"""
        # 중첩된 디렉토리 구조 생성
        (temp_dir / "file1.txt").write_text("content1")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("content2")
        nested_dir = subdir / "nested"
        nested_dir.mkdir()
        (nested_dir / "file3.txt").write_text("content3")
        
        tool = ListDirectoryTool()
        tool.allowed_paths = [str(temp_dir)]
        
        result = tool.execute(directory_path=str(temp_dir), recursive=True)
        
        assert result.success is True
        assert "file1.txt" in result.data["files"]
        assert "subdir/file2.txt" in result.data["files"] or "subdir\\file2.txt" in result.data["files"]
        assert "subdir/nested/file3.txt" in result.data["files"] or "subdir\\nested\\file3.txt" in result.data["files"]
        assert "subdir" in result.data["directories"]
        assert "subdir/nested" in result.data["directories"] or "subdir\\nested" in result.data["directories"]
    
    def test_execute_access_denied(self):
        """접근 권한 거부 테스트"""
        tool = ListDirectoryTool()
        # allowed_paths를 빈 리스트로 설정하여 모든 경로 거부
        tool.allowed_paths = []
        
        result = tool.execute(directory_path="/tmp")
        
        assert result.success is False
        assert result.error_message is not None
        assert "Access denied to directory" in result.error_message
    
    def test_execute_directory_not_found(self):
        """존재하지 않는 디렉토리 테스트"""
        tool = ListDirectoryTool()
        tool.allowed_paths = ["/nonexistent"]
        
        result = tool.execute(directory_path="/nonexistent/directory")
        
        assert result.success is False
        assert result.error_message is not None
        assert "Directory not found" in result.error_message
    
    def test_execute_not_a_directory(self, temp_file):
        """파일을 디렉토리로 지정한 경우 테스트"""
        tool = ListDirectoryTool()
        tool.allowed_paths = [str(temp_file.parent)]
        
        result = tool.execute(directory_path=str(temp_file))
        
        assert result.success is False
        assert result.error_message is not None
        assert "Path is not a directory" in result.error_message
    
    @patch('os.listdir', side_effect=PermissionError("Permission denied"))
    def test_execute_permission_error(self, mock_listdir):
        """권한 오류 테스트"""
        tool = ListDirectoryTool()
        tool.allowed_paths = ["/tmp"]
        
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True):
            result = tool.execute(directory_path="/tmp")
        
        assert result.success is False
        assert result.error_message is not None
        assert "Failed to list directory" in result.error_message
    
    @pytest.mark.parametrize("test_path,allowed_paths,expected", [
        ("/tmp/test", ["/tmp"], True),
        ("/tmp/test/subdir", ["/tmp"], True),
        ("/home/user", ["/tmp"], False),
        ("/usr/bin", ["/tmp", "/usr"], True),
        ("../outside", ["/tmp"], False),
    ])
    def test_validate_path_access(self, test_path, allowed_paths, expected):
        """경로 접근 권한 검증 테스트"""
        tool = ListDirectoryTool()
        tool.allowed_paths = allowed_paths
        assert tool._validate_path_access(test_path) == expected
    
    def test_sorted_output(self, temp_dir):
        """정렬된 출력 테스트"""
        # 알파벳 순서가 아닌 순서로 파일/디렉토리 생성
        files = ["zebra.txt", "alpha.txt", "beta.txt"]
        dirs = ["zoo", "ant", "bear"]
        
        for file in files:
            (temp_dir / file).write_text("content")
        for dir in dirs:
            (temp_dir / dir).mkdir()
        
        tool = ListDirectoryTool()
        tool.allowed_paths = [str(temp_dir)]
        
        result = tool.execute(directory_path=str(temp_dir))
        
        assert result.success is True
        # 파일과 디렉토리가 정렬되어 있는지 확인
        assert result.data["files"] == sorted(files)
        assert result.data["directories"] == sorted(dirs)