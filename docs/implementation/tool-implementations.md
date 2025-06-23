# 도구 구현

## 타입 힌트 기반 자동 스키마 생성

모든 도구들은 이제 명시적 파라미터를 사용하며, 타입 힌트로부터 `parameters_schema`를 자동 생성합니다:

```python
from typing import Any, Dict, Optional
from .base import Tool, ToolResult, generate_parameters_schema_from_hints
```

## ExecuteSafeCommandTool - 제한된 안전 명령어 실행

```python
class ExecuteSafeCommandTool(Tool):
    """제한된 안전 명령어 실행 도구 (현대적 에이전트 패턴)"""
    
    def __init__(self):
        self.allowed_commands = {
            'jq', 'grep', 'find', 'ls', 'cat', 'head', 'tail', 'wc',
            'git', 'cp', 'mv', 'mkdir', 'touch'
        }
        self.allowed_paths = [
            './selvage-eval-results/',
            '/Users/demin_coder/Dev/cline',
            '/Users/demin_coder/Dev/selvage-deprecated',
            '/Users/demin_coder/Dev/ecommerce-microservices', 
            '/Users/demin_coder/Dev/kotlin-realworld'
        ]
        self.forbidden_patterns = [
            r'rm\s+', r'rmdir\s+', r'delete\s+',
            r'chmod\s+', r'chown\s+',
            r'curl\s+', r'wget\s+',
            r'sudo\s+', r'su\s+',
            r'echo\s+.*>', r'sed\s+-i', r'>\s*'
        ]
    
    @property
    def name(self) -> str:
        return "execute_safe_command"
    
    @property
    def description(self) -> str:
        return "제한된 안전 명령어를 실행합니다. 데이터 조회, 분석, 읽기 전용 Git 작업만 허용"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return generate_parameters_schema_from_hints(self.execute)
    
    def execute(self, command: str, cwd: Optional[str] = None, 
                timeout: int = 60, capture_output: bool = True) -> ToolResult:
        """제한된 안전 명령어를 실행합니다
        
        Args:
            command: 실행할 터미널 명령어
            cwd: 명령어 실행 디렉토리 (선택사항)
            timeout: 타임아웃 (초, 기본값: 60)
            capture_output: 출력 캡처 여부 (기본값: true)
            
        Returns:
            ToolResult: 명령어 실행 결과
        """
        
        try:
            # 보안을 위한 명령어 검증
            if not self._validate_command_safety(command):
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"Command blocked by safety filters: {command}"
                )
            
            # 명령어 실행
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=cwd,
                    capture_output=capture_output,
                    text=True,
                    timeout=timeout
                )
                stdout = result.stdout if result.stdout else ""
                stderr = result.stderr if result.stderr else ""
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"Command timed out after {timeout} seconds"
                )
            
            return ToolResult(
                success=returncode == 0,
                data={
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "command": command
                },
                error_message=stderr if returncode != 0 and stderr else None
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to execute command: {str(e)}"
            )
    
    def _validate_command_safety(self, command: str) -> bool:
        """현대적 에이전트 패턴의 안전성 검증"""
        import re
        import shlex
        
        # 금지된 패턴 확인
        for pattern in self.forbidden_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False
        
        # 명령어 파싱 및 허용 목록 확인
        try:
            tokens = shlex.split(command)
            if not tokens:
                return False
                
            base_command = tokens[0].split('/')[-1]  # 경로에서 명령어만 추출
            
            if base_command not in self.allowed_commands:
                return False
            
            # 특별 처리: git 명령어는 읽기 전용만 허용
            if base_command == 'git':
                if len(tokens) < 2:
                    return False
                git_subcommand = tokens[1]
                allowed_git_commands = {'log', 'show', 'diff', 'status', 'branch'}
                if git_subcommand not in allowed_git_commands:
                    return False
            
            return True
            
        except ValueError:  # shlex.split 실패
            return False
    
    def _validate_path_access(self, path: str) -> bool:
        """경로 접근 권한 검증"""
        import os
        abs_path = os.path.abspath(path)
        
        for allowed_path in self.allowed_paths:
            if abs_path.startswith(os.path.abspath(allowed_path)):
                return True
        return False
```

## ReadFileTool - 파일 읽기

```python
class ReadFileTool(Tool):
    """파일 내용 읽기 도구 (모든 Phase에서 사용)"""
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "지정된 파일의 내용을 읽어서 반환합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return generate_parameters_schema_from_hints(self.execute)
    
    def execute(self, file_path: str, encoding: str = "utf-8", 
                max_size_mb: int = 10, as_json: bool = False) -> ToolResult:
        """파일의 내용을 읽어서 반환합니다
        
        Args:
            file_path: 읽을 파일의 경로
            encoding: 파일 인코딩 (기본값: utf-8)
            max_size_mb: 최대 파일 크기 (MB, 기본값: 10)
            as_json: JSON으로 파싱 여부 (기본값: false)
            
        Returns:
            ToolResult: 파일 내용 및 메타데이터
        """
        
        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"File not found: {file_path}"
                )
            
            # 파일 크기 확인
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB"
                )
            
            # 파일 읽기
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # JSON 파싱 (필요시)
            if as_json:
                try:
                    content = json.loads(content)
                except json.JSONDecodeError as e:
                    return ToolResult(
                        success=False,
                        data=None,
                        error_message=f"Invalid JSON format: {str(e)}"
                    )
            
            return ToolResult(
                success=True,
                data={
                    "content": content,
                    "file_path": file_path,
                    "file_size_bytes": os.path.getsize(file_path),
                    "encoding": encoding
                }
            )
            
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Unable to decode file with encoding: {encoding}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to read file: {str(e)}"
            )
```

## WriteFileTool - 파일 쓰기

```python
class WriteFileTool(Tool):
    """파일 쓰기 도구 (결과 저장용)"""
    
    @property
    def name(self) -> str:
        return "write_file"
    
    @property
    def description(self) -> str:
        return "파일에 내용을 쓰고 저장합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return generate_parameters_schema_from_hints(self.execute)
    
    def execute(self, file_path: str, content: Any, encoding: str = "utf-8",
                create_dirs: bool = True, as_json: bool = False) -> ToolResult:
        """지정된 파일에 내용을 쓰고 저장합니다
        
        Args:
            file_path: 쓸 파일의 경로
            content: 파일에 쓸 내용
            encoding: 파일 인코딩 (기본값: utf-8)
            create_dirs: 상위 디렉토리 자동 생성 여부 (기본값: true)
            as_json: JSON으로 직렬화 여부 (기본값: false)
            
        Returns:
            ToolResult: 파일 쓰기 결과
        """
        
        try:
            # 디렉토리 생성 (필요시)
            if create_dirs:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # JSON 직렬화 (필요시)
            if as_json:
                content = json.dumps(content, indent=2, ensure_ascii=False)
            
            # 파일 쓰기
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                data={
                    "file_path": file_path,
                    "bytes_written": len(content.encode(encoding)),
                    "encoding": encoding
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to write file: {str(e)}"
            )
```

## ListDirectoryTool - 디렉토리 탐색

```python
class ListDirectoryTool(Tool):
    """디렉토리 내용 조회 도구"""
    
    @property
    def name(self) -> str:
        return "list_directory"
    
    @property
    def description(self) -> str:
        return "지정된 디렉토리의 파일과 폴더 목록을 반환합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return generate_parameters_schema_from_hints(self.execute)
    
    def execute(self, directory_path: str, recursive: bool = False,
                include_hidden: bool = False) -> ToolResult:
        """지정된 디렉토리의 파일과 하위 디렉토리 목록을 반환합니다
        
        Args:
            directory_path: 나열할 디렉토리 경로
            recursive: 하위 디렉토리 재귀 탐색 여부 (기본값: false)
            include_hidden: 숨겨진 파일 포함 여부 (기본값: false)
            
        Returns:
            ToolResult: 디렉토리 내용 목록
        """
        
        try:
            if not os.path.exists(directory_path):
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"Directory not found: {directory_path}"
                )
            
            if not os.path.isdir(directory_path):
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"Path is not a directory: {directory_path}"
                )
            
            files = []
            
            if recursive:
                for root, dirs, filenames in os.walk(directory_path):
                    for filename in filenames:
                        if not include_hidden and filename.startswith('.'):
                            continue
                        file_path = os.path.join(root, filename)
                        files.append({
                            "name": filename,
                            "path": file_path,
                            "type": "file",
                            "size": os.path.getsize(file_path)
                        })
                    for dirname in dirs:
                        if not include_hidden and dirname.startswith('.'):
                            continue
                        dir_path = os.path.join(root, dirname)
                        files.append({
                            "name": dirname,
                            "path": dir_path,
                            "type": "directory"
                        })
            else:
                for item in os.listdir(directory_path):
                    if not include_hidden and item.startswith('.'):
                        continue
                    item_path = os.path.join(directory_path, item)
                    if os.path.isfile(item_path):
                        files.append({
                            "name": item,
                            "path": item_path,
                            "type": "file",
                            "size": os.path.getsize(item_path)
                        })
                    elif os.path.isdir(item_path):
                        files.append({
                            "name": item,
                            "path": item_path,
                            "type": "directory"
                        })
            
            return ToolResult(
                success=True,
                data={
                    "directory": directory_path,
                    "items": files,
                    "count": len(files)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to list directory: {str(e)}"
            )
```

## FileExistsTool - 파일 존재 확인

```python
class FileExistsTool(Tool):
    """파일/디렉토리 존재 확인 도구"""
    
    @property
    def name(self) -> str:
        return "file_exists"
    
    @property
    def description(self) -> str:
        return "지정된 파일이나 디렉토리의 존재 여부를 확인합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return generate_parameters_schema_from_hints(self.execute)
    
    def execute(self, file_path: str) -> ToolResult:
        """지정된 파일 또는 디렉토리의 존재 여부를 확인합니다
        
        Args:
            file_path: 확인할 파일/디렉토리 경로
            
        Returns:
            ToolResult: 파일 존재 여부 및 정보
        """
        
        try:
            exists = os.path.exists(file_path)
            is_file = os.path.isfile(file_path) if exists else False
            is_dir = os.path.isdir(file_path) if exists else False
            
            return ToolResult(
                success=True,
                data={
                    "path": file_path,
                    "exists": exists,
                    "is_file": is_file,
                    "is_directory": is_dir,
                    "size": os.path.getsize(file_path) if is_file else None
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to check file existence: {str(e)}"
            )
```

## ToolExecutor - LLM Tool Calls 실행기

`ToolExecutor` 클래스는 LLM이 반환한 tool_calls를 파싱하여 명시적 파라미터로 변환하고 실행하는 핵심 구성요소입니다.

```python
class ToolExecutor:
    """도구 실행기 - LLM tool_calls를 파싱하여 실제 도구 함수 호출"""
    
    def __init__(self):
        self.registered_tools: Dict[str, Tool] = {}
    
    def register_tool(self, tool: Tool) -> None:
        """도구를 등록합니다"""
        self.registered_tools[tool.name] = tool
    
    def execute_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """LLM tool_call을 실행합니다
        
        Args:
            tool_name: 실행할 도구 이름
            parameters: 도구 실행에 필요한 파라미터
            
        Returns:
            ToolResult: 도구 실행 결과
        """
        if tool_name not in self.registered_tools:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Tool '{tool_name}' is not registered"
            )
        
        tool = self.registered_tools[tool_name]
        
        try:
            # 파라미터 타입 체크 및 변환
            validated_params = self._validate_and_convert_parameters(tool, parameters)
            
            # 도구 실행 (명시적 파라미터로)
            return tool.execute(**validated_params)
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Tool execution failed: {str(e)}"
            )
    
    def _validate_and_convert_parameters(self, tool: Tool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """도구의 execute 메서드 시그니처에 맞게 파라미터를 검증하고 변환합니다"""
        import inspect
        from typing import get_type_hints
        
        execute_method = tool.execute
        signature = inspect.signature(execute_method)
        type_hints = get_type_hints(execute_method)
        
        validated_params = {}
        
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
                
            param_type = type_hints.get(param_name, Any)
            
            # 필수 파라미터 체크
            if param.default == inspect.Parameter.empty:
                if param_name not in parameters:
                    raise ValueError(f"Required parameter '{param_name}' is missing")
            
            # 파라미터 값 가져오기 및 타입 변환
            if param_name in parameters:
                value = parameters[param_name]
                validated_value = self._convert_and_validate_type(value, param_type, param_name)
                validated_params[param_name] = validated_value
        
        return validated_params
    
    def _convert_and_validate_type(self, value: Any, expected_type: Type, param_name: str) -> Any:
        """값을 기대하는 타입으로 변환하고 검증합니다"""
        # 타입 변환 로직 (str, int, float, bool, Optional 등 지원)
        # 자세한 구현은 src/selvage_eval/tools/tool_executor.py 참조
        pass
```

### 사용 예시

```python
# 에이전트에서 ToolExecutor 초기화
executor = ToolExecutor()
executor.register_tool(ExecuteSafeCommandTool())
executor.register_tool(ReadFileTool())

# LLM tool_calls 실행
tool_calls = [
    {
        "tool": "execute_safe_command",
        "parameters": {
            "command": "jq '.commits[]' data.json",
            "timeout": 30
        }
    },
    {
        "tool": "read_file", 
        "parameters": {
            "file_path": "./results/summary.json",
            "as_json": True
        }
    }
]

results = executor.execute_multiple_tool_calls(tool_calls)
```

### 핵심 기능

1. **타입 안전성**: 타입 힌트 기반 파라미터 검증
2. **자동 변환**: 문자열 → 정수, 불린 등 자동 변환
3. **에러 처리**: 상세한 에러 메시지 제공
4. **유연성**: Optional 파라미터, Union 타입 지원