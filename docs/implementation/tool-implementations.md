# 도구 구현

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
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string", 
                    "description": "실행할 터미널 명령어"
                },
                "cwd": {
                    "type": "string", 
                    "description": "명령어 실행 디렉토리 (선택사항)"
                },
                "timeout": {
                    "type": "integer", 
                    "description": "타임아웃 (초, 기본값: 60)"
                },
                "capture_output": {
                    "type": "boolean", 
                    "description": "출력 캡처 여부 (기본값: true)"
                }
            },
            "required": ["command"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        command = kwargs["command"]
        cwd = kwargs.get("cwd", None)
        timeout = kwargs.get("timeout", 60)
        capture_output = kwargs.get("capture_output", True)
        
        try:
            # 보안을 위한 명령어 검증
            if not self._validate_command_safety(command):
                return ToolResult(
                    success=False,
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
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string", 
                    "description": "읽을 파일의 경로"
                },
                "encoding": {
                    "type": "string", 
                    "description": "파일 인코딩 (기본값: utf-8)"
                },
                "max_size_mb": {
                    "type": "integer", 
                    "description": "최대 파일 크기 (MB, 기본값: 10)"
                },
                "as_json": {
                    "type": "boolean", 
                    "description": "JSON으로 파싱 여부 (기본값: false)"
                }
            },
            "required": ["file_path"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        file_path = kwargs["file_path"]
        encoding = kwargs.get("encoding", "utf-8")
        max_size_mb = kwargs.get("max_size_mb", 10)
        as_json = kwargs.get("as_json", False)
        
        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                return ToolResult(
                    success=False,
                    error_message=f"File not found: {file_path}"
                )
            
            # 파일 크기 확인
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return ToolResult(
                    success=False,
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
                error_message=f"Unable to decode file with encoding: {encoding}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
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
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string", 
                    "description": "쓸 파일의 경로"
                },
                "content": {
                    "type": "string", 
                    "description": "파일에 쓸 내용"
                },
                "encoding": {
                    "type": "string", 
                    "description": "파일 인코딩 (기본값: utf-8)"
                },
                "create_dirs": {
                    "type": "boolean", 
                    "description": "디렉토리 자동 생성 여부 (기본값: true)"
                },
                "as_json": {
                    "type": "boolean", 
                    "description": "JSON으로 직렬화 여부 (기본값: false)"
                }
            },
            "required": ["file_path", "content"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        file_path = kwargs["file_path"]
        content = kwargs["content"]
        encoding = kwargs.get("encoding", "utf-8")
        create_dirs = kwargs.get("create_dirs", True)
        as_json = kwargs.get("as_json", False)
        
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
        return {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string", 
                    "description": "조회할 디렉토리 경로"
                },
                "recursive": {
                    "type": "boolean", 
                    "description": "하위 디렉토리 포함 여부 (기본값: false)"
                },
                "include_hidden": {
                    "type": "boolean", 
                    "description": "숨김 파일 포함 여부 (기본값: false)"
                }
            },
            "required": ["directory_path"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        directory_path = kwargs["directory_path"]
        recursive = kwargs.get("recursive", False)
        include_hidden = kwargs.get("include_hidden", False)
        
        try:
            if not os.path.exists(directory_path):
                return ToolResult(
                    success=False,
                    error_message=f"Directory not found: {directory_path}"
                )
            
            if not os.path.isdir(directory_path):
                return ToolResult(
                    success=False,
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
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string", 
                    "description": "확인할 파일/디렉토리 경로"
                }
            },
            "required": ["file_path"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        file_path = kwargs["file_path"]
        
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
                error_message=f"Failed to check file existence: {str(e)}"
            )
```