"""제한된 안전 명령어 실행 도구

ExecuteSafeCommandTool 클래스 정의입니다.
현대적 에이전트 패턴에 따라 화이트리스트 기반 보안을 적용합니다.
"""

import subprocess
import re
import shlex
import os
import time
from typing import Any, Dict, Optional

from .tool import Tool, generate_parameters_schema_from_hints
from .tool_result import ToolResult


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
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """파라미터 유효성 검증
        
        Args:
            params: 검증할 파라미터 딕셔너리
            
        Returns:
            bool: 유효성 검증 결과
        """
        # 필수 파라미터 확인
        if 'command' not in params:
            return False
            
        # command 타입 확인
        if not isinstance(params['command'], str):
            return False
            
        # command 빈 문자열 확인
        if not params['command'].strip():
            return False
            
        # 선택적 파라미터 타입 확인
        if 'cwd' in params and params['cwd'] is not None and not isinstance(params['cwd'], str):
            return False
            
        if 'timeout' in params and not isinstance(params['timeout'], int):
            return False
            
        if 'capture_output' in params and not isinstance(params['capture_output'], bool):
            return False
            
        return True
    
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
                    error_message=f"Command timed out after {timeout} seconds",
                )
            
            return ToolResult(
                success=returncode == 0,
                data={
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "command": command
                },
                error_message=stderr if returncode != 0 and stderr else None,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to execute command: {str(e)}",
            )
    
    def _validate_command_safety(self, command: str) -> bool:
        """현대적 에이전트 패턴의 안전성 검증"""
        
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
        abs_path = os.path.abspath(path)
        
        for allowed_path in self.allowed_paths:
            if abs_path.startswith(os.path.abspath(allowed_path)):
                return True
        return False 