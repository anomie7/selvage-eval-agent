"""디렉토리 내용 나열 도구

ListDirectoryTool 클래스 정의입니다.
"""

import os
import time
from typing import Any, Dict

from .tool import Tool, generate_parameters_schema_from_hints
from .tool_result import ToolResult


class ListDirectoryTool(Tool):
    """디렉토리 내용 나열 도구"""
    
    def __init__(self):
        self.allowed_paths = [
            './selvage-eval-results/',
            '/Users/demin_coder/Dev/cline',
            '/Users/demin_coder/Dev/selvage-deprecated',
            '/Users/demin_coder/Dev/ecommerce-microservices', 
            '/Users/demin_coder/Dev/kotlin-realworld'
        ]
    
    @property
    def name(self) -> str:
        return "list_directory"
    
    @property
    def description(self) -> str:
        return "지정된 디렉토리의 파일과 하위 디렉토리 목록을 반환합니다"
    
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
        if 'directory_path' not in params:
            return False
            
        # directory_path 타입 확인
        if not isinstance(params['directory_path'], str):
            return False
            
        # directory_path 빈 문자열 확인
        if not params['directory_path'].strip():
            return False
            
        # 선택적 파라미터 타입 확인
        if 'recursive' in params and not isinstance(params['recursive'], bool):
            return False
            
        if 'include_hidden' in params and not isinstance(params['include_hidden'], bool):
            return False
            
        return True
    
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
        start_time = time.time()
        try:
            # 경로 접근 권한 확인
            if not self._validate_path_access(directory_path):
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"Access denied to directory: {directory_path}",
                    execution_time=time.time() - start_time
                )
            
            # 디렉토리 존재 확인
            if not os.path.exists(directory_path):
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"Directory not found: {directory_path}",
                    execution_time=time.time() - start_time
                )
            
            if not os.path.isdir(directory_path):
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"Path is not a directory: {directory_path}",
                    execution_time=time.time() - start_time
                )
            
            files = []
            directories = []
            
            if recursive:
                for root, dirs, filenames in os.walk(directory_path):
                    for filename in filenames:
                        if not include_hidden and filename.startswith('.'):
                            continue
                        file_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(file_path, directory_path)
                        files.append(rel_path)
                    
                    for dirname in dirs:
                        if not include_hidden and dirname.startswith('.'):
                            continue
                        dir_path = os.path.join(root, dirname)
                        rel_path = os.path.relpath(dir_path, directory_path)
                        directories.append(rel_path)
            else:
                for item in os.listdir(directory_path):
                    if not include_hidden and item.startswith('.'):
                        continue
                        
                    item_path = os.path.join(directory_path, item)
                    if os.path.isfile(item_path):
                        files.append(item)
                    elif os.path.isdir(item_path):
                        directories.append(item)
            
            return ToolResult(
                success=True,
                data={
                    "directory_path": directory_path,
                    "files": sorted(files),
                    "directories": sorted(directories),
                    "total_items": len(files) + len(directories)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to list directory: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _validate_path_access(self, path: str) -> bool:
        """경로 접근 권한 검증"""
        abs_path = os.path.abspath(path)
        
        for allowed_path in self.allowed_paths:
            if abs_path.startswith(os.path.abspath(allowed_path)):
                return True
        return False 