"""파일/디렉토리 존재 확인 도구

FileExistsTool 클래스 정의입니다.
"""

import os
from typing import Any, Dict

from .tool import Tool, generate_parameters_schema_from_hints
from .tool_result import ToolResult


class FileExistsTool(Tool):
    """파일/디렉토리 존재 확인 도구"""
    
    @property
    def name(self) -> str:
        return "file_exists"
    
    @property
    def description(self) -> str:
        return "지정된 파일 또는 디렉토리의 존재 여부를 확인합니다"
    
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
        if 'file_path' not in params:
            return False
            
        # file_path 타입 확인
        if not isinstance(params['file_path'], str):
            return False
            
        # file_path 빈 문자열 확인
        if not params['file_path'].strip():
            return False
            
        return True
    
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
                    "exists": exists,
                    "is_file": is_file,
                    "is_directory": is_dir,
                    "file_path": file_path
                },
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to check file existence: {str(e)}",
            ) 