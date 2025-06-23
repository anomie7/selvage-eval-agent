"""파일 내용 읽기 도구

ReadFileTool 클래스 정의입니다.
"""

import os
import json
from typing import Any, Dict

from .tool import Tool, generate_parameters_schema_from_hints
from .tool_result import ToolResult


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
            
        # 선택적 파라미터 타입 확인
        if 'encoding' in params and not isinstance(params['encoding'], str):
            return False
            
        if 'max_size_mb' in params and not isinstance(params['max_size_mb'], int):
            return False
            
        if 'as_json' in params and not isinstance(params['as_json'], bool):
            return False
            
        return True
    
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
                    error_message=f"File not found: {file_path}",
                )
            
            # 파일 크기 확인
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB",
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
                        error_message=f"Invalid JSON format: {str(e)}",
                    )
            
            return ToolResult(
                success=True,
                data={
                    "content": content,
                    "file_path": file_path,
                    "file_size_bytes": os.path.getsize(file_path),
                    "encoding": encoding
                },
            )
            
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Unable to decode file with encoding: {encoding}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to read file: {str(e)}",
            ) 