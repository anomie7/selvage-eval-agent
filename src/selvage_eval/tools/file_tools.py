"""파일 작업 관련 도구들

파일 읽기, 쓰기 등 파일 시스템 작업을 안전하게 수행하는 도구들입니다.
"""

import os
import json
import time
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, generate_parameters_schema_from_hints


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
        if not isinstance(params, dict):
            return False
            
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
        start_time = time.time()
        
        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"File not found: {file_path}",
                    execution_time=time.time() - start_time
                )
            
            # 파일 크기 확인
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB",
                    execution_time=time.time() - start_time
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
                        execution_time=time.time() - start_time
                    )
            
            return ToolResult(
                success=True,
                data={
                    "content": content,
                    "file_path": file_path,
                    "file_size_bytes": os.path.getsize(file_path),
                    "encoding": encoding
                },
                execution_time=time.time() - start_time
            )
            
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Unable to decode file with encoding: {encoding}",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to read file: {str(e)}",
                execution_time=time.time() - start_time
            )


class WriteFileTool(Tool):
    """파일 쓰기 도구 (결과 저장용)"""
    
    @property
    def name(self) -> str:
        return "write_file"
    
    @property
    def description(self) -> str:
        return "지정된 파일에 내용을 쓰고 저장합니다"
    
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
        if not isinstance(params, dict):
            return False
            
        # 필수 파라미터 확인
        required_params = ['file_path', 'content']
        for param in required_params:
            if param not in params:
                return False
                
        # file_path 타입 확인
        if not isinstance(params['file_path'], str):
            return False
            
        # file_path 빈 문자열 확인
        if not params['file_path'].strip():
            return False
            
        # content는 모든 타입 허용 (str, dict, list 등)
        if params['content'] is None:
            return False
            
        # 선택적 파라미터 타입 확인
        if 'encoding' in params and not isinstance(params['encoding'], str):
            return False
            
        if 'create_dirs' in params and not isinstance(params['create_dirs'], bool):
            return False
            
        if 'as_json' in params and not isinstance(params['as_json'], bool):
            return False
            
        return True
    
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
        start_time = time.time()        
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
                    "bytes_written": len(content.encode(encoding)) if isinstance(content, str) else len(str(content).encode(encoding)),
                    "encoding": encoding
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to write file: {str(e)}",
                execution_time=time.time() - start_time
            )


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
        if not isinstance(params, dict):
            return False
            
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
        start_time = time.time()
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
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to check file existence: {str(e)}",
                execution_time=time.time() - start_time
            )