"""파일 작업 관련 도구들

파일 읽기, 쓰기 등 파일 시스템 작업을 안전하게 수행하는 도구들입니다.
"""

import os
import json
from typing import Any, Dict
import aiofiles

from .base import Tool, ToolResult


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
    
    async def execute(self, **kwargs) -> ToolResult:
        file_path = kwargs["file_path"]
        encoding = kwargs.get("encoding", "utf-8")
        max_size_mb = kwargs.get("max_size_mb", 10)
        as_json = kwargs.get("as_json", False)
        
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
            async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                content = await f.read()
            
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
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string", 
                    "description": "쓸 파일의 경로"
                },
                "content": {
                    "type": ["string", "object"], 
                    "description": "파일에 쓸 내용"
                },
                "encoding": {
                    "type": "string", 
                    "description": "파일 인코딩 (기본값: utf-8)"
                },
                "create_dirs": {
                    "type": "boolean", 
                    "description": "상위 디렉토리 자동 생성 여부 (기본값: true)"
                },
                "as_json": {
                    "type": "boolean", 
                    "description": "JSON으로 직렬화 여부 (기본값: false)"
                }
            },
            "required": ["file_path", "content"]
        }
    
    async def execute(self, **kwargs) -> ToolResult:
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
            async with aiofiles.open(file_path, 'w', encoding=encoding) as f:
                await f.write(content)
            
            return ToolResult(
                success=True,
                data={
                    "file_path": file_path,
                    "bytes_written": len(content.encode(encoding)) if isinstance(content, str) else len(str(content).encode(encoding)),
                    "encoding": encoding
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to write file: {str(e)}"
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
    
    async def execute(self, **kwargs) -> ToolResult:
        file_path = kwargs["file_path"]
        
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
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to check file existence: {str(e)}"
            )