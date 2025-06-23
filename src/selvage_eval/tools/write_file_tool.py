"""파일 쓰기 도구

WriteFileTool 클래스 정의입니다.
"""

import os
import json
from typing import Any, Dict

from .tool import Tool, generate_parameters_schema_from_hints
from .tool_result import ToolResult


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
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Failed to write file: {str(e)}",
            ) 