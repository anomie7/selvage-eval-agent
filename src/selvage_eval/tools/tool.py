"""도구 시스템 기본 인터페이스

Tool 추상 클래스와 관련 헬퍼 함수들을 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, get_type_hints
import inspect

from .tool_result import ToolResult


class Tool(ABC):
    """모든 도구의 기본 인터페이스"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """도구 이름"""
        pass
    
    @property 
    @abstractmethod
    def description(self) -> str:
        """도구 설명"""
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """매개변수 스키마 (JSON Schema 형식)"""
        pass
    
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """매개변수 유효성 검증
        
        Args:
            params: 검증할 매개변수 딕셔너리
            
        Returns:
            bool: 유효성 검증 결과
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """도구 실행
        
        Args:
            **kwargs: 도구 실행에 필요한 파라미터
            
        Returns:
            ToolResult: 도구 실행 결과
        """
        pass


class PhaseExecutionError(Exception):
    """Phase 실행 중 발생하는 에러"""
    pass


class ResourceLimitExceeded(Exception):
    """리소스 한계 초과 에러"""  
    pass


def generate_parameters_schema_from_hints(execute_method) -> Dict[str, Any]:
    """타입 힌트로부터 parameters_schema를 자동 생성하는 헬퍼 함수
    
    Args:
        execute_method: execute 메서드 객체
        
    Returns:
        Dict[str, Any]: JSON Schema 형식의 파라미터 스키마
    """
    signature = inspect.signature(execute_method)
    type_hints = get_type_hints(execute_method)
    
    properties = {}
    required = []
    
    for param_name, param in signature.parameters.items():
        if param_name == 'self':
            continue
            
        param_type = type_hints.get(param_name, Any)
        
        # 타입별 스키마 정의
        schema_type = _get_json_schema_type(param_type)
        
        properties[param_name] = {
            "type": schema_type["type"],
            "description": f"{param_name} 파라미터"
        }
        
        # 추가 스키마 속성
        if "items" in schema_type:
            properties[param_name]["items"] = schema_type["items"]
        if "enum" in schema_type:
            properties[param_name]["enum"] = schema_type["enum"]
            
        # 필수 파라미터 확인 (기본값이 없는 경우)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
        else:
            properties[param_name]["default"] = param.default
    
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def _get_json_schema_type(python_type) -> Dict[str, Any]:
    """Python 타입을 JSON Schema 타입으로 변환
    
    Args:
        python_type: Python 타입 객체
        
    Returns:
        Dict[str, Any]: JSON Schema 타입 정의
    """
    # Union 타입 처리 (Optional 포함)
    if hasattr(python_type, '__origin__') and python_type.__origin__ is Union:
        args = python_type.__args__
        if len(args) == 2 and type(None) in args:
            # Optional 타입
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return _get_json_schema_type(non_none_type)
        else:
            # 다중 Union 타입 - 첫 번째 타입 사용
            return _get_json_schema_type(args[0])
    
    # 기본 타입 매핑
    type_mapping: Dict[Any, Dict[str, Any]] = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        List: {"type": "array"},
        Dict: {"type": "object"},
    }
    
    # 제네릭 타입 처리
    if hasattr(python_type, '__origin__'):
        origin = python_type.__origin__
        if origin in type_mapping:
            schema = type_mapping[origin].copy()
            
            # List 타입의 경우 아이템 타입 추가
            if origin is list or origin is List:
                if hasattr(python_type, '__args__') and python_type.__args__:
                    item_type = _get_json_schema_type(python_type.__args__[0])
                    schema["items"] = item_type
            
            return schema
    
    # 직접 매핑
    return type_mapping.get(python_type, {"type": "string"}) 