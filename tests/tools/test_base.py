"""base.py 단위 테스트

기본 인터페이스, 데이터 클래스 및 헬퍼 함수들을 테스트합니다.
"""

import pytest
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass

from selvage_eval.tools.base import (
    ToolResult, ToolCall, ExecutionPlan, Tool,
    generate_parameters_schema_from_hints, _get_json_schema_type
)


@pytest.mark.unit
class TestToolResult:
    """ToolResult 데이터클래스 테스트"""
    
    def test_tool_result_creation(self):
        """ToolResult 객체 생성 테스트"""
        result = ToolResult(
            success=True,
            data={"test": "data"},
            error_message=None,
            execution_time=1.5,
            metadata={"extra": "info"}
        )
        
        assert result.success is True
        assert result.data == {"test": "data"}
        assert result.error_message is None
        assert result.execution_time == 1.5
        assert result.metadata == {"extra": "info"}
    
    def test_tool_result_defaults(self):
        """ToolResult 기본값 테스트"""
        result = ToolResult(success=False, data=None)
        
        assert result.success is False
        assert result.data is None
        assert result.error_message is None
        assert result.execution_time == 0.0
        assert result.metadata == {}
    
    def test_tool_result_with_error(self):
        """오류가 있는 ToolResult 테스트"""
        result = ToolResult(
            success=False,
            data=None,
            error_message="파일을 찾을 수 없습니다"
        )
        
        assert result.success is False
        assert result.data is None
        assert result.error_message == "파일을 찾을 수 없습니다"


@pytest.mark.unit 
class TestToolCall:
    """ToolCall 데이터클래스 테스트"""
    
    def test_tool_call_creation(self):
        """ToolCall 객체 생성 테스트"""
        tool_call = ToolCall(
            tool="read_file",
            params={"file_path": "/test/file.txt"},
            rationale="파일 내용을 읽기 위해"
        )
        
        assert tool_call.tool == "read_file"
        assert tool_call.params == {"file_path": "/test/file.txt"}
        assert tool_call.rationale == "파일 내용을 읽기 위해"


@pytest.mark.unit
class TestExecutionPlan:
    """ExecutionPlan 데이터클래스 테스트"""
    
    def test_execution_plan_creation(self):
        """ExecutionPlan 객체 생성 테스트"""
        tool_calls = [
            ToolCall("read_file", {"file_path": "/test/file.txt"}, "파일 읽기"),
            ToolCall("write_file", {"file_path": "/test/output.txt"}, "결과 저장")
        ]
        
        plan = ExecutionPlan(
            intent_summary="파일 처리 작업",
            confidence=0.9,
            parameters={"mode": "safe"},
            tool_calls=tool_calls,
            safety_check="안전한 읽기 전용 작업",
            expected_outcome="파일 내용 처리 완료"
        )
        
        assert plan.intent_summary == "파일 처리 작업"
        assert plan.confidence == 0.9
        assert plan.parameters == {"mode": "safe"}
        assert len(plan.tool_calls) == 2
        assert plan.safety_check == "안전한 읽기 전용 작업"
        assert plan.expected_outcome == "파일 내용 처리 완료"
    
    def test_execution_plan_from_json(self):
        """ExecutionPlan.from_json 메서드 테스트"""
        json_data = {
            "intent_summary": "상태 확인",
            "confidence": 0.8,
            "parameters": {"check_type": "status"},
            "tool_calls": [
                {
                    "tool": "read_file",
                    "params": {"file_path": "/status.json"},
                    "rationale": "상태 파일 읽기"
                }
            ],
            "safety_check": "읽기 전용 작업",
            "expected_outcome": "현재 상태 정보"
        }
        
        plan = ExecutionPlan.from_json(json_data)
        
        assert plan.intent_summary == "상태 확인"
        assert plan.confidence == 0.8
        assert plan.parameters == {"check_type": "status"}
        assert len(plan.tool_calls) == 1
        assert plan.tool_calls[0].tool == "read_file"
        assert plan.tool_calls[0].params == {"file_path": "/status.json"}
        assert plan.tool_calls[0].rationale == "상태 파일 읽기"


@pytest.mark.unit
class TestGenerateParametersSchema:
    """generate_parameters_schema_from_hints 함수 테스트"""
    
    def test_simple_types(self):
        """기본 타입 스키마 생성 테스트"""
        def sample_function(file_path: str, count: int, enabled: bool):
            pass
        
        schema = generate_parameters_schema_from_hints(sample_function)
        
        assert schema["type"] == "object"
        assert "file_path" in schema["properties"]
        assert schema["properties"]["file_path"]["type"] == "string"
        assert "count" in schema["properties"]
        assert schema["properties"]["count"]["type"] == "integer"
        assert "enabled" in schema["properties"]
        assert schema["properties"]["enabled"]["type"] == "boolean"
        assert set(schema["required"]) == {"file_path", "count", "enabled"}
    
    def test_optional_types(self):
        """Optional 타입 스키마 생성 테스트"""
        def sample_function(file_path: str, encoding: Optional[str] = "utf-8"):
            pass
        
        schema = generate_parameters_schema_from_hints(sample_function)
        
        assert "file_path" in schema["required"]
        assert "encoding" not in schema["required"]
        assert schema["properties"]["encoding"]["default"] == "utf-8"
        assert schema["properties"]["encoding"]["type"] == "string"
    
    def test_list_and_dict_types(self):
        """List와 Dict 타입 스키마 생성 테스트"""
        def sample_function(files: List[str], config: Dict[str, Any]):
            pass
        
        schema = generate_parameters_schema_from_hints(sample_function)
        
        assert schema["properties"]["files"]["type"] == "array"
        assert schema["properties"]["files"]["items"]["type"] == "string"
        assert schema["properties"]["config"]["type"] == "object"
    
    def test_exclude_self_parameter(self):
        """self 파라미터 제외 테스트"""
        class MockTool:
            def execute(self, file_path: str, count: int):
                pass
        
        tool = MockTool()
        schema = generate_parameters_schema_from_hints(tool.execute)
        
        assert "self" not in schema["properties"]
        assert "file_path" in schema["properties"]
        assert "count" in schema["properties"]


@pytest.mark.unit
class TestGetJsonSchemaType:
    """_get_json_schema_type 함수 테스트"""
    
    @pytest.mark.parametrize("python_type,expected", [
        (str, {"type": "string"}),
        (int, {"type": "integer"}),
        (float, {"type": "number"}),
        (bool, {"type": "boolean"}),
        (list, {"type": "array"}),
        (dict, {"type": "object"}),
    ])
    def test_basic_types(self, python_type, expected):
        """기본 타입 변환 테스트"""
        result = _get_json_schema_type(python_type)
        assert result == expected
    
    def test_optional_type(self):
        """Optional 타입 변환 테스트"""
        result = _get_json_schema_type(Optional[str])
        assert result == {"type": "string"}
    
    def test_union_type(self):
        """Union 타입 변환 테스트 (첫 번째 타입 사용)"""
        result = _get_json_schema_type(Union[str, int])
        assert result == {"type": "string"}
    
    def test_list_with_item_type(self):
        """List[타입] 변환 테스트"""
        result = _get_json_schema_type(List[str])
        assert result == {"type": "array", "items": {"type": "string"}}
    
    def test_unknown_type_fallback(self):
        """알 수 없는 타입의 기본값 테스트"""
        @dataclass
        class CustomType:
            value: str
        
        result = _get_json_schema_type(CustomType)
        assert result == {"type": "string"}
    
    def test_nested_list_type(self):
        """중첩된 List 타입 테스트"""
        result = _get_json_schema_type(List[List[str]])
        expected = {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}}
        }
        assert result == expected


@pytest.mark.unit
class TestToolInterface:
    """Tool 추상 클래스 인터페이스 테스트"""
    
    def test_tool_interface_is_abstract(self):
        """Tool 클래스가 추상 클래스인지 확인"""
        with pytest.raises(TypeError):
            Tool()
    
    def test_tool_interface_methods(self):
        """Tool 인터페이스의 추상 메서드들 확인"""
        abstract_methods = Tool.__abstractmethods__
        expected_methods = {
            'name', 'description', 'parameters_schema', 
            'validate_parameters', 'execute'
        }
        assert abstract_methods == expected_methods


@pytest.mark.unit
class TestExceptions:
    """사용자 정의 예외 클래스 테스트"""
    
    def test_phase_execution_error(self):
        """PhaseExecutionError 예외 테스트"""
        from selvage_eval.tools.base import PhaseExecutionError
        
        with pytest.raises(PhaseExecutionError):
            raise PhaseExecutionError("Phase 실행 실패")
    
    def test_resource_limit_exceeded(self):
        """ResourceLimitExceeded 예외 테스트"""
        from selvage_eval.tools.base import ResourceLimitExceeded
        
        with pytest.raises(ResourceLimitExceeded):
            raise ResourceLimitExceeded("리소스 한계 초과")