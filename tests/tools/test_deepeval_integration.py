"""DeepEval 도구들 통합 테스트

DeepEval 관련 도구들이 ToolGenerator와 함께 올바르게 작동하는지 테스트합니다.
"""

import pytest
from selvage_eval.tools.tool_generator import ToolGenerator
from selvage_eval.tools.deepeval_test_case_converter_tool import DeepEvalTestCaseConverterTool
from selvage_eval.tools.deepeval_executor_tool import DeepEvalExecutorTool


# 테스트 마커 정의
pytestmark = [
    pytest.mark.unit,
]


@pytest.mark.unit
class TestDeepEvalToolGeneration:
    """DeepEval 도구 생성 테스트"""
    
    def test_generate_deepeval_test_case_converter_tool(self):
        """DeepEval 테스트 케이스 변환 도구 생성 테스트"""
        generator = ToolGenerator()
        tool = generator.generate_tool("deepeval_test_case_converter", {})
        
        assert isinstance(tool, DeepEvalTestCaseConverterTool)
        assert tool.name == "deepeval_test_case_converter"
        assert "리뷰 로그를 DeepEval 테스트 케이스 형식으로 변환합니다" in tool.description
    
    def test_generate_deepeval_executor_tool(self):
        """DeepEval 실행 도구 생성 테스트"""
        generator = ToolGenerator()
        tool = generator.generate_tool("deepeval_executor", {})
        
        assert isinstance(tool, DeepEvalExecutorTool)
        assert tool.name == "deepeval_executor"
        assert "DeepEval을 사용하여 코드 리뷰 품질을 평가합니다" in tool.description
    
    @pytest.mark.parametrize("tool_name,expected_class", [
        ("deepeval_test_case_converter", DeepEvalTestCaseConverterTool),
        ("deepeval_executor", DeepEvalExecutorTool),
    ])
    def test_all_deepeval_tools_generation(self, tool_name, expected_class):
        """모든 DeepEval 도구 생성 테스트"""
        generator = ToolGenerator()
        tool = generator.generate_tool(tool_name, {})
        
        assert isinstance(tool, expected_class)
        assert tool.name == tool_name
        assert hasattr(tool, 'execute')
        assert hasattr(tool, 'validate_parameters')
        assert hasattr(tool, 'parameters_schema')
    
    def test_generator_supports_all_tools(self):
        """ToolGenerator가 모든 도구를 지원하는지 테스트"""
        generator = ToolGenerator()
        
        # 기존 도구들
        existing_tools = [
            "read_file",
            "write_file", 
            "file_exists",
            "execute_safe_command",
            "list_directory",
            "execute_reviews"
        ]
        
        # 새로 추가된 DeepEval 도구들
        deepeval_tools = [
            "deepeval_test_case_converter",
            "deepeval_executor"
        ]
        
        all_tools = existing_tools + deepeval_tools
        
        for tool_name in all_tools:
            tool = generator.generate_tool(tool_name, {})
            assert tool is not None
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'execute')
    
    def test_deepeval_tools_have_required_properties(self):
        """DeepEval 도구들이 필요한 속성들을 가지고 있는지 테스트"""
        generator = ToolGenerator()
        
        for tool_name in ["deepeval_test_case_converter", "deepeval_executor"]:
            tool = generator.generate_tool(tool_name, {})
            
            # Tool 인터페이스 필수 속성들
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'parameters_schema')
            assert hasattr(tool, 'validate_parameters')
            assert hasattr(tool, 'execute')
            
            # 실제 호출 가능한지 확인
            assert callable(tool.validate_parameters)
            assert callable(tool.execute)
            
            # 스키마가 올바른 형식인지 확인
            schema = tool.parameters_schema
            assert isinstance(schema, dict)
            assert "type" in schema
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema