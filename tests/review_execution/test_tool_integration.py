"""Tool 통합 테스트"""

import pytest
from selvage_eval.tools.tool_generator import ToolGenerator
from selvage_eval.tools.review_executor_tool import ReviewExecutorTool


class TestToolIntegration:
    """Tool 통합 테스트"""
    
    def test_tool_generator_creates_review_executor_tool(self):
        """ToolGenerator가 ReviewExecutorTool을 생성하는지 테스트"""
        generator = ToolGenerator()
        
        tool = generator.generate_tool("execute_reviews", {})
        
        assert isinstance(tool, ReviewExecutorTool)
        assert tool.name == "execute_reviews"
        assert "다중 모델로 Selvage 리뷰를 실행" in tool.description
    
    def test_tool_generator_unknown_tool(self):
        """ToolGenerator가 알 수 없는 도구에 대해 오류를 발생시키는지 테스트"""
        generator = ToolGenerator()
        
        with pytest.raises(ValueError, match="Unknown tool"):
            generator.generate_tool("unknown_tool", {})