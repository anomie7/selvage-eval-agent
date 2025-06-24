"""generate_response 메서드 단위 테스트

LLM 기반 최종 응답 생성 메서드를 테스트합니다.
현재 템플릿 기반 구현과 향후 LLM 연동 부분을 모두 다룹니다.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from selvage_eval.agent.core_agent import SelvageEvaluationAgent
from selvage_eval.config.settings import EvaluationConfig
from selvage_eval.tools.execution_plan import ExecutionPlan
from selvage_eval.tools.tool_call import ToolCall
from selvage_eval.tools.tool_result import ToolResult


class TestGenerateResponse:
    """generate_response 메서드 테스트 클래스"""

    @pytest.fixture
    def mock_config(self, temp_dir):
        """모킹된 설정 객체"""
        config = Mock(spec=EvaluationConfig)
        config.agent_model = "gemini-2.5-pro"
        
        # evaluation 속성을 Mock으로 설정
        evaluation_mock = Mock()
        evaluation_mock.output_dir = str(temp_dir)
        config.evaluation = evaluation_mock
        
        return config

    @pytest.fixture
    def agent(self, mock_config):
        """테스트용 에이전트 인스턴스"""
        with patch('selvage_eval.agent.core_agent.SessionState') as mock_session:
            mock_session_instance = Mock()
            mock_session_instance.session_id = "test-session-123"
            mock_session_instance.auto_persist = Mock()
            mock_session_instance.get_conversation_context = Mock(return_value=[])
            mock_session.return_value = mock_session_instance
            
            with patch.object(SelvageEvaluationAgent, '_save_session_metadata'):
                agent = SelvageEvaluationAgent(mock_config)
                agent.session_state = mock_session_instance
                return agent

    @pytest.fixture
    def sample_execution_plan(self):
        """샘플 실행 계획"""
        return ExecutionPlan(
            intent_summary="Read file content",
            confidence=0.9,
            parameters={},
            tool_calls=[
                ToolCall(
                    tool="read_file",
                    params={"file_path": "/test/sample.txt"},
                    rationale="Read sample file"
                )
            ],
            safety_check="Read-only operation, safe",
            expected_outcome="File content display"
        )

    def test_generate_response_with_gemini_success(self, agent, sample_execution_plan):
        """Gemini API를 통한 성공적인 응답 생성 테스트 (향후 구현)"""
        # Given
        user_query = "sample.txt 파일의 내용을 보여주세요"
        tool_results = [
            {
                "tool": "read_file",
                "result": ToolResult(
                    success=True,
                    data={"content": "안녕하세요\n테스트 파일입니다"},
                    error_message=None
                ),
                "rationale": "Read sample file"
            }
        ]
        
        # 예상되는 LLM 응답
        expected_llm_response = "sample.txt 파일의 내용은 다음과 같습니다:\n\n안녕하세요\n테스트 파일입니다\n\n파일을 성공적으로 읽었습니다."
        
        # 에이전트의 gemini_client를 모킹
        with patch.object(agent, 'gemini_client') as mock_gemini:
            mock_gemini.query.return_value = expected_llm_response
            
            # When
            result = agent.generate_response(user_query, sample_execution_plan, tool_results)
            
            # Then
            assert result == expected_llm_response
            
            # query 메서드가 호출되었는지 확인
            mock_gemini.query.assert_called_once()

    def test_generate_response_with_conversation_context(self, agent, sample_execution_plan):
        """대화 컨텍스트를 활용한 응답 생성 테스트"""
        # Given
        user_query = "그 파일 내용을 요약해줘"
        tool_results = []
        
        # 이전 대화 컨텍스트 모킹
        conversation_context = [
            {
                "user_message": "sample.txt 파일을 읽어줘",
                "assistant_response": "파일 내용: 안녕하세요\n테스트 파일입니다",
                "tool_results": [{"tool": "read_file", "result": {"content": "안녕하세요\n테스트 파일입니다"}}]
            }
        ]
        agent.session_state.get_conversation_context.return_value = conversation_context
        
        expected_llm_response = "이전에 읽은 sample.txt 파일의 내용을 요약하면: 간단한 인사말이 포함된 테스트 파일입니다."
        
        with patch.object(agent, 'gemini_client') as mock_gemini:
            mock_gemini.query.return_value = expected_llm_response
            
            # When
            result = agent.generate_response(user_query, sample_execution_plan, tool_results)
            
            # Then
            assert result == expected_llm_response
            
            # query 메서드가 호출되었는지 확인
            mock_gemini.query.assert_called_once()

    def test_generate_response_gemini_api_failure(self, agent, sample_execution_plan):
        """Gemini API 호출 실패 테스트"""
        # Given
        user_query = "테스트 쿼리"
        tool_results = []
        
        with patch.object(agent, 'gemini_client') as mock_gemini:
            mock_gemini.query.side_effect = Exception("API quota exceeded")
            
            # When & Then
            with pytest.raises(Exception) as exc_info:
                agent.generate_response(user_query, sample_execution_plan, tool_results)
            
            assert "API quota exceeded" in str(exc_info.value)

    def test_generate_response_empty_tool_results(self, agent, sample_execution_plan):
        """빈 도구 결과에 대한 LLM 기반 응답 생성 테스트"""
        # Given
        user_query = "안녕하세요"
        tool_results = []
        
        # LLM 응답 모킹
        expected_response = "안녕하세요! 실행할 도구가 없어서 구체적인 작업을 수행하지는 못했지만, 무엇을 도와드릴까요?"
        
        with patch.object(agent, 'gemini_client') as mock_gemini:
            mock_gemini.query.return_value = expected_response
            
            # When
            result = agent.generate_response(user_query, sample_execution_plan, tool_results)
            
            # Then
            assert result == expected_response

    # 템플릿 기반 테스트들은 제거됨 (LLM만 사용)

