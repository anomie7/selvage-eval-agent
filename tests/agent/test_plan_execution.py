"""plan_execution 메서드 단위 테스트

LLM 기반 실행 계획 수립 메서드를 테스트합니다.
현재 임시 구현(_create_simple_plan)과 향후 LLM 연동 부분을 모두 다룹니다.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from selvage_eval.agent.core_agent import SelvageEvaluationAgent
from selvage_eval.config.settings import EvaluationConfig
from selvage_eval.tools.execution_plan import ExecutionPlan
from selvage_eval.tools.tool_call import ToolCall


class TestPlanExecution:
    """plan_execution 메서드 테스트 클래스"""

    @pytest.fixture
    def mock_config(self, temp_dir):
        """모킹된 설정 객체"""
        config = Mock(spec=EvaluationConfig)
        config.agent_model = "gemini-2.5-pro"
        
        # evaluation 속성을 Mock으로 설정
        evaluation_mock = Mock()
        evaluation_mock.output_dir = str(temp_dir)
        config.evaluation = evaluation_mock
        
        config.get_output_path = Mock(side_effect=lambda *args: str(temp_dir / "_".join(args)))
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

    def test_plan_execution_with_gemini_success(self, agent):
        """Gemini API Function Calling을 통한 성공적인 계획 수립 테스트"""
        # Given
        user_query = "프로젝트의 최신 커밋들을 분석해주세요"
        
        # Function calling 응답 모킹
        mock_function_call1 = MagicMock()
        mock_function_call1.name = "execute_safe_command"
        mock_function_call1.args = {"command": "git log --oneline -10"}
        
        mock_function_call2 = MagicMock()
        mock_function_call2.name = "read_file"
        mock_function_call2.args = {"file_path": "./README.md"}
        
        mock_part1 = MagicMock()
        mock_part1.function_call = mock_function_call1
        
        mock_part2 = MagicMock()
        mock_part2.function_call = mock_function_call2
        
        mock_content = MagicMock()
        mock_content.parts = [mock_part1, mock_part2]
        
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        
        # 에이전트의 gemini_client를 모킹
        with patch.object(agent, 'gemini_client') as mock_gemini:
            mock_gemini.query.return_value = mock_response
            
            # _analyze_current_state 모킹
            mock_current_state = {"session_id": "test-123", "completed_phases": []}
            with patch.object(agent, '_analyze_current_state', return_value=mock_current_state):
                # When
                result = agent.plan_execution(user_query)
                
                # Then
                assert isinstance(result, ExecutionPlan)
                assert "2개의 도구 호출" in result.intent_summary
                assert result.confidence == 0.9
                assert len(result.tool_calls) == 2
                assert result.tool_calls[0].tool == "execute_safe_command"
                assert result.tool_calls[1].tool == "read_file"
                assert "Function calling 방식" in result.safety_check
                
                # query 메서드가 tools와 함께 호출되었는지 확인
                mock_gemini.query.assert_called_once()
                call_args = mock_gemini.query.call_args
                assert 'tools' in call_args[1]  # tools 파라미터가 전달되었는지 확인

    def test_plan_execution_with_conversation_context(self, agent):
        """대화 컨텍스트를 활용한 계획 수립 테스트"""
        # Given
        user_query = "그 파일의 내용도 보여줘"
        
        # 이전 대화 컨텍스트 모킹
        conversation_context = [
            {
                "user_message": "README.md 파일이 있는지 확인해줘",
                "assistant_response": "README.md 파일이 존재합니다.",
                "tool_results": [{"tool": "file_exists", "result": {"exists": True}}]
            }
        ]
        agent.session_state.get_conversation_context.return_value = conversation_context
        
        # Function calling 응답 모킹
        mock_function_call = MagicMock()
        mock_function_call.name = "read_file"
        mock_function_call.args = {"file_path": "./README.md"}
        
        mock_part = MagicMock()
        mock_part.function_call = mock_function_call
        
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        
        with patch.object(agent, 'gemini_client') as mock_gemini:
            mock_gemini.query.return_value = mock_response
            
            with patch.object(agent, '_analyze_current_state', return_value={}):
                # When
                result = agent.plan_execution(user_query)
                
                # Then
                assert "1개의 도구 호출" in result.intent_summary
                assert result.confidence == 0.9
                assert len(result.tool_calls) == 1
                assert result.tool_calls[0].tool == "read_file"
                assert result.tool_calls[0].params["file_path"] == "./README.md"
                
                # query 메서드가 호출되었는지 확인
                mock_gemini.query.assert_called_once()

    def test_plan_execution_gemini_api_failure(self, agent):
        """Gemini API 호출 실패 테스트"""
        # Given
        user_query = "테스트 쿼리"
        
        with patch.object(agent, 'gemini_client') as mock_gemini:
            mock_gemini.query.side_effect = Exception("API rate limit exceeded")
            
            with patch.object(agent, '_analyze_current_state', return_value={}):
                # When & Then
                with pytest.raises(Exception) as exc_info:
                    agent.plan_execution(user_query)
                
                assert "API rate limit exceeded" in str(exc_info.value)

    def test_plan_execution_invalid_response_format(self, agent):
        """Function call 응답 없는 경우 테스트"""
        # Given
        user_query = "테스트 쿼리"
        
        # Function call이 없는 응답 모킹 (텍스트만 있는 경우)
        mock_response = MagicMock()
        mock_response.candidates = []  # 빈 candidates
        mock_response.text = "텍스트 응답"
        
        with patch.object(agent, 'gemini_client') as mock_gemini:
            mock_gemini.query.return_value = mock_response
            
            with patch.object(agent, '_analyze_current_state', return_value={}):
                # When
                result = agent.plan_execution(user_query)
                
                # Then
                # function call이 없으면 기본 계획이 생성됨
                assert isinstance(result, ExecutionPlan)
                assert len(result.tool_calls) == 0
                assert "텍스트 응답" in result.intent_summary

    # LLM 기반 플래닝만 사용하므로 규칙 기반 테스트 제거됨


    def test_plan_execution_integration_with_current_state(self, agent):
        """현재 상태 분석과 통합된 계획 수립 테스트"""
        # Given
        user_query = "진행 상황을 알려주세요"
        
        mock_current_state = {
            "session_id": "test-123",
            "completed_phases": ["commit_collection"],
            "next_required_phase": "review_execution"
        }
        
        # LLM이 반환할 JSON 응답
        mock_llm_response = json.dumps({
            "intent_summary": "Show current progress status",
            "confidence": 0.9,
            "tool_calls": [],
            "safety_check": "Status inquiry is safe",
            "expected_outcome": "Current progress information"
        }, ensure_ascii=False)
        
        with patch.object(agent, 'gemini_client') as mock_gemini:
            mock_gemini.query.return_value = mock_llm_response
            
            with patch.object(agent, '_analyze_current_state', return_value=mock_current_state):
                # When
                result = agent.plan_execution(user_query)
                
                # Then
                assert isinstance(result, ExecutionPlan)
                # _analyze_current_state가 호출되었는지 확인
                agent._analyze_current_state.assert_called_once()

    def test_plan_execution_empty_conversation_context(self, agent):
        """빈 대화 컨텍스트에서의 계획 수립 테스트"""
        # Given
        user_query = "파일 목록을 보여주세요"
        agent.session_state.get_conversation_context.return_value = []
        
        # LLM이 반환할 JSON 응답
        mock_llm_response = json.dumps({
            "intent_summary": "List files in current directory",
            "confidence": 0.9,
            "tool_calls": [
                {
                    "tool": "list_directory",
                    "params": {"directory_path": "."},
                    "rationale": "List current directory files"
                }
            ],
            "safety_check": "Directory listing is safe",
            "expected_outcome": "File list display"
        }, ensure_ascii=False)
        
        with patch.object(agent, 'gemini_client') as mock_gemini:
            mock_gemini.query.return_value = mock_llm_response
            
            with patch.object(agent, '_analyze_current_state', return_value={}):
                # When
                result = agent.plan_execution(user_query)
                
                # Then
                assert isinstance(result, ExecutionPlan)
                # 빈 컨텍스트에서도 정상 동작하는지 확인