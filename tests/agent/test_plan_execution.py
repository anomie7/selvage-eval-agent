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
        """Gemini API를 통한 성공적인 계획 수립 테스트"""
        # Given
        user_query = "프로젝트의 최신 커밋들을 분석해주세요"
        
        # LLM이 반환할 JSON 응답
        mock_llm_response = json.dumps({
            "intent_summary": "Analyze recent commits",
            "confidence": 0.9,
            "tool_calls": [
                {
                    "tool": "execute_safe_command",
                    "params": {"command": "git log --oneline -10"},
                    "rationale": "Get recent commit history"
                },
                {
                    "tool": "read_file",
                    "params": {"file_path": "./README.md"},
                    "rationale": "Check project information"
                }
            ],
            "safety_check": "Read-only operations, safe",
            "expected_outcome": "Recent commit analysis"
        }, ensure_ascii=False)
        
        # 에이전트의 gemini_client를 모킹
        with patch.object(agent, 'gemini_client') as mock_gemini:
            mock_gemini.query.return_value = mock_llm_response
            
            # _analyze_current_state 모킹
            mock_current_state = {"session_id": "test-123", "completed_phases": []}
            with patch.object(agent, '_analyze_current_state', return_value=mock_current_state):
                # When
                result = agent.plan_execution(user_query)
                
                # Then
                assert isinstance(result, ExecutionPlan)
                assert result.intent_summary == "Analyze recent commits"
                assert result.confidence == 0.9
                assert len(result.tool_calls) == 2
                assert result.tool_calls[0].tool == "execute_safe_command"
                assert result.tool_calls[1].tool == "read_file"
                assert result.safety_check == "Read-only operations, safe"
                
                # query 메서드가 호출되었는지 확인
                mock_gemini.query.assert_called_once()

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
        
        # LLM이 반환할 JSON 응답
        mock_llm_response = json.dumps({
            "intent_summary": "Read README.md file based on previous context",
            "confidence": 0.95,
            "tool_calls": [
                {
                    "tool": "read_file",
                    "params": {"file_path": "./README.md"},
                    "rationale": "Read README.md file mentioned in previous conversation"
                }
            ],
            "safety_check": "Read-only operation, safe",
            "expected_outcome": "README.md file content"
        }, ensure_ascii=False)
        
        with patch.object(agent, 'gemini_client') as mock_gemini:
            mock_gemini.query.return_value = mock_llm_response
            
            with patch.object(agent, '_analyze_current_state', return_value={}):
                # When
                result = agent.plan_execution(user_query)
                
                # Then
                assert result.intent_summary == "Read README.md file based on previous context"
                assert result.confidence == 0.95
                
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
        """LLM 응답 형식 오류 테스트"""
        # Given
        user_query = "테스트 쿼리"
        
        # 잘못된 형식의 LLM 응답 (JSON 문자열로)
        invalid_llm_response = json.dumps({
            "intent": "missing required fields",  # intent_summary가 아님
            "tools": []  # tool_calls가 아님
        })
        
        with patch.object(agent, 'gemini_client') as mock_gemini:
            mock_gemini.query.return_value = invalid_llm_response
            
            with patch.object(agent, '_analyze_current_state', return_value={}):
                # When & Then
                with pytest.raises(KeyError):  # 필수 필드 누락으로 인한 오류
                    agent.plan_execution(user_query)

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