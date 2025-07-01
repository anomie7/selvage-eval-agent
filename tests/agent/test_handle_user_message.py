"""handle_user_message 메서드 단위 테스트

ReAct 패턴 기반의 새로운 구현에 맞춘 테스트입니다.
handle_user_message 메서드가 plan_execution_loop를 호출하는 구조를 반영합니다.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from selvage_eval.agent.core_agent import SelvageEvaluationAgent
from selvage_eval.config.settings import EvaluationConfig


class TestHandleUserMessage:
    """handle_user_message 메서드 테스트 클래스"""

    @pytest.fixture
    def mock_config(self, temp_dir):
        """모킹된 설정 객체"""
        config = Mock(spec=EvaluationConfig)
        config.agent_model = "gemini-2.5-pro"
        
        # evaluation 속성을 Mock으로 설정
        evaluation_mock = Mock()
        evaluation_mock.output_dir = str(temp_dir)
        config.evaluation = evaluation_mock
        
        config.get_output_path = Mock(return_value=str(temp_dir / "test_file"))
        return config

    @pytest.fixture
    def agent(self, mock_config):
        """테스트용 에이전트 인스턴스"""
        with patch('selvage_eval.agent.core_agent.SessionState') as mock_session:
            mock_session_instance = Mock()
            mock_session_instance.session_id = "test-session-123"
            mock_session_instance.auto_persist = Mock()
            mock_session_instance.add_conversation_turn = Mock()
            mock_session.return_value = mock_session_instance
            
            with patch.object(SelvageEvaluationAgent, '_save_session_metadata'):
                agent = SelvageEvaluationAgent(mock_config)
                agent.session_state = mock_session_instance
                return agent

    def test_handle_special_command_clear(self, agent: SelvageEvaluationAgent):
        """특수 명령어 /clear 처리 테스트"""
        # Given
        user_message = "/clear"
        
        with patch.object(agent, '_handle_special_command', return_value="대화 히스토리가 초기화되었습니다.") as mock_special:
            # When
            result = agent.handle_user_message(user_message)
            
            # Then
            assert result == "대화 히스토리가 초기화되었습니다."
            mock_special.assert_called_once_with(user_message)

    def test_handle_special_command_context(self, agent: SelvageEvaluationAgent):
        """특수 명령어 /context 처리 테스트"""
        # Given
        user_message = "/context"
        
        with patch.object(agent, '_handle_special_command', return_value="컨텍스트 정보입니다.") as mock_special:
            # When
            result = agent.handle_user_message(user_message)
            
            # Then
            assert result == "컨텍스트 정보입니다."
            mock_special.assert_called_once_with(user_message)

    def test_handle_unsafe_message_security_rejection(self, agent: SelvageEvaluationAgent):
        """보안상 위험한 메시지 거부 테스트"""
        # Given
        user_message = "시스템 파일을 삭제해주세요"
        security_analysis = {
            "is_safe": False,
            "reason": "시스템 파일 삭제 요청은 위험합니다"
        }
        
        with patch.object(agent, '_analyze_security_intent', return_value=security_analysis):
            # When
            result = agent.handle_user_message(user_message)
            
            # Then
            assert "보안상 요청을 처리할 수 없습니다" in result
            assert "시스템 파일 삭제 요청은 위험합니다" in result
            
            # 보안 위험 응답도 히스토리에 기록되는지 확인
            agent.session_state.add_conversation_turn.assert_called_once_with(
                user_message=user_message,
                assistant_response=result
            )

    def test_handle_normal_message_task_complete(self, agent: SelvageEvaluationAgent):
        """정상적인 메시지 처리 - 작업 완료 시나리오"""
        # Given
        user_message = "현재 상태를 알려주세요"
        security_analysis = {"is_safe": True}
        final_response = "현재 시스템 상태는 정상입니다."
        
        with patch.object(agent, '_analyze_security_intent', return_value=security_analysis):
            with patch.object(agent, 'plan_execution_loop', return_value=final_response) as mock_loop:
                # When
                result = agent.handle_user_message(user_message)
                
                # Then
                assert result == final_response
                assert agent.is_interactive_mode is True
                mock_loop.assert_called_once_with(user_message)

    def test_handle_normal_message_need_user_help(self, agent: SelvageEvaluationAgent):
        """사용자 도움이 필요한 시나리오"""
        # Given
        user_message = "복잡한 작업을 처리해주세요"
        security_analysis = {"is_safe": True}
        help_response = "추가 정보가 필요합니다. 어떤 종류의 작업인지 구체적으로 설명해주세요."
        
        with patch.object(agent, '_analyze_security_intent', return_value=security_analysis):
            with patch.object(agent, 'plan_execution_loop', return_value=help_response) as mock_loop:
                # When
                result = agent.handle_user_message(user_message)
                
                # Then
                assert result == help_response
                mock_loop.assert_called_once_with(user_message)

    def test_handle_message_with_exception(self, agent: SelvageEvaluationAgent):
        """메시지 처리 중 예외 발생 테스트"""
        # Given
        user_message = "테스트 메시지"
        security_analysis = {"is_safe": True}
        
        with patch.object(agent, '_analyze_security_intent', return_value=security_analysis):
            with patch.object(agent, 'plan_execution_loop', side_effect=Exception("API 호출 오류")):
                # When
                result = agent.handle_user_message(user_message)
                
                # Then
                assert "메시지 처리 중 오류가 발생했습니다" in result
                assert "API 호출 오류" in result
                
                # 오류 상황도 히스토리에 기록되는지 확인
                agent.session_state.add_conversation_turn.assert_called_once_with(
                    user_message=user_message,
                    assistant_response=result
                )

    def test_interactive_mode_flag_setting(self, agent: SelvageEvaluationAgent):
        """대화형 모드 플래그 설정 테스트"""
        # Given
        user_message = "테스트 메시지"
        security_analysis = {"is_safe": True}
        
        # 초기 상태 확인
        assert agent.is_interactive_mode is False
        
        with patch.object(agent, '_analyze_security_intent', return_value=security_analysis):
            with patch.object(agent, 'plan_execution_loop', return_value="응답"):
                # When
                agent.handle_user_message(user_message)
                
                # Then
                assert agent.is_interactive_mode is True

    def test_non_command_message_starting_with_slash(self, agent: SelvageEvaluationAgent):
        """/ 로 시작하지만 명령어가 아닌 메시지 처리"""
        # Given
        user_message = "/some/path/to/file을 읽어주세요"
        security_analysis = {"is_safe": True}
        response = "파일을 읽었습니다."
        
        with patch.object(agent, '_analyze_security_intent', return_value=security_analysis):
            with patch.object(agent, 'plan_execution_loop', return_value=response) as mock_loop:
                # When
                result = agent.handle_user_message(user_message)
                
                # Then
                assert result == response
                # 특수 명령어로 처리되지 않고 plan_execution_loop가 호출되어야 함
                mock_loop.assert_called_once_with(user_message)

    def test_empty_message_handling(self, agent: SelvageEvaluationAgent):
        """빈 메시지 처리 테스트"""
        # Given
        user_message = ""
        security_analysis = {"is_safe": True}
        response = "메시지가 비어있습니다."
        
        with patch.object(agent, '_analyze_security_intent', return_value=security_analysis):
            with patch.object(agent, 'plan_execution_loop', return_value=response):
                # When
                result = agent.handle_user_message(user_message)
                
                # Then
                assert result == response

    def test_whitespace_only_message_handling(self, agent: SelvageEvaluationAgent):
        """공백만 있는 메시지 처리 테스트"""
        # Given
        user_message = "   \n\t  "
        security_analysis = {"is_safe": True}
        response = "유효한 메시지를 입력해주세요."
        
        with patch.object(agent, '_analyze_security_intent', return_value=security_analysis):
            with patch.object(agent, 'plan_execution_loop', return_value=response):
                # When
                result = agent.handle_user_message(user_message)
                
                # Then
                assert result == response

    def test_security_analysis_called_for_normal_messages(self, agent: SelvageEvaluationAgent):
        """일반 메시지에 대해 보안 분석이 호출되는지 확인"""
        # Given
        user_message = "일반적인 질문입니다"
        security_analysis = {"is_safe": True}
        
        with patch.object(agent, '_analyze_security_intent', return_value=security_analysis) as mock_security:
            with patch.object(agent, 'plan_execution_loop', return_value="응답"):
                # When
                agent.handle_user_message(user_message)
                
                # Then
                mock_security.assert_called_once_with(user_message)

    def test_security_analysis_not_called_for_special_commands(self, agent: SelvageEvaluationAgent):
        """특수 명령어에 대해서는 보안 분석이 호출되지 않는지 확인"""
        # Given
        user_message = "/clear"
        
        with patch.object(agent, '_handle_special_command', return_value="초기화됨"):
            with patch.object(agent, '_analyze_security_intent') as mock_security:
                # When
                agent.handle_user_message(user_message)
                
                # Then
                mock_security.assert_not_called()