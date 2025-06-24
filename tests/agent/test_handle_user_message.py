"""handle_user_message 메서드 단위 테스트

전체 플로우의 시작점인 handle_user_message 메서드를 테스트합니다.
TDD 방식으로 먼저 테스트를 작성하고, 실제 구현을 개선합니다.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from selvage_eval.agent.core_agent import SelvageEvaluationAgent
from selvage_eval.config.settings import EvaluationConfig
from selvage_eval.tools.execution_plan import ExecutionPlan
from selvage_eval.tools.tool_call import ToolCall
from selvage_eval.tools.tool_result import ToolResult


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
            mock_session.return_value = mock_session_instance
            
            with patch.object(SelvageEvaluationAgent, '_save_session_metadata'):
                agent = SelvageEvaluationAgent(mock_config)
                agent.session_state = mock_session_instance
                return agent

    def test_handle_normal_user_message_success(self, agent):
        """정상적인 사용자 메시지 처리 테스트"""
        # Given
        user_message = "현재 상태를 알려주세요"
        
        # ExecutionPlan 모킹
        mock_plan = ExecutionPlan(
            intent_summary="Query current status",
            confidence=0.9,
            parameters={},
            tool_calls=[
                ToolCall(
                    tool="read_file",
                    params={"file_path": "/test/session_metadata.json"},
                    rationale="Read session metadata"
                )
            ],
            safety_check="Read-only operation, safe",
            expected_outcome="Current session status information"
        )
        
        # 도구 실행 결과 모킹
        mock_tool_result = ToolResult(
            success=True,
            data={"content": {"session_id": "test-123", "status": "active"}},
            error_message=None
        )
        
        # 메서드 모킹
        with patch.object(agent, 'plan_execution', return_value=mock_plan) as mock_plan_exec:
            with patch.object(agent, '_validate_plan_safety', return_value=True) as mock_safety:
                with patch.object(agent, 'execute_tool', return_value=mock_tool_result) as mock_exec_tool:
                    with patch.object(agent, 'generate_response', return_value="상태 정보입니다") as mock_gen_response:
                        # When
                        result = agent.handle_user_message(user_message)
                        
                        # Then
                        assert result == "상태 정보입니다"
                        assert agent.is_interactive_mode is True
                        
                        # 메서드 호출 검증
                        mock_plan_exec.assert_called_once_with(user_message)
                        mock_safety.assert_called_once_with(mock_plan)
                        mock_exec_tool.assert_called_once_with("read_file", {"file_path": "/test/session_metadata.json"})
                        mock_gen_response.assert_called_once()
                        
                        # 세션 상태 업데이트 검증
                        agent.session_state.add_conversation_turn.assert_called_once()

    def test_handle_special_command_clear(self, agent):
        """특수 명령어 /clear 처리 테스트"""
        # Given
        user_message = "/clear"
        
        with patch.object(agent, '_handle_special_command', return_value="대화 히스토리가 초기화되었습니다.") as mock_special:
            # When
            result = agent.handle_user_message(user_message)
            
            # Then
            assert result == "대화 히스토리가 초기화되었습니다."
            mock_special.assert_called_once_with(user_message)

    def test_handle_special_command_context(self, agent):
        """특수 명령어 /context 처리 테스트"""
        # Given
        user_message = "/context"
        
        with patch.object(agent, '_handle_special_command', return_value="컨텍스트 정보입니다.") as mock_special:
            # When
            result = agent.handle_user_message(user_message)
            
            # Then
            assert result == "컨텍스트 정보입니다."
            mock_special.assert_called_once_with(user_message)

    def test_handle_unsafe_plan_rejection(self, agent):
        """안전하지 않은 계획 거부 테스트"""
        # Given
        user_message = "시스템 파일을 삭제해주세요"
        
        # 안전하지 않은 ExecutionPlan 모킹
        unsafe_plan = ExecutionPlan(
            intent_summary="Delete system files",
            confidence=0.8,
            parameters={},
            tool_calls=[
                ToolCall(
                    tool="delete_file",
                    params={"file_path": "/system/important.conf"},
                    rationale="Delete system file"
                )
            ],
            safety_check="Dangerous operation detected",
            expected_outcome="System file deletion"
        )
        
        with patch.object(agent, 'plan_execution', return_value=unsafe_plan):
            with patch.object(agent, '_validate_plan_safety', return_value=False):
                # When
                result = agent.handle_user_message(user_message)
                
                # Then
                assert "보안상 실행할 수 없습니다" in result
                assert "Dangerous operation detected" in result
                
                # 세션 상태에 오류 기록 확인
                agent.session_state.add_conversation_turn.assert_called_once()

    def test_handle_plan_execution_failure(self, agent):
        """계획 수립 실패 테스트"""
        # Given
        user_message = "테스트 메시지"
        
        with patch.object(agent, 'plan_execution', side_effect=Exception("LLM API 오류")):
            # When
            result = agent.handle_user_message(user_message)
            
            # Then
            assert "메시지 처리 중 오류가 발생했습니다" in result
            assert "LLM API 오류" in result
            
            # 오류 상황도 세션 상태에 기록되는지 확인
            agent.session_state.add_conversation_turn.assert_called_once()

    def test_handle_tool_execution_failure(self, agent):
        """도구 실행 실패 테스트"""
        # Given
        user_message = "파일을 읽어주세요"
        
        # 정상적인 ExecutionPlan
        mock_plan = ExecutionPlan(
            intent_summary="Read file",
            confidence=0.9,
            parameters={},
            tool_calls=[
                ToolCall(
                    tool="read_file",
                    params={"file_path": "/nonexistent/file.txt"},
                    rationale="Read file"
                )
            ],
            safety_check="Read-only operation, safe",
            expected_outcome="File content"
        )
        
        # 도구 실행 실패 결과
        failed_tool_result = ToolResult(
            success=False,
            data=None,
            error_message="파일을 찾을 수 없습니다"
        )
        
        with patch.object(agent, 'plan_execution', return_value=mock_plan):
            with patch.object(agent, '_validate_plan_safety', return_value=True):
                with patch.object(agent, 'execute_tool', return_value=failed_tool_result):
                    with patch.object(agent, 'generate_response', return_value="파일 읽기에 실패했습니다") as mock_gen_response:
                        # When
                        result = agent.handle_user_message(user_message)
                        
                        # Then
                        assert result == "파일 읽기에 실패했습니다"
                        
                        # generate_response에 실패한 도구 결과가 전달되는지 확인
                        mock_gen_response.assert_called_once()
                        call_args = mock_gen_response.call_args
                        tool_results = call_args[0][2]  # 세 번째 인자 (tool_results)
                        assert len(tool_results) == 1
                        assert tool_results[0]["result"].success is False

    def test_handle_multiple_tools_execution(self, agent):
        """여러 도구 실행 테스트"""
        # Given
        user_message = "상태를 확인하고 파일 목록도 보여주세요"
        
        # 여러 도구를 포함한 ExecutionPlan
        mock_plan = ExecutionPlan(
            intent_summary="Check status and list files",
            confidence=0.9,
            parameters={},
            tool_calls=[
                ToolCall(
                    tool="read_file",
                    params={"file_path": "/test/status.json"},
                    rationale="Read status"
                ),
                ToolCall(
                    tool="list_directory",
                    params={"directory_path": "/test"},
                    rationale="List files"
                )
            ],
            safety_check="Read-only operations, safe",
            expected_outcome="Status and file list"
        )
        
        # 각 도구의 실행 결과
        tool_results = [
            ToolResult(success=True, data={"content": {"status": "ok"}}, error_message=None),
            ToolResult(success=True, data={"files": ["file1.txt", "file2.txt"]}, error_message=None)
        ]
        
        with patch.object(agent, 'plan_execution', return_value=mock_plan):
            with patch.object(agent, '_validate_plan_safety', return_value=True):
                with patch.object(agent, 'execute_tool', side_effect=tool_results):
                    with patch.object(agent, 'generate_response', return_value="상태와 파일 목록입니다") as mock_gen_response:
                        # When
                        result = agent.handle_user_message(user_message)
                        
                        # Then
                        assert result == "상태와 파일 목록입니다"
                        
                        # 두 번의 도구 실행 확인
                        assert agent.execute_tool.call_count == 2
                        
                        # generate_response에 두 도구의 결과가 모두 전달되는지 확인
                        call_args = mock_gen_response.call_args
                        tool_results_arg = call_args[0][2]
                        assert len(tool_results_arg) == 2

    def test_conversation_history_recording(self, agent):
        """대화 히스토리 기록 테스트"""
        # Given
        user_message = "안녕하세요"
        
        mock_plan = ExecutionPlan(
            intent_summary="Greeting",
            confidence=0.5,
            parameters={},
            tool_calls=[],
            safety_check="Safe",
            expected_outcome="Greeting response"
        )
        
        with patch.object(agent, 'plan_execution', return_value=mock_plan):
            with patch.object(agent, '_validate_plan_safety', return_value=True):
                with patch.object(agent, 'generate_response', return_value="안녕하세요!"):
                    # When
                    result = agent.handle_user_message(user_message)
                    
                    # Then
                    assert result == "안녕하세요!"
                    
                    # 대화 히스토리 기록 확인
                    agent.session_state.add_conversation_turn.assert_called_once_with(
                        user_message="안녕하세요",
                        assistant_response="안녕하세요!",
                        tool_results=[]
                    )

    def test_interactive_mode_flag_setting(self, agent):
        """대화형 모드 플래그 설정 테스트"""
        # Given
        user_message = "테스트 메시지"
        
        # 초기 상태 확인
        assert agent.is_interactive_mode is False
        
        with patch.object(agent, 'plan_execution', return_value=Mock()):
            with patch.object(agent, '_validate_plan_safety', return_value=True):
                with patch.object(agent, 'generate_response', return_value="응답"):
                    # When
                    agent.handle_user_message(user_message)
                    
                    # Then
                    assert agent.is_interactive_mode is True