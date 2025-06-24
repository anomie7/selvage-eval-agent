"""Gemini API 클라이언트 단위 테스트

새로운 아키텍처에서 GeminiClient는 단순한 API 래퍼 역할만 합니다.
비즈니스 로직은 core_agent.py에서 처리됩니다.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from selvage_eval.llm.gemini_client import GeminiClient


class TestGeminiClient:
    """GeminiClient 테스트 클래스"""

    @pytest.fixture
    def api_key(self):
        """테스트용 API 키"""
        return "test-api-key-12345"

    @pytest.fixture
    def mock_genai(self):
        """Google GenAI SDK 모킹"""
        with patch('selvage_eval.llm.gemini_client.genai') as mock_genai:
            mock_client = Mock()
            mock_models = Mock()
            mock_client.models = mock_models
            mock_genai.Client.return_value = mock_client
            yield mock_genai, mock_client, mock_models

    @pytest.fixture
    def gemini_client(self, api_key, mock_genai):
        """테스트용 GeminiClient 인스턴스"""
        return GeminiClient(api_key)

    def test_client_initialization_success(self, api_key, mock_genai):
        """클라이언트 초기화 성공 테스트"""
        # Given & When
        client = GeminiClient(api_key)
        
        # Then
        assert client.api_key == api_key
        assert client.model_name == "gemini-2.5-pro"
        assert client.client is not None

    def test_client_initialization_with_custom_model(self, api_key, mock_genai):
        """커스텀 모델로 클라이언트 초기화 테스트"""
        # Given
        custom_model = "gemini-2.5-pro"
        
        # When
        client = GeminiClient(api_key, model_name=custom_model)
        
        # Then
        assert client.model_name == custom_model

    def test_query_single_message_success(self, gemini_client: GeminiClient, mock_genai):
        """단일 메시지 쿼리 성공 테스트"""
        # Given
        _, mock_client, mock_models = mock_genai
        messages = [{"role": "user", "content": "안녕하세요"}]
        expected_response = "안녕하세요! 무엇을 도와드릴까요?"
        
        mock_response = Mock()
        mock_response.text = expected_response
        mock_models.generate_content.return_value = mock_response
        
        # When
        result = gemini_client.query(messages, system_instruction="")
        
        # Then
        assert result == expected_response
        mock_models.generate_content.assert_called_once()

    def test_query_multiple_messages_success(self, gemini_client: GeminiClient, mock_genai):
        """다중 메시지 쿼리 성공 테스트"""
        # Given
        _, mock_client, mock_models = mock_genai
        messages = [
            {"role": "user", "content": "파일을 읽어주세요"},
            {"role": "user", "content": "그리고 요약도 해주세요"}
        ]
        expected_response = "어떤 파일을 읽어드릴까요?"
        
        mock_response = Mock()
        mock_response.text = expected_response
        mock_models.generate_content.return_value = mock_response
        
        # When
        result = gemini_client.query(messages, system_instruction="당신은 도움이 되는 어시스턴트입니다.")
        
        # Then
        assert result == expected_response
        # 여러 메시지들이 적절히 결합되어 전달되었는지 확인
        call_args = mock_models.generate_content.call_args
        contents_arg = call_args[1]['contents']  # kwargs에서 contents 추출
        assert "[USER]" in contents_arg

    def test_query_with_response_format_and_max_tokens(self, gemini_client: GeminiClient, mock_genai):
        """응답 형식 및 최대 토큰 수 지정 테스트"""
        # Given
        _, mock_client, mock_models = mock_genai
        messages = [{"role": "user", "content": "JSON으로 응답해주세요"}]
        expected_response = '{"status": "success"}'
        
        mock_response = Mock()
        mock_response.text = expected_response
        mock_models.generate_content.return_value = mock_response
        
        # When
        result = gemini_client.query(
            messages=messages,
            system_instruction="당신은 도움이 되는 어시스턴트입니다."
        )
        
        # Then
        assert result == expected_response

    def test_query_client_not_initialized(self, api_key):
        """클라이언트가 초기화되지 않은 경우 테스트"""
        # Given
        with patch('selvage_eval.llm.gemini_client.genai', None):
            client = GeminiClient(api_key)
            messages = [{"role": "user", "content": "테스트"}]
        
        # When & Then
        with pytest.raises(RuntimeError, match="Gemini client not initialized"):
            client.query(messages, system_instruction="")

    def test_query_empty_response(self, gemini_client, mock_genai):
        """빈 응답 처리 테스트"""
        # Given
        _, mock_client, mock_models = mock_genai
        messages = [{"role": "user", "content": "테스트"}]
        
        mock_response = Mock()
        mock_response.text = None
        mock_models.generate_content.return_value = mock_response
        
        # When & Then
        with pytest.raises(RuntimeError, match="Empty response from Gemini API"):
            gemini_client.query(messages, system_instruction="")

    def test_query_api_client_error(self, gemini_client: GeminiClient, mock_genai):
        """Gemini API 클라이언트 오류 테스트"""
        # Given
        _, mock_client, mock_models = mock_genai
        messages = [{"role": "user", "content": "테스트"}]
        
        from selvage_eval.llm.gemini_client import ClientError
        mock_models.generate_content.side_effect = ClientError(429, {"error": {"message": "Rate limit exceeded"}}, None)
        
        # When & Then
        with pytest.raises(ClientError):
            gemini_client.query(messages, system_instruction="")

    def test_query_api_server_error(self, gemini_client: GeminiClient, mock_genai):
        """Gemini API 서버 오류 테스트"""
        # Given
        _, mock_client, mock_models = mock_genai
        messages = [{"role": "user", "content": "테스트"}]
        
        from selvage_eval.llm.gemini_client import ServerError
        mock_models.generate_content.side_effect = ServerError(500, {"error": {"message": "Internal server error"}}, None)
        
        # When & Then
        with pytest.raises(ServerError):
            gemini_client.query(messages, system_instruction="")

    def test_query_unexpected_error(self, gemini_client: GeminiClient, mock_genai):
        """예상치 못한 오류 테스트"""
        # Given
        _, mock_client, mock_models = mock_genai
        messages = [{"role": "user", "content": "테스트"}]
        
        mock_models.generate_content.side_effect = Exception("Unexpected error")
        
        # When & Then
        with pytest.raises(Exception, match="Unexpected error"):
            gemini_client.query(messages, system_instruction="")

    def test_query_empty_messages_list(self, gemini_client: GeminiClient, mock_genai):
        """빈 메시지 리스트 처리 테스트"""
        # Given
        _, mock_client, mock_models = mock_genai
        messages = []
        expected_response = "빈 메시지에 대한 응답"
        
        mock_response = Mock()
        mock_response.text = expected_response
        mock_models.generate_content.return_value = mock_response
        
        # When
        result = gemini_client.query(messages, system_instruction="")
        
        # Then
        # 빈 메시지 리스트는 빈 콘텐츠로 처리됨
        assert result == expected_response
        # 빈 문자열이 전달되었는지 확인
        call_args = mock_models.generate_content.call_args
        assert call_args[1]['contents'] == ""