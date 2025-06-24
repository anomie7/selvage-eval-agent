"""Gemini API 클라이언트

Google Gemini API에 대한 단순한 래퍼 클라이언트입니다.
"""

import logging
from typing import Dict, List, Optional

from google.genai.types import GenerateContentConfig

try:
    from google import genai
    from google.genai.errors import ClientError, ServerError
except ImportError:
    # 테스트 환경에서는 모킹을 사용하므로 ImportError 무시
    genai = None
    ClientError = Exception
    ServerError = Exception

logger = logging.getLogger(__name__)


class GeminiClient:
    """Gemini API 클라이언트"""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro"):
        """클라이언트 초기화
        
        Args:
            api_key: Gemini API 키
            model_name: 사용할 모델 이름
        """
        self.api_key = api_key
        self.model_name = model_name
        
        if genai is not None:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
        
        logger.info(f"Initialized GeminiClient with model: {model_name}")

    def query(
        self, 
        messages: List[Dict[str, str]], 
        system_instruction: str,    
    ) -> str:
        """Gemini API에 쿼리를 보내고 응답을 받습니다
        
        Args:
            messages: 메시지 리스트 (role, content 포함)
            response_format: 응답 형식 (예: "json")
            max_tokens: 최대 토큰 수
            
        Returns:
            str: API 응답 텍스트
            
        Raises:
            RuntimeError: 클라이언트가 초기화되지 않았거나 응답이 비어있는 경우
            ClientError, ServerError: API 오류
        """
        try:
            if self.client is None:
                raise RuntimeError("Gemini client not initialized")
            
            # 메시지를 단일 프롬프트로 변환
            if len(messages) == 1:
                contents = messages[0]["content"]
            else:
                # 여러 메시지가 있으면 결합
                content_parts = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        content_parts.append(f"[SYSTEM] {content}")
                    elif role == "user":
                        content_parts.append(f"[USER] {content}")
                    else:
                        content_parts.append(content)
                contents = "\n\n".join(content_parts)
            
            logger.debug(f"Sending query to Gemini: {contents[:100]}...")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0
                )
            )
            
            if response.text is None:
                raise RuntimeError("Empty response from Gemini API")
            
            logger.debug("Received response from Gemini")
            return response.text
            
        except (ClientError, ServerError) as e:
            logger.error(f"Gemini API error: {e}")
            raise e