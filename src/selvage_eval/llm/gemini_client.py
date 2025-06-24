"""Gemini API 클라이언트

Google Gemini API에 대한 단순한 래퍼 클라이언트입니다.
"""

import logging
from typing import Dict, List, Optional, Any

from google.genai.types import GenerateContentConfig, FunctionDeclaration, Tool as GeminiTool

from selvage_eval.tools.tool import Tool

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
        tools: Optional[List[Tool]] = None    
    ) -> Any:
        """Gemini API에 쿼리를 보내고 응답을 받습니다
        
        Args:
            messages: 메시지 리스트 (role, content 포함)
            system_instruction: 시스템 인스트럭션
            tools: 사용할 도구 리스트 (Tool 객체들)
            
        Returns:
            Any: API 응답 (텍스트 또는 function call 포함 응답)
            
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
            
            # GenerateContentConfig 구성
            config = GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0
            )
            
            # tools가 제공된 경우 function calling 설정
            if tools:
                function_declarations = self._build_function_declarations(tools)
                gemini_tools = GeminiTool(function_declarations=function_declarations)
                config.tools = [gemini_tools]
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            # tools가 제공된 경우 전체 응답 반환 (function call 처리를 위해)
            if tools:
                logger.debug("Received function calling response from Gemini")
                return response
            else:
                # 기존 방식: 텍스트만 반환
                if response.text is None:
                    raise RuntimeError("Empty response from Gemini API")
                
                logger.debug("Received text response from Gemini")
                return response.text
            
        except (ClientError, ServerError) as e:
            logger.error(f"Gemini API error: {e}")
            raise e
    
    def _build_function_declarations(self, tools: List[Tool]) -> List[FunctionDeclaration]:
        """Tool 객체들을 Gemini function declarations로 변환
        
        Args:
            tools: Tool 객체들의 리스트
            
        Returns:
            List[FunctionDeclaration]: Gemini function declarations
        """
        function_declarations = []
        for tool in tools:
            if hasattr(tool, 'get_function_declaration'):
                declaration_dict = tool.get_function_declaration()
                # Dict를 FunctionDeclaration 객체로 변환
                function_declaration = FunctionDeclaration(
                    name=declaration_dict["name"],
                    description=declaration_dict["description"],
                    parameters=declaration_dict["parameters"]
                )
                function_declarations.append(function_declaration)
            else:
                logger.warning(f"Tool {tool} does not have get_function_declaration method")
        return function_declarations