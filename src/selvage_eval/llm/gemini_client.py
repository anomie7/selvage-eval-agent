"""Gemini API 클라이언트

Google Gemini API에 대한 단순한 래퍼 클라이언트입니다.
"""

import concurrent.futures
import logging
from typing import Dict, List, Optional, Any, Type

from pydantic import BaseModel


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
        tools: Optional[List[Tool]] = None,
        response_schema: Optional[Type[BaseModel]] = None
    ) -> Any:
        """Gemini API에 쿼리를 보내고 응답을 받습니다
        
        Args:
            messages: 메시지 리스트 (role, content 포함)
            system_instruction: 시스템 인스트럭션
            tools: 사용할 도구 리스트 (Tool 객체들)
            response_schema: 구조화된 출력을 위한 Pydantic BaseModel 클래스
            
        Returns:
            Any: API 응답 (텍스트, function call, 또는 구조화된 JSON 응답)
            
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
            
            # response_schema가 제공된 경우 구조화된 출력 설정
            if response_schema:
                if BaseModel and issubclass(response_schema, BaseModel):
                    # Pydantic 모델에서 JSON 스키마 추출
                    json_schema = response_schema.model_json_schema()
                    
                    # Gemini API는 additionalProperties를 지원하지 않으므로 제거
                    def remove_additional_properties(schema_dict):
                        if isinstance(schema_dict, dict):
                            schema_dict.pop('additionalProperties', None)
                            for value in schema_dict.values():
                                if isinstance(value, dict):
                                    remove_additional_properties(value)
                                elif isinstance(value, list):
                                    for item in value:
                                        if isinstance(item, dict):
                                            remove_additional_properties(item)
                    
                    remove_additional_properties(json_schema)
                    
                    config.response_mime_type = "application/json"
                    config.response_schema = json_schema
                    logger.debug("Configured structured output with Pydantic model schema")
                else:
                    raise ValueError("response_schema must be a Pydantic BaseModel subclass")
            
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
            # response_schema가 제공된 경우 구조화된 응답 반환
            elif response_schema:
                if response.text is None:
                    raise RuntimeError("Empty response from Gemini API")
                logger.debug("Received structured JSON response from Gemini")
                return response.text  # JSON 문자열 반환
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

    def batch_query(
        self,
        batch_requests: List[Dict[str, Any]],
        system_instruction: str,
        max_workers: int = 5,
        response_schema: Optional[Type[BaseModel]] = None
    ) -> List[Any]:
        """여러 요청을 병렬로 처리하여 Gemini API 성능 최적화
        
        Args:
            batch_requests: 배치 요청 리스트, 각 요청은 {'messages': [...]} 형식
            system_instruction: 시스템 인스트럭션
            max_workers: 최대 동시 처리 스레드 수 (기본값: 5)
            response_schema: 구조화된 출력을 위한 Pydantic BaseModel 클래스
            
        Returns:
            List[Any]: 각 요청에 대한 응답 리스트
            
        Raises:
            RuntimeError: 클라이언트가 초기화되지 않은 경우
        """
        if self.client is None:
            raise RuntimeError("Gemini client not initialized")
        
        if not batch_requests:
            return []
        
        logger.info(f"병렬 처리 시작: {len(batch_requests)}개 요청, 최대 {max_workers}개 워커")
        
        def process_single_request(request_data: Dict[str, Any]) -> Any:
            """단일 요청 처리"""
            try:
                messages = request_data.get('messages', [])
                return self.query(
                    messages=messages,
                    system_instruction=system_instruction,
                    response_schema=response_schema
                )
            except Exception as e:
                logger.error(f"배치 요청 처리 중 오류: {e}")
                return None
        
        # ThreadPoolExecutor를 사용한 병렬 처리
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 모든 요청을 제출
            future_to_index = {
                executor.submit(process_single_request, request): i 
                for i, request in enumerate(batch_requests)
            }
            
            # 결과를 원래 순서대로 수집
            results = [None] * len(batch_requests)
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"배치 요청 {index} 처리 실패: {e}")
                    results[index] = None
        
        logger.info(f"병렬 처리 완료: {len([r for r in results if r is not None])}/{len(batch_requests)} 성공")
        return results