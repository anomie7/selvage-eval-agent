"""OpenRouter API 클라이언트

OpenRouter를 통해 다양한 모델을 호출할 수 있는 클라이언트입니다.
"""

import concurrent.futures
import logging
from typing import Dict, List, Optional, Any, Type, cast, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

from pydantic import BaseModel

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    # 테스트 환경에서는 모킹을 사용하므로 ImportError 무시
    OpenAI = None
    OPENAI_AVAILABLE = False

from selvage_eval.tools.tool import Tool

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """OpenRouter API 클라이언트"""

    def __init__(self, api_key: str):
        """클라이언트 초기화
        
        Args:
            api_key: OpenRouter API 키
        """
        self.api_key = api_key
        
        if OpenAI is not None:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            self.client = None
        
        logger.info("Initialized OpenRouterClient")

    def query(
        self, 
        messages: List[Dict[str, str]], 
        system_instruction: str,
        model_name: str,
        tools: Optional[List[Tool]] = None,
        response_schema: Optional[Type[BaseModel]] = None
    ) -> Any:
        """OpenRouter API에 쿼리를 보내고 응답을 받습니다
        
        Args:
            messages: 메시지 리스트 (role, content 포함)
            system_instruction: 시스템 인스트럭션
            model_name: 사용할 모델 이름 (예: "gemini-2.5-flash", "gemini-2.5-pro")
            tools: 사용할 도구 리스트 (Tool 객체들) - 현재 미지원
            response_schema: 구조화된 출력을 위한 Pydantic BaseModel 클래스 - 현재 미지원
            
        Returns:
            Any: API 응답 텍스트
            
        Raises:
            RuntimeError: 클라이언트가 초기화되지 않았거나 응답이 비어있는 경우
            Exception: API 오류
        """
        try:
            if self.client is None:
                raise RuntimeError("OpenRouter client not initialized")
            
            # 모델명을 OpenRouter 형식으로 변환
            openrouter_model = self._convert_model_name(model_name)
            
            # 메시지 형식을 OpenAI 형식으로 변환
            openai_messages = self._convert_messages_to_openai_format(messages, system_instruction)
            
            logger.debug(f"Sending query to OpenRouter with model: {openrouter_model}")
            
            # tools와 response_schema는 현재 미지원이므로 경고 로그 출력
            if tools:
                logger.warning("Tools are not yet supported in OpenRouterClient")
            if response_schema:
                logger.warning("Response schema is not yet supported in OpenRouterClient")
            
            response = self.client.chat.completions.create(
                model=openrouter_model,
                messages=openai_messages,
                temperature=0.0
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise RuntimeError("Empty response from OpenRouter API")
            
            result = response.choices[0].message.content
            logger.debug("Received text response from OpenRouter")
            return result
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise e
    
    def _convert_model_name(self, model_name: str) -> str:
        """모델명을 OpenRouter 형식으로 변환
        
        Args:
            model_name: 원본 모델명 (예: "gemini-2.5-flash")
            
        Returns:
            OpenRouter 형식의 모델명 (예: "google/gemini-2.5-flash")
        """
        # Gemini 모델들을 OpenRouter 형식으로 변환
        if model_name.startswith("gemini-"):
            return f"google/{model_name}"
        
        # 이미 provider/model 형식이면 그대로 반환
        if "/" in model_name:
            return model_name
        
        # 다른 모델들의 경우 기본값으로 그대로 반환
        logger.warning(f"Unknown model format: {model_name}, using as-is")
        return model_name
    
    def _convert_messages_to_openai_format(self, messages: List[Dict[str, str]], system_instruction: str) -> List["ChatCompletionMessageParam"]:
        """메시지를 OpenAI 형식으로 변환
        
        Args:
            messages: 원본 메시지 리스트
            system_instruction: 시스템 인스트럭션
            
        Returns:
            OpenAI 형식의 메시지 리스트
        """
        openai_messages: List["ChatCompletionMessageParam"] = []
        
        # 시스템 인스트럭션 추가
        if system_instruction:
            openai_messages.append(cast("ChatCompletionMessageParam", {
                "role": "system",
                "content": system_instruction
            }))
        
        # 기존 메시지들 변환
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # role 매핑
            if role == "user":
                openai_messages.append(cast("ChatCompletionMessageParam", {
                    "role": "user",
                    "content": content
                }))
            elif role == "assistant":
                openai_messages.append(cast("ChatCompletionMessageParam", {
                    "role": "assistant", 
                    "content": content
                }))
            elif role == "system":
                # 시스템 메시지는 이미 추가했으므로 사용자 메시지로 변환
                openai_messages.append(cast("ChatCompletionMessageParam", {
                    "role": "user",
                    "content": f"[SYSTEM] {content}"
                }))
            else:
                # 알 수 없는 role은 user로 처리
                openai_messages.append(cast("ChatCompletionMessageParam", {
                    "role": "user",
                    "content": content
                }))
        
        return openai_messages

    def batch_query(
        self,
        batch_requests: List[Dict[str, Any]],
        system_instruction: str,
        model_name: str,
        max_workers: int = 5,
        response_schema: Optional[Type[BaseModel]] = None
    ) -> List[Any]:
        """여러 요청을 병렬로 처리하여 OpenRouter API 성능 최적화
        
        Args:
            batch_requests: 배치 요청 리스트, 각 요청은 {'messages': [...]} 형식
            system_instruction: 시스템 인스트럭션
            model_name: 사용할 모델 이름
            max_workers: 최대 동시 처리 스레드 수 (기본값: 5)
            response_schema: 구조화된 출력을 위한 Pydantic BaseModel 클래스 - 현재 미지원
            
        Returns:
            List[Any]: 각 요청에 대한 응답 리스트
            
        Raises:
            RuntimeError: 클라이언트가 초기화되지 않은 경우
        """
        if self.client is None:
            raise RuntimeError("OpenRouter client not initialized")
        
        if not batch_requests:
            return []
        
        logger.info(f"병렬 처리 시작: {len(batch_requests)}개 요청, 최대 {max_workers}개 워커, 모델: {model_name}")
        
        def process_single_request(request_data: Dict[str, Any]) -> Any:
            """단일 요청 처리"""
            try:
                messages = request_data.get('messages', [])
                return self.query(
                    messages=messages,
                    system_instruction=system_instruction,
                    model_name=model_name,
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