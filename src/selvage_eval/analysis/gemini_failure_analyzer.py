"""Gemini 기반 실패 패턴 분석기

Gemini API를 사용하여 실패 사유를 자동으로 분류하고 분석합니다.
"""

import json
import os
from typing import Tuple, Optional
import logging
from pydantic import BaseModel, Field

from selvage_eval.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class GeminiFailureAnalyzer:
    """Gemini 기반 실패 패턴 분석기"""
    
    def __init__(self):
        self.gemini_client = self._initialize_gemini_client()
        self.cache = {}  # 비용 효율성을 위한 캐시
        
        # Gemini 클라이언트가 초기화되지 않으면 예외 발생
        if not self.gemini_client:
            raise RuntimeError(
                "Gemini 클라이언트 초기화에 실패했습니다. "
                "GEMINI_API_KEY 환경 변수를 확인하고 API 키가 유효한지 확인해주세요."
            )
        
    def _initialize_gemini_client(self) -> GeminiClient:
        """Gemini 클라이언트 초기화"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                logger.error("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
                raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
                
            client = GeminiClient(api_key=api_key, model_name="gemini-2.5-flash")
            logger.info("Gemini 클라이언트 초기화 성공")
            return client
            
        except Exception as e:
            logger.error(f"Gemini 클라이언트 초기화 실패: {e}")
            raise RuntimeError(f"Gemini 클라이언트 초기화 실패: {e}")
    
    def categorize_failure(self, reason: str, failed_metric: Optional[str] = None) -> Tuple[str, float]:
        """실패 사유를 자유 형식 카테고리로 분류 (신뢰도 점수 포함)
        
        Args:
            reason: 실패 사유 텍스트
            failed_metric: 실패한 메트릭 이름 (correctness, clarity, actionability, json_correctness)
            
        Returns:
            Tuple[str, float]: (카테고리명, 신뢰도 점수)
        """
        logger.debug(f"실패 사유 분류 시작 - 메트릭: {failed_metric or 'unknown'}, 사유: {reason[:100]}...")
        
        # 캐시 키에 failed_metric 포함
        cache_key = hash(f"{reason}:{failed_metric or 'unknown'}")
        if cache_key in self.cache:
            logger.debug("캐시에서 분류 결과 반환")
            return self.cache[cache_key]
        
        # Gemini 분류 실행 (필수)
        try:
            logger.info(f"Gemini 실패 분류 시작 - 메트릭: {failed_metric or 'unknown'}")
            import time
            start_time = time.time()
            
            category, confidence = self._gemini_categorize_failure(reason, failed_metric)
            
            classification_time = time.time() - start_time
            logger.info(f"Gemini 실패 분류 완료 - 카테고리: '{category}', 신뢰도: {confidence:.3f} (소요시간: {classification_time:.2f}초)")
            
            self.cache[cache_key] = (category, confidence)
            logger.debug(f"분류 결과 캐시에 저장 - 캐시 크기: {len(self.cache)}")
            
            return category, confidence
        except Exception as e:
            error_msg = f"Gemini 분류 실패 - 메트릭: {failed_metric or 'unknown'}, 실패 사유: '{reason[:100]}{'...' if len(reason) > 100 else ''}', 오류: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _gemini_categorize_failure(self, reason: str, failed_metric: Optional[str] = None) -> Tuple[str, float]:
        """Gemini를 사용한 실패 사유 분류"""
        # 메트릭별 전문 컨텍스트 설명
        metric_contexts = {
            'correctness': {
                'description': '리뷰의 정확성 - 코드 이슈를 올바르게 식별했는가?',
                'common_failures': '이슈 미탐지, 잘못된 진단, 중요도 오판, 거짓 양성 오탐'
            },
            'clarity': {
                'description': '리뷰의 명확성 - 개발자가 이해하기 쉬운 설명인가?',
                'common_failures': '모호한 설명, 전문용어 남용, 불명확한 표현, 이해하기 어려운 구조'
            },
            'actionability': {
                'description': '실행가능성 - 구체적이고 실행 가능한 개선 방안을 제시했는가?',
                'common_failures': '추상적 제안, 비현실적 조치, 구체성 부족, 실행 방법 미제시'
            },
            'json_correctness': {
                'description': 'JSON 형식 유효성 - 올바른 JSON 스키마를 따르는가?',
                'common_failures': 'JSON 구문 오류, 스키마 불일치, 필드 누락, 데이터 타입 오류'
            }
        }
        
        # 메트릭 컨텍스트 추가
        metric_context = ""
        if failed_metric and failed_metric in metric_contexts:
            context = metric_contexts[failed_metric]
            metric_context = f"""
## 실패한 메트릭: {failed_metric.upper()}
**메트릭 설명:** {context['description']}
**일반적 실패 패턴:** {context['common_failures']}

이 메트릭에서 실패했다는 점을 고려하여 분석해주세요.
"""
        elif failed_metric:
            metric_context = f"""
## 실패한 메트릭: {failed_metric.upper()}
이 메트릭에서 실패했다는 점을 고려하여 분석해주세요.
"""
        
        prompt = f"""
## 컨텍스트
당신은 LLM 기반 코드 리뷰 도구 'Selvage'의 성능을 평가하는 전문가입니다.

### 평가 시스템 구조:
1. **Selvage**: AI 기반 코드 리뷰 도구 (피평가 시스템)
   - 코드 변경사항을 분석하여 리뷰 피드백 생성
   - JSON 형식으로 구조화된 리뷰 결과 출력

2. **DeepEval**: LLM 평가 프레임워크 (평가 시스템)
   - Selvage가 생성한 리뷰의 품질을 4가지 메트릭으로 평가
   - 각 메트릭은 0.7 이상 점수 시 통과, 미달 시 실패

### DeepEval 평가 메트릭:
- **Correctness**: 리뷰의 정확성 (코드 이슈를 올바르게 식별했는가?)
- **Clarity**: 리뷰의 명확성 (개발자가 이해하기 쉬운 설명인가?)
- **Actionability**: 실행가능성 (구체적이고 실행 가능한 개선 방안을 제시했는가?)
- **JSON Correctness**: JSON 형식 유효성 (올바른 JSON 스키마를 따르는가?)
{metric_context}
## 분석 대상
다음은 DeepEval이 Selvage의 리뷰 결과를 평가한 후, 특정 메트릭에서 **실패(threshold 0.7 미달)**한 사례의 실패 사유입니다.

**실패 사유:**
{reason}

## 분석 요청
위 실패 사유를 분석하여 Selvage의 코드 리뷰 품질 문제를 가장 잘 설명하는 카테고리명을 생성해주세요.

### 분류 관점:
- **코드 분석 능력**: 코드 이슈 탐지, 라인 번호 정확성, 중요도 판단 등
- **의사소통 능력**: 설명의 명확성, 개발자 친화적 표현, 전문용어 사용 등  
- **실용성**: 구체적 해결방안 제시, 실행 가능한 제안, 우선순위 가이드 등
- **기술적 정확성**: JSON 형식 준수, 스키마 일치성, 구조적 완성도 등
- **도메인 특화**: 보안, 성능, 가독성, 유지보수성 등 특정 영역 전문성

### 지침:
- 실패의 근본 원인을 반영하는 구체적인 카테고리명 생성
- 영어 스네이크케이스 사용 (예: "missing_security_vulnerabilities", "vague_improvement_suggestions")
- Selvage 개선에 직접적으로 도움이 되는 실행 가능한 카테고리명 선호
- 유사한 실패 패턴들을 그룹핑할 수 있는 일반적 수준의 추상화

응답 형식:
JSON 형식으로 다음 스키마를 따라 응답해주세요:
```json
{{
  "category": "카테고리명 (영어 스네이크케이스)",
  "confidence": 0.95,
  "explanation": "카테고리 선택 이유 및 Selvage 개선 방향"
}}
```
"""
        
        # JSON 스키마 정의
        class FailureAnalysisResponse(BaseModel):
            category: str = Field(description="실패 사유를 분류한 카테고리명 (영어 스네이크케이스)")
            confidence: float = Field(description="분류 신뢰도 (0.0-1.0)", ge=0.0, le=1.0)
            explanation: str = Field(description="카테고리 선택 이유 및 Selvage 개선 방향")
        
        messages = [{"role": "user", "content": prompt}]
        
        logger.debug(f"Gemini API 호출 시작 - 프롬프트 길이: {len(prompt)} 문자")
        import time
        api_start_time = time.time()
        
        response = self.gemini_client.query(
            messages=messages,
            system_instruction="실패 사유를 분석하여 JSON 형식으로 반환해주세요.",
            response_schema=FailureAnalysisResponse
        )
        
        api_time = time.time() - api_start_time
        logger.debug(f"Gemini API 호출 완료 (소요시간: {api_time:.2f}초)")
        
        return self._parse_gemini_response(response)
    
    def _parse_gemini_response(self, response_text: str) -> Tuple[str, float]:
        """Gemini structured output 응답 파싱"""
        try:
            # JSON 파싱
            response_data = json.loads(response_text.strip())
            
            category = response_data.get('category', 'unknown_failure')
            confidence = float(response_data.get('confidence', 0.8))
            explanation = response_data.get('explanation', '')
            
            # 카테고리명 정규화 (공백을 언더스코어로 변환, 소문자 변환)
            category = category.lower().replace(' ', '_').replace('-', '_')
            
            # 신뢰도 범위 검증
            confidence = max(0.0, min(1.0, confidence))
            
            logger.info(f"Gemini 분류 결과: {category} (신뢰도: {confidence:.3f}) - {explanation[:100]}")
            
            return category, confidence
            
        except json.JSONDecodeError as e:
            logger.warning(f"Gemini JSON 파싱 실패: {e}")
            logger.warning(f"Raw response: {response_text[:200]}...")
            
            # fallback: 기본 문자열 파싱 시도
            return self._fallback_parse_response(response_text)
        
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Gemini 응답 파싱 오류: {e}")
            return self._fallback_parse_response(response_text)
    
    def _fallback_parse_response(self, response_text: str) -> Tuple[str, float]:
        """JSON 파싱 실패 시 fallback 파싱"""
        # 기본 문자열 파싱 시도
        lines = response_text.strip().split('\n')
        category_str = "unknown_failure"
        confidence = 0.8
        
        for line in lines:
            line = line.strip()
            if '카테고리' in line and ':' in line:
                category_str = line.split(':')[1].strip()
            elif '신뢰도' in line and ':' in line:
                try:
                    confidence = float(line.split(':')[1].strip())
                except ValueError:
                    pass
        
        # 카테고리명 정규화
        category_str = category_str.lower().replace(' ', '_').replace('-', '_')
        logger.info(f"Fallback 파싱 결과: {category_str} (신뢰도: {confidence})")
        
        return category_str, confidence
    
    def get_cache_stats(self) -> dict:
        """캐시 통계 반환"""
        return {
            'cache_size': len(self.cache),
            'cache_entries': list(self.cache.keys())[:10]  # 첫 10개만 표시
        }
    
    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()
        logger.info("Gemini 분석 캐시가 초기화되었습니다.")