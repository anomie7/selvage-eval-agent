"""Gemini 실패 분석기 테스트"""

import json
import unittest
from unittest.mock import patch, MagicMock, Mock
from selvage_eval.analysis.gemini_failure_analyzer import GeminiFailureAnalyzer


class TestGeminiFailureAnalyzer(unittest.TestCase):
    """Gemini 실패 분석기 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.sample_response = {
            "category": "missing_security_vulnerabilities",
            "confidence": 0.85,
            "explanation": "The review failed to identify critical security vulnerabilities in the code"
        }
    
    @patch('selvage_eval.analysis.gemini_failure_analyzer.os.getenv')
    def test_initialize_gemini_client_no_api_key(self, mock_getenv):
        """API 키 없이 초기화 실패 테스트"""
        mock_getenv.return_value = None
        
        with self.assertRaises(RuntimeError) as context:
            GeminiFailureAnalyzer()
        
        self.assertIn("Gemini 클라이언트 초기화에 실패했습니다", str(context.exception))
    
    def test_parse_gemini_response_success(self):
        """Gemini 응답 파싱 성공 테스트"""
        # 임시 분석기 생성 (초기화 없이)
        analyzer = GeminiFailureAnalyzer.__new__(GeminiFailureAnalyzer)
        analyzer.cache = {}
        
        # 응답 파싱 테스트
        response_text = json.dumps(self.sample_response)
        category, confidence = analyzer._parse_gemini_response(response_text)
        
        self.assertEqual(category, "missing_security_vulnerabilities")
        self.assertEqual(confidence, 0.85)
    
    def test_parse_gemini_response_invalid_json(self):
        """잘못된 JSON 응답 파싱 테스트"""
        # 임시 분석기 생성 (초기화 없이)
        analyzer = GeminiFailureAnalyzer.__new__(GeminiFailureAnalyzer)
        analyzer.cache = {}
        
        # 잘못된 JSON 응답 테스트
        invalid_response = "This is not valid JSON"
        category, confidence = analyzer._parse_gemini_response(invalid_response)
        
        # fallback 파싱 결과 검증
        self.assertEqual(category, "unknown_failure")
        self.assertEqual(confidence, 0.8)
    
    def test_fallback_parse_response(self):
        """fallback 응답 파싱 테스트"""
        # 임시 분석기 생성 (초기화 없이)
        analyzer = GeminiFailureAnalyzer.__new__(GeminiFailureAnalyzer)
        analyzer.cache = {}
        
        # fallback 파싱 테스트
        response_text = """
        카테고리: Vague Error Messages
        신뢰도: 0.9
        설명: 오류 메시지가 모호함
        """
        category, confidence = analyzer._fallback_parse_response(response_text)
        
        self.assertEqual(category, "vague_error_messages")
        self.assertEqual(confidence, 0.9)
    
    def test_category_normalization(self):
        """카테고리명 정규화 테스트"""
        # 임시 분석기 생성 (초기화 없이)
        analyzer = GeminiFailureAnalyzer.__new__(GeminiFailureAnalyzer)
        analyzer.cache = {}
        
        # 공백과 대문자가 포함된 응답
        response_with_spaces = {
            "category": "Missing Security Vulnerabilities",
            "confidence": 0.85,
            "explanation": "Test explanation"
        }
        
        category, confidence = analyzer._parse_gemini_response(json.dumps(response_with_spaces))
        
        # 정규화 검증 (소문자, 언더스코어 변환)
        self.assertEqual(category, "missing_security_vulnerabilities")
        self.assertEqual(confidence, 0.85)
    
    def test_confidence_range_validation(self):
        """신뢰도 범위 검증 테스트"""
        # 임시 분석기 생성 (초기화 없이)
        analyzer = GeminiFailureAnalyzer.__new__(GeminiFailureAnalyzer)
        analyzer.cache = {}
        
        # 범위를 벗어난 신뢰도 값
        response_with_invalid_confidence = {
            "category": "test_category",
            "confidence": 1.5,  # 1.0을 초과
            "explanation": "Test explanation"
        }
        
        category, confidence = analyzer._parse_gemini_response(json.dumps(response_with_invalid_confidence))
        
        # 신뢰도가 1.0으로 제한되어야 함
        self.assertEqual(confidence, 1.0)
    
    def test_cache_management(self):
        """캐시 관리 기능 테스트"""
        # 임시 분석기 생성 (초기화 없이)
        analyzer = GeminiFailureAnalyzer.__new__(GeminiFailureAnalyzer)
        analyzer.cache = {}
        
        # 캐시에 데이터 추가
        test_key = hash("test_reason:correctness")
        analyzer.cache[test_key] = ("test_category", 0.8)
        
        # 캐시 통계 확인
        stats = analyzer.get_cache_stats()
        self.assertEqual(stats['cache_size'], 1)
        self.assertIsInstance(stats['cache_entries'], list)
        
        # 캐시 초기화
        analyzer.clear_cache()
        stats_after_clear = analyzer.get_cache_stats()
        self.assertEqual(stats_after_clear['cache_size'], 0)
    
    def test_fallback_parse_response_with_colon(self):
        """콜론이 포함된 fallback 응답 파싱 테스트"""
        # 임시 분석기 생성 (초기화 없이)
        analyzer = GeminiFailureAnalyzer.__new__(GeminiFailureAnalyzer)
        analyzer.cache = {}
        
        # 콜론이 포함된 응답
        response_text = """
        카테고리: Invalid JSON Format
        신뢰도: 0.95
        """
        category, confidence = analyzer._fallback_parse_response(response_text)
        
        self.assertEqual(category, "invalid_json_format")
        self.assertEqual(confidence, 0.95)
    
    def test_fallback_parse_response_no_colon(self):
        """콜론이 없는 fallback 응답 파싱 테스트"""
        # 임시 분석기 생성 (초기화 없이)
        analyzer = GeminiFailureAnalyzer.__new__(GeminiFailureAnalyzer)
        analyzer.cache = {}
        
        # 콜론이 없는 응답
        response_text = """
        이것은 형식이 잘못된 응답입니다
        """
        category, confidence = analyzer._fallback_parse_response(response_text)
        
        self.assertEqual(category, "unknown_failure")
        self.assertEqual(confidence, 0.8)
    
    def test_json_parsing_error_handling(self):
        """JSON 파싱 오류 처리 테스트"""
        # 임시 분석기 생성 (초기화 없이)
        analyzer = GeminiFailureAnalyzer.__new__(GeminiFailureAnalyzer)
        analyzer.cache = {}
        
        # 올바르지 않은 JSON 구조
        invalid_json = '{"category": "test", "confidence": "not_a_number"}'
        category, confidence = analyzer._parse_gemini_response(invalid_json)
        
        # fallback 파싱으로 처리되어야 함
        self.assertEqual(category, "unknown_failure")
        self.assertEqual(confidence, 0.8)
    
    def test_missing_required_fields(self):
        """필수 필드가 누락된 응답 처리 테스트"""
        # 임시 분석기 생성 (초기화 없이)
        analyzer = GeminiFailureAnalyzer.__new__(GeminiFailureAnalyzer)
        analyzer.cache = {}
        
        # 필수 필드 누락
        incomplete_response = {
            "category": "test_category"
            # confidence와 explanation이 누락됨
        }
        
        category, confidence = analyzer._parse_gemini_response(json.dumps(incomplete_response))
        
        # 기본값으로 처리되어야 함
        self.assertEqual(category, "test_category")
        self.assertEqual(confidence, 0.8)


if __name__ == "__main__":
    unittest.main()