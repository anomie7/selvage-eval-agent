"""DeepEval 로그 파서 테스트"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from selvage_eval.analysis.deepeval_log_parser import DeepEvalLogParser, TestCaseResult, MetricScore


class TestDeepEvalLogParser(unittest.TestCase):
    """DeepEval 로그 파서 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.parser = DeepEvalLogParser()
        
        # 샘플 로그 콘텐츠 생성
        self.sample_log_content = """
======================================================================

Metrics Summary

  - ✅ Correctness [GEval] (score: 0.85, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: "The review correctly identifies the security vulnerability in the code.", error: None)
  - ✅ Clarity [GEval] (score: 0.90, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: "The explanation is clear and easy to understand.", error: None)  
  - ✅ Actionability [GEval] (score: 0.75, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: "Provides specific actionable suggestions.", error: None)
  - ✅ Json Correctness (score: 1.0, threshold: 1.0, strict: True, evaluation model: None, reason: "JSON format is valid and follows the schema.", error: None)

For test case:

  - input: [{"role": "system", "content": "You are a code reviewer."}, {"role": "user", "content": "Review this code: function test() { return 'hello'; }"}]
  - actual output: {"issues": [{"type": "suggestion", "line": 1, "description": "Consider using arrow function syntax"}]}

======================================================================

Metrics Summary

  - ❌ Correctness [GEval] (score: 0.60, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: "The review missed important security issues.", error: None)
  - ✅ Clarity [GEval] (score: 0.80, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: "Clear explanation provided.", error: None)  
  - ❌ Actionability [GEval] (score: 0.65, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: "Suggestions are too vague.", error: None)
  - ✅ Json Correctness (score: 1.0, threshold: 1.0, strict: True, evaluation model: None, reason: "Valid JSON format.", error: None)

For test case:

  - input: [{"role": "system", "content": "You are a security-focused code reviewer."}, {"role": "user", "content": "Review this code: var userInput = req.body.query; eval(userInput);"}]
  - actual output: {"issues": [{"type": "warning", "line": 1, "description": "Avoid using eval() function"}]}
"""
    
    def test_parse_log_file(self):
        """로그 파일 파싱 테스트"""
        # 임시 로그 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write(self.sample_log_content)
            temp_path = Path(f.name)
        
        try:
            # 로그 파일 파싱
            results = list(self.parser.parse_log_file(temp_path))
            
            # 결과 검증
            self.assertEqual(len(results), 2)
            
            # 첫 번째 테스트 케이스 검증
            first_result = results[0]
            self.assertIsInstance(first_result, TestCaseResult)
            self.assertEqual(first_result.correctness.score, 0.85)
            self.assertTrue(first_result.correctness.passed)
            self.assertEqual(first_result.clarity.score, 0.90)
            self.assertTrue(first_result.clarity.passed)
            self.assertEqual(first_result.actionability.score, 0.75)
            self.assertTrue(first_result.actionability.passed)
            self.assertEqual(first_result.json_correctness.score, 1.0)
            self.assertTrue(first_result.json_correctness.passed)
            
            # 두 번째 테스트 케이스 검증
            second_result = results[1]
            self.assertEqual(second_result.correctness.score, 0.60)
            self.assertFalse(second_result.correctness.passed)
            self.assertEqual(second_result.clarity.score, 0.80)
            self.assertTrue(second_result.clarity.passed)
            self.assertEqual(second_result.actionability.score, 0.65)
            self.assertFalse(second_result.actionability.passed)
            self.assertEqual(second_result.json_correctness.score, 1.0)
            self.assertTrue(second_result.json_correctness.passed)
            
        finally:
            # 임시 파일 삭제
            temp_path.unlink()
    
    def test_parse_test_case(self):
        """개별 테스트 케이스 파싱 테스트"""
        lines = [
            "Metrics Summary\n",
            "  - ✅ Correctness [GEval] (score: 0.85, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: \"Good analysis\", error: None)\n",
            "  - ❌ Clarity [GEval] (score: 0.65, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: \"Unclear explanation\", error: None)\n",
            "  - ✅ Actionability [GEval] (score: 0.80, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: \"Actionable suggestions\", error: None)\n",
            "  - ✅ Json Correctness (score: 1.0, threshold: 1.0, strict: True, evaluation model: None, reason: \"Valid JSON\", error: None)\n",
            "For test case:\n",
            "  - input: [{\"role\": \"user\", \"content\": \"test\"}]\n",
            "  - actual output: {\"result\": \"test\"}\n"
        ]
        
        result = self.parser._parse_test_case(lines)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.correctness.score, 0.85)
        self.assertTrue(result.correctness.passed)
        self.assertEqual(result.clarity.score, 0.65)
        self.assertFalse(result.clarity.passed)
        self.assertEqual(result.actionability.score, 0.80)
        self.assertTrue(result.actionability.passed)
        self.assertEqual(result.json_correctness.score, 1.0)
        self.assertTrue(result.json_correctness.passed)
        
        # 실패 사유 확인
        self.assertEqual(result.correctness.reason, "Good analysis")
        self.assertEqual(result.clarity.reason, "Unclear explanation")
    
    def test_convert_to_test_case_result(self):
        """딕셔너리 데이터를 TestCaseResult로 변환 테스트"""
        test_case_data = {
            'metrics': {
                'correctness': {
                    'score': 0.85,
                    'passed': True,
                    'reason': 'Good analysis',
                    'error': None
                },
                'clarity': {
                    'score': 0.70,
                    'passed': True,
                    'reason': 'Clear explanation',
                    'error': None
                },
                'actionability': {
                    'score': 0.60,
                    'passed': False,
                    'reason': 'Vague suggestions',
                    'error': None
                },
                'json_correctness': {
                    'score': 1.0,
                    'passed': True,
                    'reason': 'Valid JSON',
                    'error': None
                }
            },
            'input': '[{"role": "user", "content": "test"}]',
            'actual_output': '{"result": "test"}',
            'raw_content': 'test content'
        }
        
        result = self.parser.convert_to_test_case_result(test_case_data)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.correctness.score, 0.85)
        self.assertTrue(result.correctness.passed)
        self.assertEqual(result.clarity.score, 0.70)
        self.assertTrue(result.clarity.passed)
        self.assertEqual(result.actionability.score, 0.60)
        self.assertFalse(result.actionability.passed)
        self.assertEqual(result.json_correctness.score, 1.0)
        self.assertTrue(result.json_correctness.passed)
        
        self.assertEqual(result.input_data, '[{"role": "user", "content": "test"}]')
        self.assertEqual(result.actual_output, '{"result": "test"}')
        self.assertEqual(result.raw_content, 'test content')
    
    def test_missing_metrics_handling(self):
        """누락된 메트릭 처리 테스트"""
        lines = [
            "Metrics Summary\n",
            "  - ✅ Correctness [GEval] (score: 0.85, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: \"Good analysis\", error: None)\n",
            "For test case:\n",
            "  - input: [{\"role\": \"user\", \"content\": \"test\"}]\n",
            "  - actual output: {\"result\": \"test\"}\n"
        ]
        
        result = self.parser._parse_test_case(lines)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.correctness.score, 0.85)
        self.assertTrue(result.correctness.passed)
        
        # 누락된 메트릭들이 기본값으로 설정되었는지 확인
        self.assertEqual(result.clarity.score, 0.0)
        self.assertFalse(result.clarity.passed)
        self.assertEqual(result.clarity.reason, "메트릭 정보 없음")
        
        self.assertEqual(result.actionability.score, 0.0)
        self.assertFalse(result.actionability.passed)
        
        self.assertEqual(result.json_correctness.score, 0.0)
        self.assertFalse(result.json_correctness.passed)
    
    def test_empty_log_file(self):
        """빈 로그 파일 처리 테스트"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("")
            temp_path = Path(f.name)
        
        try:
            results = list(self.parser.parse_log_file(temp_path))
            self.assertEqual(len(results), 0)
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    unittest.main()