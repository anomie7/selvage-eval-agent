"""DeepEval 로그 파서 테스트"""

import tempfile
import unittest
from pathlib import Path

from selvage_eval.analysis.deepeval_log_parser import DeepEvalLogParser, TestCaseResult


class TestDeepEvalLogParser(unittest.TestCase):
    """DeepEval 로그 파서 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.parser = DeepEvalLogParser()
        
        # 샘플 로그 콘텐츠 생성 (실제 DeepEval 로그 형식)
        self.sample_log_content = """
======================================================================

Metrics Summary

  - ✅ Correctness [GEval] (score: 1.0, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: The model correctly identified that the submitted code is a large-scale, high-quality refactoring and that there are no pertinent issues to report, justifiably leaving the 'issues' array empty. The summary accurately describes the introduction of the `MessageStateHandler` class and its positive impact on separating concerns from the `Task` class, which improves maintainability. Furthermore, the recommendations provide a valuable suggestion for improving type safety by replacing `any` with more specific error types, which is an appropriate use of the recommendations field for non-critical improvements., error: None)
  - ✅ Clarity [GEval] (score: 1.0, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: The output provides a concise summary that clearly explains the purpose and benefits of the refactoring. The recommendations are specific and actionable, particularly the suggestion to improve type safety in error handling. This point is well-supported by a clear, easy-to-understand code example, demonstrating strong alignment with all evaluation steps., error: None)
  - ✅ Actionability [GEval] (score: 1.0, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: The response provides a high-quality assessment. The `summary` accurately describes the major refactoring that introduced the `MessageStateHandler` and correctly evaluates its positive impact on code quality and maintainability. Although no critical issues were found, the `recommendations` section offers a specific, actionable, and valuable suggestion to improve type safety by replacing `any` in `catch` blocks. This suggestion is practical, includes a clear code example, and would lead to a substantial improvement in code quality, demonstrating a thorough and constructive review., error: None)
  - ✅ Json Correctness (score: 1.0, threshold: 1.0, strict: True, evaluation model: None, reason: The generated Json matches and is syntactically correct to the expected schema., error: None)

For test case:

  - input: (생략)
  - actual output: (생략)
  - expected output: None
  - context: None
  - retrieval context: None

======================================================================
Overall Metric Pass Rates
Correctness [GEval]: 100.00% pass rateClarity [GEval]: 100.00% pass rateActionability [GEval]: 100.00% pass rateJson Correctness: 100.00% pass rate
======================================================================

======================================================================

Metrics Summary

  - ❌ Correctness [GEval] (score: 0.6, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: The response correctly identifies two key issues: the use of outdated GitHub Actions versions in `.github/workflows/build.yml` and the duplicated SonarCloud properties across multiple `pom.xml` files. The details for these issues, including file paths, line numbers, types, and severities, are accurate. However, the assessment is incomplete as it fails to report other pertinent issues within the modified files. For instance, it missed hardcoded dependency versions (e.g., `spring-cloud-starter-circuitbreaker-reactor-resilience4j` in `notification-service/pom.xml`) which contradicts Maven's best practice of using `dependencyManagement` in a multi-module project. It also overlooked several instances of commented-out code blocks in various `pom.xml` files, which is a notable style issue., error: None)
  - ✅ Clarity [GEval] (score: 1.0, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: The response provides a high-quality and thorough code review. The language is concise and direct across the summary, issues, and recommendations. The issues identified are specific and clear, referencing outdated action versions (`v3`) in the `build.yml` and duplicated properties in the `pom.xml`. The purpose of the original change is well-understood and articulated in the summary. Finally, the improved code examples are easy to understand, clearly showing the version updates and providing a helpful, commented-out example for refactoring the `pom.xml` structure., error: None)
  - ✅ Actionability [GEval] (score: 1.0, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: The response provides specific, actionable solutions for each identified issue. For the `.github/workflows/build.yml` file, it correctly suggests updating the GitHub Actions versions and provides the exact code. For the duplicated properties in multiple `pom.xml` files, it offers a practical and substantial improvement by recommending a parent POM, complete with a clear code example. All suggestions are implementable, would significantly improve code quality and maintainability, and are highly relevant to the project's context as a multi-module Maven project., error: None)
  - ✅ Json Correctness (score: 1.0, threshold: 1.0, strict: True, evaluation model: None, reason: The generated Json matches and is syntactically correct to the expected schema., error: None)

For test case:

  - input: (생략)
  - actual output: (생략)
  - expected output: None
  - context: None
  - retrieval context: None

======================================================================
Overall Metric Pass Rates
Correctness [GEval]: 40.00% pass rateClarity [GEval]: 60.00% pass rateActionability [GEval]: 80.00% pass rateJson Correctness: 100.00% pass rate
======================================================================
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
            
            # 결과 검증 (Overall Metric Pass Rates 섹션은 필터링되어 2개 테스트 케이스만 파싱됨)
            self.assertEqual(len(results), 2)
            
            # 첫 번째 테스트 케이스 검증 (모든 메트릭 성공)
            first_result = results[0]
            self.assertIsInstance(first_result, TestCaseResult)
            self.assertEqual(first_result.correctness.score, 1.0)
            self.assertTrue(first_result.correctness.passed)
            self.assertEqual(first_result.clarity.score, 1.0)
            self.assertTrue(first_result.clarity.passed)
            self.assertEqual(first_result.actionability.score, 1.0)
            self.assertTrue(first_result.actionability.passed)
            self.assertEqual(first_result.json_correctness.score, 1.0)
            self.assertTrue(first_result.json_correctness.passed)
            
            # 두 번째 테스트 케이스 검증 (Correctness 실패, 나머지 성공)
            second_result = results[1]
            self.assertEqual(second_result.correctness.score, 0.6)
            self.assertFalse(second_result.correctness.passed)
            self.assertEqual(second_result.clarity.score, 1.0)
            self.assertTrue(second_result.clarity.passed)
            self.assertEqual(second_result.actionability.score, 1.0)
            self.assertTrue(second_result.actionability.passed)
            self.assertEqual(second_result.json_correctness.score, 1.0)
            self.assertTrue(second_result.json_correctness.passed)
            
        finally:
            # 임시 파일 삭제
            temp_path.unlink()
    
    def test_parse_test_case(self):
        """개별 테스트 케이스 파싱 테스트"""
        lines = [
            "Metrics Summary\n",
            "  - ✅ Correctness [GEval] (score: 0.85, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: The model correctly identified the key improvements in the refactoring process. The analysis shows good understanding of the code structure and identifies relevant issues appropriately., error: None)\n",
            "  - ❌ Clarity [GEval] (score: 0.65, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: The explanation lacks sufficient detail and could be clearer. Some technical terms are used without proper context, making it difficult for developers to understand the specific issues., error: None)\n",
            "  - ✅ Actionability [GEval] (score: 0.80, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: The suggestions provide concrete steps that can be implemented. Examples include specific code changes and best practices that developers can follow immediately., error: None)\n",
            "  - ✅ Json Correctness (score: 1.0, threshold: 1.0, strict: True, evaluation model: None, reason: The generated JSON structure matches the expected schema perfectly with all required fields present., error: None)\n",
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
        self.assertIn("model correctly identified", result.correctness.reason)
        self.assertIn("lacks sufficient detail", result.clarity.reason)
    
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
            "  - ✅ Correctness [GEval] (score: 0.85, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: The model correctly identified the key improvements in the refactoring process. The analysis shows good understanding of the code structure and identifies relevant issues appropriately., error: None)\n",
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