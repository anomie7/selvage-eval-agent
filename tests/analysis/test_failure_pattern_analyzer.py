"""실패 패턴 분석기 테스트"""

import unittest
from unittest.mock import patch, MagicMock
from selvage_eval.analysis.failure_pattern_analyzer import FailurePatternAnalyzer
from selvage_eval.analysis.deepeval_log_parser import TestCaseResult, MetricScore


class TestFailurePatternAnalyzer(unittest.TestCase):
    """실패 패턴 분석기 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        # 실패한 테스트 케이스 결과 샘플 생성
        self.failed_cases = [
            TestCaseResult(
                correctness=MetricScore(0.60, False, "Missing security vulnerabilities"),
                clarity=MetricScore(0.80, True, "Clear explanation"),
                actionability=MetricScore(0.65, False, "Vague suggestions provided"),
                json_correctness=MetricScore(1.0, True, "Valid JSON"),
                input_data='{"test": "input1"}',
                actual_output='{"result": "output1"}',
                raw_content="test content 1"
            ),
            TestCaseResult(
                correctness=MetricScore(0.55, False, "Incorrect analysis of code issue"),
                clarity=MetricScore(0.65, False, "Unclear explanation with technical jargon"),
                actionability=MetricScore(0.70, True, "Actionable suggestions"),
                json_correctness=MetricScore(0.60, False, "JSON format error"),
                input_data='{"test": "input2"}',
                actual_output='{"result": "output2"}',
                raw_content="test content 2"
            ),
            TestCaseResult(
                correctness=MetricScore(0.50, False, "Failed to identify critical issues"),
                clarity=MetricScore(0.75, True, "Clear enough"),
                actionability=MetricScore(0.60, False, "Abstract suggestions without specifics"),
                json_correctness=MetricScore(0.50, False, "Missing required fields"),
                input_data='{"test": "input3"}',
                actual_output='{"result": "output3"}',
                raw_content="test content 3"
            )
        ]
        
        # 빈 케이스
        self.empty_cases = []
    
    @patch('selvage_eval.analysis.failure_pattern_analyzer.GeminiFailureAnalyzer')
    def test_init_with_gemini_success(self, mock_gemini_class):
        """Gemini 초기화 성공 테스트"""
        mock_gemini_instance = MagicMock()
        mock_gemini_class.return_value = mock_gemini_instance
        
        analyzer = FailurePatternAnalyzer()
        
        self.assertIsNotNone(analyzer.gemini_analyzer)
        self.assertEqual(analyzer.gemini_analyzer, mock_gemini_instance)
    
    @patch('selvage_eval.analysis.failure_pattern_analyzer.GeminiFailureAnalyzer')
    def test_init_with_gemini_failure(self, mock_gemini_class):
        """Gemini 초기화 실패 테스트"""
        mock_gemini_class.side_effect = RuntimeError("API key not found")
        
        analyzer = FailurePatternAnalyzer()
        
        self.assertIsNone(analyzer.gemini_analyzer)
    
    @patch('selvage_eval.analysis.failure_pattern_analyzer.GeminiFailureAnalyzer')
    def test_analyze_failure_patterns_empty_cases(self, mock_gemini_class):
        """빈 실패 케이스 분석 테스트"""
        mock_gemini_class.return_value = MagicMock()
        analyzer = FailurePatternAnalyzer()
        
        result = analyzer.analyze_failure_patterns(self.empty_cases)
        
        self.assertEqual(result['total_failures'], 0)
        self.assertEqual(result['by_metric'], {})
        self.assertEqual(result['by_category'], {})
        self.assertEqual(result['critical_patterns'], [])
        self.assertEqual(result['confidence_scores'], {})
    
    @patch('selvage_eval.analysis.failure_pattern_analyzer.GeminiFailureAnalyzer')
    def test_analyze_failure_patterns_with_gemini(self, mock_gemini_class):
        """Gemini 사용 가능한 경우 실패 패턴 분석 테스트"""
        mock_gemini_instance = MagicMock()
        
        # 각 메트릭에 대해 반복적으로 호출되는 categorize_failure 모킹
        def mock_categorize_failure(reason, metric):
            if "security" in reason.lower():
                return ("missing_security_vulnerabilities", 0.9)
            elif "vague" in reason.lower():
                return ("vague_improvement_suggestions", 0.8)
            elif "incorrect" in reason.lower():
                return ("incorrect_analysis", 0.85)
            elif "technical" in reason.lower():
                return ("technical_jargon", 0.7)
            elif "abstract" in reason.lower():
                return ("lack_of_specificity", 0.75)
            elif "format" in reason.lower():
                return ("json_format_error", 0.95)
            elif "missing" in reason.lower():
                return ("missing_issues", 0.8)
            elif "field" in reason.lower():
                return ("missing_fields", 0.9)
            else:
                return ("general_failure", 0.7)
        
        mock_gemini_instance.categorize_failure.side_effect = mock_categorize_failure
        mock_gemini_class.return_value = mock_gemini_instance
        
        analyzer = FailurePatternAnalyzer()
        result = analyzer.analyze_failure_patterns(self.failed_cases)
        
        self.assertEqual(result['total_failures'], 3)
        self.assertTrue(result['gemini_available'])
        
        # 메트릭별 분석 검증
        self.assertIn('correctness', result['by_metric'])
        self.assertIn('clarity', result['by_metric'])
        self.assertIn('actionability', result['by_metric'])
        self.assertIn('json_correctness', result['by_metric'])
        
        # Correctness 메트릭 검증
        correctness_data = result['by_metric']['correctness']
        self.assertEqual(correctness_data['total_failures'], 3)
        self.assertGreater(correctness_data['avg_confidence'], 0.7)
        
        # 카테고리별 분석 검증 (일부 카테고리가 존재하는지 확인)
        self.assertGreater(len(result['by_category']), 0)
    
    @patch('selvage_eval.analysis.failure_pattern_analyzer.GeminiFailureAnalyzer')
    def test_analyze_failure_patterns_without_gemini(self, mock_gemini_class):
        """Gemini 사용 불가능한 경우 실패 패턴 분석 테스트"""
        mock_gemini_class.side_effect = RuntimeError("API key not found")
        
        analyzer = FailurePatternAnalyzer()
        result = analyzer.analyze_failure_patterns(self.failed_cases)
        
        self.assertEqual(result['total_failures'], 3)
        self.assertFalse(result['gemini_available'])
        
        # fallback 분류 결과 검증
        self.assertIn('by_metric', result)
        self.assertIn('by_category', result)
        
        # 신뢰도가 낮아야 함 (fallback 사용)
        self.assertLess(result['confidence_scores']['overall'], 0.7)
    
    @patch('selvage_eval.analysis.failure_pattern_analyzer.GeminiFailureAnalyzer')
    def test_fallback_categorize_failure_correctness(self, mock_gemini_class):
        """Correctness 메트릭 fallback 분류 테스트"""
        mock_gemini_class.side_effect = RuntimeError("API key not found")
        analyzer = FailurePatternAnalyzer()
        
        test_cases = [
            ("Missing security vulnerabilities", "missing_issues"),
            ("Incorrect analysis of the code", "incorrect_analysis"),
            ("Severity assessment is problematic", "severity_misjudgment"),
            ("Some other issue", "general_correctness_issue")
        ]
        
        for reason, expected_category in test_cases:
            with self.subTest(reason=reason):
                category = analyzer._fallback_categorize_failure(reason, 'correctness')
                self.assertEqual(category, expected_category)
    
    @patch('selvage_eval.analysis.failure_pattern_analyzer.GeminiFailureAnalyzer')
    def test_fallback_categorize_failure_clarity(self, mock_gemini_class):
        """Clarity 메트릭 fallback 분류 테스트"""
        mock_gemini_class.side_effect = RuntimeError("API key not found")
        analyzer = FailurePatternAnalyzer()
        
        test_cases = [
            ("Unclear explanation provided", "unclear_explanation"),
            ("Too much technical jargon", "technical_jargon"),
            ("Poor structure in response", "poor_structure"),
            ("Some other clarity issue", "general_clarity_issue")
        ]
        
        for reason, expected_category in test_cases:
            with self.subTest(reason=reason):
                category = analyzer._fallback_categorize_failure(reason, 'clarity')
                self.assertEqual(category, expected_category)
    
    @patch('selvage_eval.analysis.failure_pattern_analyzer.GeminiFailureAnalyzer')
    def test_fallback_categorize_failure_actionability(self, mock_gemini_class):
        """Actionability 메트릭 fallback 분류 테스트"""
        mock_gemini_class.side_effect = RuntimeError("API key not found")
        analyzer = FailurePatternAnalyzer()
        
        test_cases = [
            ("Vague suggestions provided", "vague_suggestions"),
            ("Lacks specific details", "lack_of_specificity"),
            ("Impractical to implement", "impractical_suggestions"),
            ("Some other actionability issue", "general_actionability_issue")
        ]
        
        for reason, expected_category in test_cases:
            with self.subTest(reason=reason):
                category = analyzer._fallback_categorize_failure(reason, 'actionability')
                self.assertEqual(category, expected_category)
    
    @patch('selvage_eval.analysis.failure_pattern_analyzer.GeminiFailureAnalyzer')
    def test_fallback_categorize_failure_json_correctness(self, mock_gemini_class):
        """JSON Correctness 메트릭 fallback 분류 테스트"""
        mock_gemini_class.side_effect = RuntimeError("API key not found")
        analyzer = FailurePatternAnalyzer()
        
        test_cases = [
            ("JSON format error", "json_format_error"),
            ("Schema property violation", "schema_violation"),
            ("Missing required data", "missing_fields"),
            ("Some other JSON issue", "general_json_issue")
        ]
        
        for reason, expected_category in test_cases:
            with self.subTest(reason=reason):
                category = analyzer._fallback_categorize_failure(reason, 'json_correctness')
                self.assertEqual(category, expected_category)
    
    @patch('selvage_eval.analysis.failure_pattern_analyzer.GeminiFailureAnalyzer')
    def test_extract_worst_cases(self, mock_gemini_class):
        """최악 케이스 추출 테스트"""
        mock_gemini_class.side_effect = RuntimeError("API key not found")
        analyzer = FailurePatternAnalyzer()
        
        worst_cases = analyzer._extract_worst_cases(self.failed_cases, 'correctness')
        
        self.assertEqual(len(worst_cases), 3)
        
        # 점수 순으로 정렬되어야 함
        scores = [case['score'] for case in worst_cases]
        self.assertEqual(scores, sorted(scores))
        
        # 가장 낮은 점수가 첫 번째여야 함
        self.assertEqual(worst_cases[0]['score'], 0.50)
        self.assertIn('reason', worst_cases[0])
        self.assertIn('input_preview', worst_cases[0])
    
    @patch('selvage_eval.analysis.failure_pattern_analyzer.GeminiFailureAnalyzer')
    def test_identify_critical_patterns(self, mock_gemini_class):
        """중요한 패턴 식별 테스트"""
        mock_gemini_class.side_effect = RuntimeError("API key not found")
        analyzer = FailurePatternAnalyzer()
        
        # 패턴 분석 결과 생성
        patterns = analyzer.analyze_failure_patterns(self.failed_cases)
        critical_patterns = analyzer._identify_critical_patterns(patterns)
        
        self.assertIsInstance(critical_patterns, list)
        
        # 중요한 패턴이 있어야 함
        if critical_patterns:
            pattern = critical_patterns[0]
            self.assertIn('category', pattern)
            self.assertIn('count', pattern)
            self.assertIn('percentage', pattern)
            self.assertIn('severity', pattern)
            self.assertIn('reason', pattern)
            self.assertIn(pattern['severity'], ['high', 'medium', 'low'])
    
    @patch('selvage_eval.analysis.failure_pattern_analyzer.GeminiFailureAnalyzer')
    def test_get_failure_summary(self, mock_gemini_class):
        """실패 패턴 요약 테스트"""
        mock_gemini_class.side_effect = RuntimeError("API key not found")
        analyzer = FailurePatternAnalyzer()
        
        patterns = analyzer.analyze_failure_patterns(self.failed_cases)
        summary = analyzer.get_failure_summary(patterns)
        
        self.assertIn('total_failures', summary)
        self.assertIn('top_categories', summary)
        self.assertIn('most_problematic_metric', summary)
        self.assertIn('critical_patterns_count', summary)
        self.assertIn('gemini_available', summary)
        
        self.assertEqual(summary['total_failures'], 3)
        self.assertFalse(summary['gemini_available'])
        
        # 가장 문제가 많은 메트릭 검증
        if summary['most_problematic_metric']:
            problematic_metric = summary['most_problematic_metric']
            self.assertIn('metric', problematic_metric)
            self.assertIn('failures', problematic_metric)
            self.assertIn('failure_rate', problematic_metric)
    
    @patch('selvage_eval.analysis.failure_pattern_analyzer.GeminiFailureAnalyzer')
    def test_gemini_error_handling(self, mock_gemini_class):
        """Gemini 오류 처리 테스트"""
        mock_gemini_instance = MagicMock()
        mock_gemini_instance.categorize_failure.side_effect = Exception("Network error")
        mock_gemini_class.return_value = mock_gemini_instance
        
        analyzer = FailurePatternAnalyzer()
        result = analyzer.analyze_failure_patterns(self.failed_cases)
        
        # 오류 발생 시 fallback 분류가 사용되어야 함
        self.assertEqual(result['total_failures'], 3)
        self.assertTrue(result['gemini_available'])  # Gemini는 사용 가능하지만 오류 발생
        
        # 신뢰도가 낮아야 함 (fallback 사용)
        self.assertLess(result['confidence_scores']['overall'], 0.7)
    
    @patch('selvage_eval.analysis.failure_pattern_analyzer.GeminiFailureAnalyzer')
    def test_unknown_metric_fallback(self, mock_gemini_class):
        """알 수 없는 메트릭에 대한 fallback 분류 테스트"""
        mock_gemini_class.side_effect = RuntimeError("API key not found")
        analyzer = FailurePatternAnalyzer()
        
        category = analyzer._fallback_categorize_failure("Some failure reason", "unknown_metric")
        self.assertEqual(category, "unknown_failure")


if __name__ == "__main__":
    unittest.main()