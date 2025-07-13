"""실패 패턴 분석기 테스트"""

import unittest
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
    
    def test_init(self):
        """분석기 초기화 테스트"""
        analyzer = FailurePatternAnalyzer()
        self.assertIsInstance(analyzer, FailurePatternAnalyzer)
    
    def test_analyze_failure_patterns_empty_cases(self):
        """빈 실패 케이스 분석 테스트"""
        analyzer = FailurePatternAnalyzer()
        
        result = analyzer.analyze_failure_patterns(self.empty_cases)
        
        self.assertEqual(result['total_failures'], 0)
    
    def test_analyze_failure_patterns_with_cases(self):
        """실패 케이스가 있는 경우 분석 테스트"""
        analyzer = FailurePatternAnalyzer()
        result = analyzer.analyze_failure_patterns(self.failed_cases)
        
        self.assertEqual(result['total_failures'], 3)
    
    # 이전 테스트들은 더 이상 사용하지 않는 기능들을 테스트하므로 제거
    
    def test_get_failure_summary(self):
        """실패 패턴 요약 테스트"""
        analyzer = FailurePatternAnalyzer()
        
        patterns = analyzer.analyze_failure_patterns(self.failed_cases)
        summary = analyzer.get_failure_summary(patterns)
        
        self.assertIn('total_failures', summary)
        self.assertEqual(summary['total_failures'], 3)


if __name__ == "__main__":
    unittest.main()