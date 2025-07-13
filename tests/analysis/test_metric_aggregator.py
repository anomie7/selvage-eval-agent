"""메트릭 집계기 테스트"""

import unittest
from selvage_eval.analysis.metric_aggregator import MetricAggregator
from selvage_eval.analysis.deepeval_log_parser import TestCaseResult, MetricScore


class TestMetricAggregator(unittest.TestCase):
    """메트릭 집계기 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.aggregator = MetricAggregator()
        
        # 샘플 테스트 케이스 결과 생성
        self.test_results = [
            TestCaseResult(
                correctness=MetricScore(0.85, True, "Good analysis"),
                clarity=MetricScore(0.90, True, "Clear explanation"),
                actionability=MetricScore(0.75, True, "Actionable suggestions"),
                json_correctness=MetricScore(1.0, True, "Valid JSON"),
                input_data='{"test": "input"}',
                actual_output='{"result": "output"}',
                raw_content="test content 1"
            ),
            TestCaseResult(
                correctness=MetricScore(0.60, False, "Missed issues"),
                clarity=MetricScore(0.80, True, "Clear enough"),
                actionability=MetricScore(0.65, False, "Vague suggestions"),
                json_correctness=MetricScore(1.0, True, "Valid JSON"),
                input_data='{"test": "input2"}',
                actual_output='{"result": "output2"}',
                raw_content="test content 2"
            ),
            TestCaseResult(
                correctness=MetricScore(0.95, True, "Excellent analysis"),
                clarity=MetricScore(0.85, True, "Very clear"),
                actionability=MetricScore(0.90, True, "Highly actionable"),
                json_correctness=MetricScore(1.0, True, "Perfect JSON"),
                input_data='{"test": "input3"}',
                actual_output='{"result": "output3"}',
                raw_content="test content 3"
            )
        ]
    
    def test_aggregate_model_performance(self):
        """모델 성능 집계 테스트"""
        result = self.aggregator.aggregate_model_performance(self.test_results)
        
        # 기본 구조 검증
        self.assertIn('correctness', result)
        self.assertIn('clarity', result)
        self.assertIn('actionability', result)
        self.assertIn('json_correctness', result)
        self.assertIn('overall', result)
        
        # Correctness 메트릭 검증
        correctness = result['correctness']
        expected_mean = (0.85 + 0.60 + 0.95) / 3
        self.assertAlmostEqual(correctness['mean_score'], expected_mean, places=3)
        self.assertEqual(correctness['total_cases'], 3)
        self.assertEqual(correctness['passed_cases'], 2)
        self.assertEqual(correctness['failed_cases'], 1)
        self.assertAlmostEqual(correctness['pass_rate'], 2/3, places=3)
        
        # Clarity 메트릭 검증
        clarity = result['clarity']
        self.assertEqual(clarity['passed_cases'], 3)
        self.assertEqual(clarity['failed_cases'], 0)
        self.assertEqual(clarity['pass_rate'], 1.0)
        
        # Overall 점수 검증
        overall = result['overall']
        self.assertIn('weighted_score', overall)
        self.assertIn('grade', overall)
        self.assertIn('consistency', overall)
        self.assertTrue(0 <= overall['weighted_score'] <= 1)
        self.assertIn(overall['grade'], ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D', 'F'])
    
    def test_empty_performance_data(self):
        """빈 성능 데이터 테스트"""
        result = self.aggregator.aggregate_model_performance([])
        
        # 모든 메트릭이 0으로 초기화되어야 함
        for metric in ['correctness', 'clarity', 'actionability', 'json_correctness']:
            self.assertEqual(result[metric]['mean_score'], 0.0)
            self.assertEqual(result[metric]['total_cases'], 0)
            self.assertEqual(result[metric]['passed_cases'], 0)
            self.assertEqual(result[metric]['failed_cases'], 0)
            self.assertEqual(result[metric]['pass_rate'], 0.0)
        
        self.assertEqual(result['overall']['weighted_score'], 0.0)
        self.assertEqual(result['overall']['grade'], 'F')
    
    def test_assign_grade(self):
        """등급 할당 테스트"""
        # 각 점수 범위별 등급 확인
        test_cases = [
            (0.95, 'A+'),
            (0.90, 'A+'),
            (0.87, 'A'),
            (0.82, 'B+'),
            (0.77, 'B'),
            (0.72, 'C+'),
            (0.67, 'C'),
            (0.62, 'D'),
            (0.50, 'F')
        ]
        
        for score, expected_grade in test_cases:
            with self.subTest(score=score):
                grade = self.aggregator._assign_grade(score)
                self.assertEqual(grade, expected_grade)
    
    def test_calculate_metric_statistics(self):
        """메트릭 통계 계산 테스트"""
        stats = self.aggregator.calculate_metric_statistics(self.test_results)
        
        # 기본 구조 검증
        self.assertIn('correctness', stats)
        self.assertIn('clarity', stats)
        self.assertIn('actionability', stats)
        self.assertIn('json_correctness', stats)
        
        # Correctness 통계 검증
        correctness_stats = stats['correctness']
        expected_mean = (0.85 + 0.60 + 0.95) / 3
        self.assertAlmostEqual(correctness_stats['mean'], expected_mean, places=3)
        self.assertEqual(correctness_stats['min'], 0.60)
        self.assertEqual(correctness_stats['max'], 0.95)
        self.assertEqual(correctness_stats['median'], 0.85)
        
        # 통계 필드 존재 확인
        for metric in stats.values():
            self.assertIn('mean', metric)
            self.assertIn('std', metric)
            self.assertIn('min', metric)
            self.assertIn('max', metric)
            self.assertIn('median', metric)
            self.assertIn('q1', metric)
            self.assertIn('q3', metric)
            self.assertIn('iqr', metric)
            self.assertIn('coefficient_of_variation', metric)
    
    def test_identify_performance_outliers(self):
        """성능 이상치 식별 테스트"""
        # 더 극단적인 이상치가 있는 데이터 생성
        outlier_results = [
            TestCaseResult(
                correctness=MetricScore(0.80, True, "Good analysis"),
                clarity=MetricScore(0.85, True, "Clear"),
                actionability=MetricScore(0.80, True, "Actionable"),
                json_correctness=MetricScore(1.0, True, "Valid JSON"),
                input_data='{"test": "input1"}',
                actual_output='{"result": "output1"}',
                raw_content="test content 1"
            ),
            TestCaseResult(
                correctness=MetricScore(0.82, True, "Good analysis"),
                clarity=MetricScore(0.87, True, "Clear"),
                actionability=MetricScore(0.83, True, "Actionable"),
                json_correctness=MetricScore(1.0, True, "Valid JSON"),
                input_data='{"test": "input2"}',
                actual_output='{"result": "output2"}',
                raw_content="test content 2"
            ),
            TestCaseResult(
                correctness=MetricScore(0.85, True, "Good analysis"),
                clarity=MetricScore(0.90, True, "Clear"),
                actionability=MetricScore(0.85, True, "Actionable"),
                json_correctness=MetricScore(1.0, True, "Valid JSON"),
                input_data='{"test": "input3"}',
                actual_output='{"result": "output3"}',
                raw_content="test content 3"
            ),
            TestCaseResult(
                correctness=MetricScore(0.20, False, "Very poor analysis"),  # 명확한 이상치
                clarity=MetricScore(0.85, True, "Clear"),
                actionability=MetricScore(0.80, True, "Actionable"),
                json_correctness=MetricScore(1.0, True, "Valid JSON"),
                input_data='{"test": "outlier"}',
                actual_output='{"result": "outlier"}',
                raw_content="outlier content"
            )
        ]
        
        outliers = self.aggregator.identify_performance_outliers(outlier_results)
        
        # 기본 구조 검증
        self.assertIn('correctness', outliers)
        self.assertIn('clarity', outliers)
        self.assertIn('actionability', outliers)
        self.assertIn('json_correctness', outliers)
        
        # Correctness에 이상치가 있어야 함
        correctness_outliers = outliers['correctness']
        self.assertTrue(len(correctness_outliers) > 0)
        
        # 이상치 정보 검증
        outlier = correctness_outliers[0]
        self.assertIn('test_case_index', outlier)
        self.assertIn('score', outlier)
        self.assertIn('reason', outlier)
        self.assertIn('type', outlier)
        self.assertIn(outlier['type'], ['low', 'high'])
        self.assertEqual(outlier['score'], 0.20)
        self.assertEqual(outlier['type'], 'low')
    
    def test_calculate_consistency_metrics(self):
        """일관성 메트릭 계산 테스트"""
        consistency = self.aggregator.calculate_consistency_metrics(self.test_results)
        
        # 기본 구조 검증
        self.assertIn('overall_consistency', consistency)
        self.assertIn('metric_balance', consistency)
        self.assertIn('consistency_by_metric', consistency)
        
        # 일관성 점수 범위 검증
        self.assertTrue(0 <= consistency['overall_consistency'] <= 1)
        self.assertTrue(0 <= consistency['metric_balance'] <= 1)
        
        # 메트릭별 일관성 점수 검증
        metric_consistency = consistency['consistency_by_metric']
        for metric in ['correctness', 'clarity', 'actionability', 'json_correctness']:
            self.assertIn(metric, metric_consistency)
            self.assertTrue(0 <= metric_consistency[metric] <= 1)
    
    def test_single_test_result(self):
        """단일 테스트 결과 처리 테스트"""
        single_result = [self.test_results[0]]
        
        aggregated = self.aggregator.aggregate_model_performance(single_result)
        
        # 단일 결과에 대한 검증
        self.assertEqual(aggregated['correctness']['total_cases'], 1)
        self.assertEqual(aggregated['correctness']['passed_cases'], 1)
        self.assertEqual(aggregated['correctness']['failed_cases'], 0)
        self.assertEqual(aggregated['correctness']['pass_rate'], 1.0)
        self.assertEqual(aggregated['correctness']['mean_score'], 0.85)
        self.assertEqual(aggregated['correctness']['std_score'], 0.0)  # 단일 값이므로 표준편차는 0
    
    def test_all_failed_results(self):
        """모든 테스트가 실패한 경우 처리 테스트"""
        failed_results = [
            TestCaseResult(
                correctness=MetricScore(0.60, False, "Poor analysis"),
                clarity=MetricScore(0.65, False, "Unclear"),
                actionability=MetricScore(0.60, False, "Not actionable"),
                json_correctness=MetricScore(0.50, False, "Invalid JSON"),
                input_data='{"test": "input"}',
                actual_output='{"result": "output"}',
                raw_content="test content"
            )
        ]
        
        aggregated = self.aggregator.aggregate_model_performance(failed_results)
        
        # 모든 메트릭이 실패해야 함
        for metric in ['correctness', 'clarity', 'actionability', 'json_correctness']:
            self.assertEqual(aggregated[metric]['passed_cases'], 0)
            self.assertEqual(aggregated[metric]['failed_cases'], 1)
            self.assertEqual(aggregated[metric]['pass_rate'], 0.0)
        
        self.assertEqual(aggregated['overall']['pass_rate'], 0.0)


if __name__ == "__main__":
    unittest.main()