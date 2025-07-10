"""모델 성능 비교기 테스트"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from selvage_eval.analysis.model_performance_comparator import ModelPerformanceComparator
from selvage_eval.analysis.deepeval_log_parser import TestCaseResult, MetricScore


class TestModelPerformanceComparator(unittest.TestCase):
    """모델 성능 비교기 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.comparator = ModelPerformanceComparator()
        
        # 3개 모델의 샘플 테스트 결과 생성
        self.model_results = {
            'model_a': [
                TestCaseResult(
                    correctness=MetricScore(0.90, True, "Excellent analysis"),
                    clarity=MetricScore(0.85, True, "Very clear"),
                    actionability=MetricScore(0.80, True, "Good suggestions"),
                    json_correctness=MetricScore(1.0, True, "Perfect JSON"),
                    input_data='{"test": "input1"}',
                    actual_output='{"result": "output1"}',
                    raw_content="test content 1"
                ),
                TestCaseResult(
                    correctness=MetricScore(0.85, True, "Good analysis"),
                    clarity=MetricScore(0.90, True, "Clear explanation"),
                    actionability=MetricScore(0.75, True, "Actionable"),
                    json_correctness=MetricScore(1.0, True, "Valid JSON"),
                    input_data='{"test": "input2"}',
                    actual_output='{"result": "output2"}',
                    raw_content="test content 2"
                )
            ],
            'model_b': [
                TestCaseResult(
                    correctness=MetricScore(0.75, True, "Decent analysis"),
                    clarity=MetricScore(0.80, True, "Clear enough"),
                    actionability=MetricScore(0.70, False, "Somewhat vague"),
                    json_correctness=MetricScore(0.95, True, "Good JSON"),
                    input_data='{"test": "input3"}',
                    actual_output='{"result": "output3"}',
                    raw_content="test content 3"
                ),
                TestCaseResult(
                    correctness=MetricScore(0.80, True, "Good analysis"),
                    clarity=MetricScore(0.75, True, "Acceptable clarity"),
                    actionability=MetricScore(0.65, False, "Needs improvement"),
                    json_correctness=MetricScore(0.90, True, "Valid JSON"),
                    input_data='{"test": "input4"}',
                    actual_output='{"result": "output4"}',
                    raw_content="test content 4"
                )
            ],
            'model_c': [
                TestCaseResult(
                    correctness=MetricScore(0.60, False, "Poor analysis"),
                    clarity=MetricScore(0.70, False, "Unclear"),
                    actionability=MetricScore(0.60, False, "Vague suggestions"),
                    json_correctness=MetricScore(0.80, True, "Acceptable JSON"),
                    input_data='{"test": "input5"}',
                    actual_output='{"result": "output5"}',
                    raw_content="test content 5"
                )
            ]
        }
    
    def test_compare_models_empty_input(self):
        """빈 입력에 대한 테스트"""
        result = self.comparator.compare_models({})
        
        self.assertEqual(result['model_count'], 0)
        self.assertEqual(result['model_statistics'], {})
        self.assertEqual(result['comparison_table'], {})
        self.assertEqual(result['rankings'], {})
        self.assertEqual(result['statistical_analysis'], {})
        self.assertEqual(result['recommendations'], [])
    
    def test_compare_models_basic_functionality(self):
        """기본 모델 비교 기능 테스트"""
        result = self.comparator.compare_models(self.model_results)
        
        # 기본 구조 검증
        self.assertEqual(result['model_count'], 3)
        self.assertIn('model_statistics', result)
        self.assertIn('comparison_table', result)
        self.assertIn('rankings', result)
        self.assertIn('statistical_analysis', result)
        self.assertIn('recommendations', result)
        
        # 모델 통계 검증
        self.assertIn('model_a', result['model_statistics'])
        self.assertIn('model_b', result['model_statistics'])
        self.assertIn('model_c', result['model_statistics'])
        
        # 각 모델의 통계에 필요한 키가 있는지 확인
        for model_name in ['model_a', 'model_b', 'model_c']:
            model_stats = result['model_statistics'][model_name]
            self.assertIn('overall', model_stats)
            self.assertIn('correctness', model_stats)
            self.assertIn('clarity', model_stats)
            self.assertIn('actionability', model_stats)
            self.assertIn('json_correctness', model_stats)
    
    def test_create_comparison_table(self):
        """비교 표 생성 테스트"""
        result = self.comparator.compare_models(self.model_results)
        comparison_table = result['comparison_table']
        
        # 비교 표 구조 검증
        self.assertIn('table_data', comparison_table)
        self.assertIn('metrics', comparison_table)
        self.assertIn('tier_distribution', comparison_table)
        self.assertIn('summary', comparison_table)
        
        # 표 데이터 검증
        table_data = comparison_table['table_data']
        self.assertEqual(len(table_data), 3)
        
        # 첫 번째 모델 데이터 검증
        first_model = table_data[0]
        self.assertIn('model_name', first_model)
        self.assertIn('overall_score', first_model)
        self.assertIn('overall_rank', first_model)
        self.assertIn('grade', first_model)
        self.assertIn('tier', first_model)
        
        # 메트릭별 데이터 검증
        for metric in ['correctness', 'clarity', 'actionability', 'json_correctness']:
            self.assertIn(f'{metric}_score', first_model)
            self.assertIn(f'{metric}_rank', first_model)
            self.assertIn(f'{metric}_pass_rate', first_model)
        
        # 점수 순으로 정렬되었는지 확인
        scores = [model['overall_score'] for model in table_data]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_calculate_model_rankings(self):
        """모델 순위 계산 테스트"""
        result = self.comparator.compare_models(self.model_results)
        rankings = result['rankings']
        
        # 순위 구조 검증
        self.assertIn('metric_rankings', rankings)
        self.assertIn('overall_ranking', rankings)
        self.assertIn('ranking_summary', rankings)
        
        # 메트릭별 순위 검증
        metric_rankings = rankings['metric_rankings']
        for metric in ['correctness', 'clarity', 'actionability', 'json_correctness']:
            self.assertIn(metric, metric_rankings)
            metric_ranking = metric_rankings[metric]
            
            # 모든 모델이 순위에 포함되어야 함
            self.assertEqual(len(metric_ranking), 3)
            self.assertIn('model_a', metric_ranking)
            self.assertIn('model_b', metric_ranking)
            self.assertIn('model_c', metric_ranking)
            
            # 순위는 1부터 시작하고 연속적이어야 함
            ranks = list(metric_ranking.values())
            self.assertEqual(set(ranks), {1, 2, 3})
        
        # 종합 순위 검증
        overall_ranking = rankings['overall_ranking']
        self.assertEqual(len(overall_ranking), 3)
        overall_ranks = list(overall_ranking.values())
        self.assertEqual(set(overall_ranks), {1, 2, 3})
    
    def test_statistical_analysis_with_sufficient_data(self):
        """충분한 데이터가 있는 경우 통계 분석 테스트"""
        result = self.comparator.compare_models(self.model_results)
        statistical_analysis = result['statistical_analysis']
        
        # 분석 수행 여부 확인
        self.assertTrue(statistical_analysis['analysis_performed'])
        self.assertEqual(statistical_analysis['model_count'], 3)
        
        # 메트릭별 분석 확인
        self.assertIn('metric_analyses', statistical_analysis)
        metric_analyses = statistical_analysis['metric_analyses']
        
        for metric in ['correctness', 'clarity', 'actionability', 'json_correctness']:
            self.assertIn(metric, metric_analyses)
            metric_analysis = metric_analyses[metric]
            
            if metric_analysis['test_performed']:
                self.assertIn('anova', metric_analysis)
                self.assertIn('kruskal_wallis', metric_analysis)
                self.assertIn('sample_sizes', metric_analysis)
                self.assertIn('model_names', metric_analysis)
                
                # ANOVA 결과 검증
                anova = metric_analysis['anova']
                self.assertIn('f_statistic', anova)
                self.assertIn('p_value', anova)
                self.assertIn('significant', anova)
                self.assertIn('interpretation', anova)
                
                # Kruskal-Wallis 결과 검증
                kw = metric_analysis['kruskal_wallis']
                self.assertIn('h_statistic', kw)
                self.assertIn('p_value', kw)
                self.assertIn('significant', kw)
                self.assertIn('interpretation', kw)
    
    def test_statistical_analysis_with_insufficient_models(self):
        """모델 수가 부족한 경우 통계 분석 테스트"""
        single_model_results = {'model_a': self.model_results['model_a']}
        
        result = self.comparator.compare_models(single_model_results)
        statistical_analysis = result['statistical_analysis']
        
        # 분석이 수행되지 않아야 함
        self.assertFalse(statistical_analysis['analysis_performed'])
        self.assertIn('reason', statistical_analysis)
        self.assertIn('at least 2 models', statistical_analysis['reason'])
    
    def test_generate_model_recommendations(self):
        """모델 권장사항 생성 테스트"""
        result = self.comparator.compare_models(self.model_results)
        recommendations = result['recommendations']
        
        # 권장사항이 생성되어야 함
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # 권장사항 내용 검증
        recommendations_text = ' '.join(recommendations)
        self.assertIn('전체 최고 성능', recommendations_text)
        self.assertIn('일관성 최고', recommendations_text)
        self.assertIn('성능 분포', recommendations_text)
        
        # 메트릭별 최고 모델 언급 확인
        metrics_mentioned = ['정확성', '명확성', '실행가능성', 'JSON 정확성']
        for metric in metrics_mentioned:
            self.assertIn(metric, recommendations_text)
    
    def test_classify_tier(self):
        """티어 분류 테스트"""
        test_cases = [
            (0.95, "Tier 1 (우수)"),
            (0.85, "Tier 1 (우수)"),
            (0.80, "Tier 2 (양호)"),
            (0.75, "Tier 2 (양호)"),
            (0.70, "Tier 3 (보통)"),
            (0.65, "Tier 3 (보통)"),
            (0.60, "Tier 4 (개선필요)"),
            (0.50, "Tier 4 (개선필요)")
        ]
        
        for score, expected_tier in test_cases:
            with self.subTest(score=score):
                tier = self.comparator._classify_tier(score)
                self.assertEqual(tier, expected_tier)
    
    def test_tier_distribution_calculation(self):
        """티어 분포 계산 테스트"""
        result = self.comparator.compare_models(self.model_results)
        comparison_table = result['comparison_table']
        
        # 티어 분포 검증
        self.assertIn('tier_distribution', comparison_table)
        tier_distribution = comparison_table['tier_distribution']
        
        # 모든 모델이 티어에 분류되어야 함
        total_models = sum(tier_distribution.values())
        self.assertEqual(total_models, 3)
        
        # 티어 이름 검증
        for tier in tier_distribution.keys():
            self.assertIn('Tier', tier)
    
    def test_comparison_summary_generation(self):
        """비교 요약 생성 테스트"""
        result = self.comparator.compare_models(self.model_results)
        comparison_table = result['comparison_table']
        
        # 요약 정보 검증
        self.assertIn('summary', comparison_table)
        summary = comparison_table['summary']
        
        self.assertIn('total_models', summary)
        self.assertIn('score_statistics', summary)
        self.assertIn('best_model', summary)
        self.assertIn('worst_model', summary)
        self.assertIn('score_gap', summary)
        
        # 통계 정보 검증
        score_stats = summary['score_statistics']
        self.assertIn('mean', score_stats)
        self.assertIn('std', score_stats)
        self.assertIn('min', score_stats)
        self.assertIn('max', score_stats)
        self.assertIn('median', score_stats)
        
        # 기본 값 검증
        self.assertEqual(summary['total_models'], 3)
        self.assertIsInstance(summary['best_model'], str)
        self.assertIsInstance(summary['worst_model'], str)
        self.assertGreaterEqual(summary['score_gap'], 0)
    
    def test_empty_model_results_handling(self):
        """빈 모델 결과 처리 테스트"""
        empty_model_results = {
            'model_a': [],
            'model_b': []
        }
        
        result = self.comparator.compare_models(empty_model_results)
        
        # 빈 결과에 대해서도 구조를 유지해야 함
        self.assertEqual(result['model_count'], 2)
        self.assertIn('model_a', result['model_statistics'])
        self.assertIn('model_b', result['model_statistics'])
        
        # 빈 결과는 0점으로 처리되어야 함
        for model_name in ['model_a', 'model_b']:
            model_stats = result['model_statistics'][model_name]
            self.assertEqual(model_stats['overall']['weighted_score'], 0.0)
            self.assertEqual(model_stats['overall']['grade'], 'F')
    
    def test_single_model_comparison(self):
        """단일 모델 비교 테스트"""
        single_model_results = {'model_a': self.model_results['model_a']}
        
        result = self.comparator.compare_models(single_model_results)
        
        # 단일 모델도 처리 가능해야 함
        self.assertEqual(result['model_count'], 1)
        self.assertIn('model_a', result['model_statistics'])
        
        # 순위는 1위가 되어야 함
        rankings = result['rankings']
        self.assertEqual(rankings['overall_ranking']['model_a'], 1)
        
        # 통계 분석은 수행되지 않아야 함
        self.assertFalse(result['statistical_analysis']['analysis_performed'])


if __name__ == "__main__":
    unittest.main()