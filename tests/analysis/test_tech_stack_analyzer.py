"""기술스택 분석기 테스트"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from selvage_eval.analysis.tech_stack_analyzer import TechStackAnalyzer
from selvage_eval.analysis.deepeval_log_parser import TestCaseResult, MetricScore


class TestTechStackAnalyzer(unittest.TestCase):
    """기술스택 분석기 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.analyzer = TechStackAnalyzer()
        
        # 기술스택별 샘플 데이터 생성
        self.repo_results = {
            'cline': {  # TypeScript/JavaScript
                'gpt-4': [
                    TestCaseResult(
                        correctness=MetricScore(0.90, True, "Excellent TypeScript analysis"),
                        clarity=MetricScore(0.85, True, "Clear explanation"),
                        actionability=MetricScore(0.80, True, "Good suggestions"),
                        json_correctness=MetricScore(1.0, True, "Perfect JSON"),
                        input_data='{"typescript": "code"}',
                        actual_output='{"result": "typescript_output"}',
                        raw_content="typescript test content"
                    )
                ],
                'claude-3': [
                    TestCaseResult(
                        correctness=MetricScore(0.85, True, "Good TypeScript analysis"),
                        clarity=MetricScore(0.80, True, "Clear"),
                        actionability=MetricScore(0.75, True, "Actionable"),
                        json_correctness=MetricScore(0.95, True, "Good JSON"),
                        input_data='{"typescript": "code2"}',
                        actual_output='{"result": "typescript_output2"}',
                        raw_content="typescript test content 2"
                    )
                ]
            },
            'ecommerce-microservices': {  # Java/Spring
                'gpt-4': [
                    TestCaseResult(
                        correctness=MetricScore(0.80, True, "Good Java analysis"),
                        clarity=MetricScore(0.75, True, "Clear enough"),
                        actionability=MetricScore(0.70, False, "Needs improvement"),
                        json_correctness=MetricScore(0.90, True, "Good JSON"),
                        input_data='{"java": "code"}',
                        actual_output='{"result": "java_output"}',
                        raw_content="java test content"
                    )
                ],
                'claude-3': [
                    TestCaseResult(
                        correctness=MetricScore(0.75, True, "Decent Java analysis"),
                        clarity=MetricScore(0.70, False, "Unclear"),
                        actionability=MetricScore(0.65, False, "Vague"),
                        json_correctness=MetricScore(0.85, True, "Acceptable JSON"),
                        input_data='{"java": "code2"}',
                        actual_output='{"result": "java_output2"}',
                        raw_content="java test content 2"
                    )
                ]
            },
            'kotlin-realworld': {  # Kotlin/JPA
                'gpt-4': [
                    TestCaseResult(
                        correctness=MetricScore(0.70, False, "Poor Kotlin analysis"),
                        clarity=MetricScore(0.65, False, "Unclear"),
                        actionability=MetricScore(0.60, False, "Not actionable"),
                        json_correctness=MetricScore(0.80, True, "Acceptable JSON"),
                        input_data='{"kotlin": "code"}',
                        actual_output='{"result": "kotlin_output"}',
                        raw_content="kotlin test content"
                    )
                ]
            }
        }
    
    def test_init(self):
        """초기화 테스트"""
        self.assertIsInstance(self.analyzer.tech_stack_mapping, dict)
        self.assertIn('cline', self.analyzer.tech_stack_mapping)
        self.assertIn('ecommerce-microservices', self.analyzer.tech_stack_mapping)
        self.assertIn('kotlin-realworld', self.analyzer.tech_stack_mapping)
        self.assertIn('selvage-deprecated', self.analyzer.tech_stack_mapping)
        
        # 매핑 내용 확인
        self.assertEqual(self.analyzer.tech_stack_mapping['cline'], 'TypeScript/JavaScript')
        self.assertEqual(self.analyzer.tech_stack_mapping['ecommerce-microservices'], 'Java/Spring')
        self.assertEqual(self.analyzer.tech_stack_mapping['kotlin-realworld'], 'Kotlin/JPA')
        self.assertEqual(self.analyzer.tech_stack_mapping['selvage-deprecated'], 'Python')
    
    def test_analyze_tech_stack_performance_empty_input(self):
        """빈 입력에 대한 테스트"""
        result = self.analyzer.analyze_tech_stack_performance({})
        
        self.assertEqual(result['by_tech_stack'], {})
        self.assertEqual(result['cross_stack_comparison'], {})
        self.assertEqual(result['recommendations'], [])
    
    def test_analyze_tech_stack_performance_basic_functionality(self):
        """기본 기술스택 분석 기능 테스트"""
        result = self.analyzer.analyze_tech_stack_performance(self.repo_results)
        
        # 기본 구조 검증
        self.assertIn('by_tech_stack', result)
        self.assertIn('cross_stack_comparison', result)
        self.assertIn('recommendations', result)
        
        # 기술스택별 분석 결과 검증
        by_tech_stack = result['by_tech_stack']
        
        # 매핑된 기술스택 이름 확인
        self.assertIn('TypeScript/JavaScript', by_tech_stack)
        self.assertIn('Java/Spring', by_tech_stack)
        self.assertIn('Kotlin/JPA', by_tech_stack)
        
        # 각 기술스택 분석 결과 구조 검증
        for tech_stack, analysis in by_tech_stack.items():
            self.assertIn('repository', analysis)
            self.assertIn('model_performance', analysis)
            self.assertIn('best_model', analysis)
            self.assertIn('performance_gap', analysis)
            self.assertIn('recommendations', analysis)
            self.assertIn('model_count', analysis)
    
    def test_find_best_model_empty_input(self):
        """빈 입력에 대한 최고 모델 찾기 테스트"""
        result = self.analyzer._find_best_model({})
        
        self.assertIsNone(result['name'])
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['grade'], 'F')
        self.assertEqual(result['metrics'], {})
    
    def test_find_best_model_with_data(self):
        """데이터가 있는 경우 최고 모델 찾기 테스트"""
        model_performance = {
            'model_a': {
                'overall': {'weighted_score': 0.85, 'grade': 'A'},
                'correctness': {'mean_score': 0.90},
                'clarity': {'mean_score': 0.80},
                'actionability': {'mean_score': 0.85},
                'json_correctness': {'mean_score': 1.0}
            },
            'model_b': {
                'overall': {'weighted_score': 0.75, 'grade': 'B'},
                'correctness': {'mean_score': 0.80},
                'clarity': {'mean_score': 0.70},
                'actionability': {'mean_score': 0.75},
                'json_correctness': {'mean_score': 0.90}
            }
        }
        
        result = self.analyzer._find_best_model(model_performance)
        
        self.assertEqual(result['name'], 'model_a')
        self.assertEqual(result['score'], 0.85)
        self.assertEqual(result['grade'], 'A')
        self.assertIn('correctness', result['metrics'])
        self.assertIn('clarity', result['metrics'])
        self.assertIn('actionability', result['metrics'])
        self.assertIn('json_correctness', result['metrics'])
    
    def test_calculate_performance_gap_empty_input(self):
        """빈 입력에 대한 성능 격차 계산 테스트"""
        result = self.analyzer._calculate_performance_gap({})
        
        self.assertEqual(result['max_score'], 0.0)
        self.assertEqual(result['min_score'], 0.0)
        self.assertEqual(result['gap'], 0.0)
        self.assertEqual(result['coefficient_of_variation'], 0.0)
    
    def test_calculate_performance_gap_with_data(self):
        """데이터가 있는 경우 성능 격차 계산 테스트"""
        model_performance = {
            'model_a': {'overall': {'weighted_score': 0.90}},
            'model_b': {'overall': {'weighted_score': 0.70}},
            'model_c': {'overall': {'weighted_score': 0.80}}
        }
        
        result = self.analyzer._calculate_performance_gap(model_performance)
        
        self.assertEqual(result['max_score'], 0.90)
        self.assertEqual(result['min_score'], 0.70)
        self.assertAlmostEqual(result['gap'], 0.20, places=5)
        self.assertGreater(result['coefficient_of_variation'], 0.0)
    
    def test_generate_tech_stack_recommendations(self):
        """기술스택별 권장사항 생성 테스트"""
        performance = {
            'best_model': {
                'name': 'gpt-4',
                'score': 0.85,
                'grade': 'A',
                'metrics': {
                    'correctness': 0.90,
                    'clarity': 0.80,
                    'actionability': 0.85,
                    'json_correctness': 1.0
                }
            },
            'performance_gap': {
                'gap': 0.15,
                'coefficient_of_variation': 0.12
            },
            'model_performance': {}
        }
        
        recommendations = self.analyzer._generate_tech_stack_recommendations(
            'TypeScript/JavaScript', performance
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # 권장사항 내용 확인
        recommendations_text = ' '.join(recommendations)
        self.assertIn('TypeScript/JavaScript', recommendations_text)
        self.assertIn('gpt-4', recommendations_text)
        self.assertIn('0.85', recommendations_text)
    
    def test_cross_stack_comparison_insufficient_data(self):
        """데이터가 부족한 경우 교차 비교 테스트"""
        single_stack_analysis = {
            'TypeScript/JavaScript': {
                'best_model': {'name': 'gpt-4', 'score': 0.85}
            }
        }
        
        result = self.analyzer._cross_stack_comparison(single_stack_analysis)
        
        self.assertFalse(result['comparison_performed'])
        self.assertIn('reason', result)
        self.assertIn('at least 2 tech stacks', result['reason'])
    
    def test_cross_stack_comparison_with_data(self):
        """데이터가 있는 경우 교차 비교 테스트"""
        tech_stack_analysis = {
            'TypeScript/JavaScript': {
                'best_model': {
                    'name': 'gpt-4',
                    'score': 0.85,
                    'metrics': {
                        'correctness': 0.90,
                        'clarity': 0.80,
                        'actionability': 0.85,
                        'json_correctness': 1.0
                    }
                }
            },
            'Java/Spring': {
                'best_model': {
                    'name': 'claude-3',
                    'score': 0.75,
                    'metrics': {
                        'correctness': 0.80,
                        'clarity': 0.70,
                        'actionability': 0.75,
                        'json_correctness': 0.90
                    }
                }
            }
        }
        
        result = self.analyzer._cross_stack_comparison(tech_stack_analysis)
        
        self.assertTrue(result['comparison_performed'])
        self.assertIn('stack_ranking', result)
        self.assertIn('metric_comparison', result)
        self.assertIn('performance_summary', result)
        
        # 순위 검증
        stack_ranking = result['stack_ranking']
        self.assertEqual(len(stack_ranking), 2)
        self.assertEqual(stack_ranking[0]['tech_stack'], 'TypeScript/JavaScript')
        self.assertEqual(stack_ranking[1]['tech_stack'], 'Java/Spring')
        
        # 성능 요약 검증
        performance_summary = result['performance_summary']
        self.assertEqual(performance_summary['best_stack'], 'TypeScript/JavaScript')
        self.assertEqual(performance_summary['worst_stack'], 'Java/Spring')
    
    def test_compare_metrics_across_stacks(self):
        """기술스택 간 메트릭 비교 테스트"""
        tech_stack_analysis = {
            'TypeScript/JavaScript': {
                'best_model': {
                    'metrics': {
                        'correctness': 0.90,
                        'clarity': 0.80,
                        'actionability': 0.85,
                        'json_correctness': 1.0
                    }
                }
            },
            'Java/Spring': {
                'best_model': {
                    'metrics': {
                        'correctness': 0.80,
                        'clarity': 0.70,
                        'actionability': 0.75,
                        'json_correctness': 0.90
                    }
                }
            }
        }
        
        result = self.analyzer._compare_metrics_across_stacks(tech_stack_analysis)
        
        # 모든 메트릭이 비교되어야 함
        for metric in ['correctness', 'clarity', 'actionability', 'json_correctness']:
            self.assertIn(metric, result)
            
            metric_result = result[metric]
            self.assertIn('rankings', metric_result)
            self.assertIn('best_stack', metric_result)
            self.assertIn('worst_stack', metric_result)
            self.assertIn('score_statistics', metric_result)
            
            # 순위 검증
            rankings = metric_result['rankings']
            self.assertEqual(len(rankings), 2)
            
            # 통계 검증
            stats = metric_result['score_statistics']
            self.assertIn('mean', stats)
            self.assertIn('std', stats)
            self.assertIn('max', stats)
            self.assertIn('min', stats)
    
    def test_generate_overall_recommendations_empty_input(self):
        """빈 입력에 대한 전체 권장사항 생성 테스트"""
        result = self.analyzer._generate_overall_recommendations({}, {})
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn('없습니다', result[0])
    
    def test_generate_overall_recommendations_with_data(self):
        """데이터가 있는 경우 전체 권장사항 생성 테스트"""
        tech_stack_analysis = {
            'TypeScript/JavaScript': {
                'recommendations': ['TypeScript 관련 권장사항']
            },
            'Java/Spring': {
                'recommendations': ['Java 관련 권장사항']
            }
        }
        
        cross_stack_comparison = {
            'comparison_performed': True,
            'performance_summary': {
                'best_stack': 'TypeScript/JavaScript',
                'worst_stack': 'Java/Spring',
                'score_range': {'gap': 0.15}
            },
            'metric_comparison': {
                'correctness': {'best_stack': 'TypeScript/JavaScript'}
            }
        }
        
        result = self.analyzer._generate_overall_recommendations(
            tech_stack_analysis, cross_stack_comparison
        )
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # 권장사항 내용 확인
        recommendations_text = ' '.join(result)
        self.assertIn('2개', recommendations_text)  # 분석된 기술스택 수
        self.assertIn('TypeScript/JavaScript', recommendations_text)
        self.assertIn('Java/Spring', recommendations_text)
    
    def test_full_analysis_integration(self):
        """전체 분석 통합 테스트"""
        result = self.analyzer.analyze_tech_stack_performance(self.repo_results)
        
        # 전체 구조 검증
        self.assertIn('by_tech_stack', result)
        self.assertIn('cross_stack_comparison', result)
        self.assertIn('recommendations', result)
        
        # 기술스택별 분석 검증
        by_tech_stack = result['by_tech_stack']
        self.assertGreater(len(by_tech_stack), 0)
        
        # 교차 비교 검증
        cross_comparison = result['cross_stack_comparison']
        if cross_comparison.get('comparison_performed'):
            self.assertIn('stack_ranking', cross_comparison)
            self.assertIn('performance_summary', cross_comparison)
        
        # 권장사항 검증
        recommendations = result['recommendations']
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
    
    def test_unknown_repository_handling(self):
        """알 수 없는 저장소 처리 테스트"""
        unknown_repo_results = {
            'unknown-repo': {
                'model_a': [
                    TestCaseResult(
                        correctness=MetricScore(0.80, True, "Test analysis"),
                        clarity=MetricScore(0.75, True, "Clear"),
                        actionability=MetricScore(0.70, False, "Needs improvement"),
                        json_correctness=MetricScore(0.90, True, "Good JSON"),
                        input_data='{"unknown": "code"}',
                        actual_output='{"result": "unknown_output"}',
                        raw_content="unknown test content"
                    )
                ]
            }
        }
        
        result = self.analyzer.analyze_tech_stack_performance(unknown_repo_results)
        
        # 알 수 없는 저장소도 저장소명을 그대로 사용해야 함
        self.assertIn('unknown-repo', result['by_tech_stack'])
        
        # 분석 결과 구조는 동일해야 함
        analysis = result['by_tech_stack']['unknown-repo']
        self.assertEqual(analysis['repository'], 'unknown-repo')
        self.assertIn('model_performance', analysis)
        self.assertIn('best_model', analysis)


if __name__ == "__main__":
    unittest.main()