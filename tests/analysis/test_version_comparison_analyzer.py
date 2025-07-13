"""버전 비교 분석기 테스트"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime
from selvage_eval.analysis.version_comparison_analyzer import VersionComparisonAnalyzer
from selvage_eval.analysis.deepeval_log_parser import TestCaseResult, MetricScore


class TestVersionComparisonAnalyzer(unittest.TestCase):
    """버전 비교 분석기 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.analyzer = VersionComparisonAnalyzer()
        
        # 버전별 샘플 데이터
        self.sample_version_data = {
            '0.1.0': {
                'version': '0.1.0',
                'sessions': [
                    {
                        'session_dir': '/path/to/session1',
                        'results': [
                            TestCaseResult(
                                correctness=MetricScore(0.70, False, "Basic analysis"),
                                clarity=MetricScore(0.65, False, "Unclear"),
                                actionability=MetricScore(0.60, False, "Vague"),
                                json_correctness=MetricScore(0.80, True, "Good JSON"),
                                input_data='{"test": "v0.1.0"}',
                                actual_output='{"result": "v0.1.0"}',
                                raw_content="v0.1.0 test content"
                            )
                        ],
                        'execution_date': datetime(2023, 1, 1)
                    }
                ],
                'execution_dates': [datetime(2023, 1, 1)],
                'latest_execution_date': datetime(2023, 1, 1)
            },
            '0.2.0': {
                'version': '0.2.0',
                'sessions': [
                    {
                        'session_dir': '/path/to/session2',
                        'results': [
                            TestCaseResult(
                                correctness=MetricScore(0.80, True, "Better analysis"),
                                clarity=MetricScore(0.75, True, "Clearer"),
                                actionability=MetricScore(0.70, False, "Somewhat actionable"),
                                json_correctness=MetricScore(0.85, True, "Better JSON"),
                                input_data='{"test": "v0.2.0"}',
                                actual_output='{"result": "v0.2.0"}',
                                raw_content="v0.2.0 test content"
                            )
                        ],
                        'execution_date': datetime(2023, 2, 1)
                    }
                ],
                'execution_dates': [datetime(2023, 2, 1)],
                'latest_execution_date': datetime(2023, 2, 1)
            },
            '0.3.0': {
                'version': '0.3.0',
                'sessions': [
                    {
                        'session_dir': '/path/to/session3',
                        'results': [
                            TestCaseResult(
                                correctness=MetricScore(0.90, True, "Excellent analysis"),
                                clarity=MetricScore(0.85, True, "Very clear"),
                                actionability=MetricScore(0.80, True, "Highly actionable"),
                                json_correctness=MetricScore(0.95, True, "Perfect JSON"),
                                input_data='{"test": "v0.3.0"}',
                                actual_output='{"result": "v0.3.0"}',
                                raw_content="v0.3.0 test content"
                            )
                        ],
                        'execution_date': datetime(2023, 3, 1)
                    }
                ],
                'execution_dates': [datetime(2023, 3, 1)],
                'latest_execution_date': datetime(2023, 3, 1)
            }
        }
    
    def test_init(self):
        """초기화 테스트"""
        self.assertIsNotNone(self.analyzer.version_pattern)
        self.assertIsNotNone(self.analyzer.parser)
        self.assertIsNotNone(self.analyzer.aggregator)
        
        # 임계값 확인
        self.assertEqual(self.analyzer.regression_threshold, 0.05)
        self.assertEqual(self.analyzer.improvement_threshold, 0.03)
        self.assertEqual(self.analyzer.excellent_threshold, 0.8)
        self.assertEqual(self.analyzer.needs_improvement_threshold, 0.7)
    
    def test_extract_version_from_metadata(self):
        """메타데이터에서 버전 추출 테스트"""
        # 정상적인 버전 정보
        metadata1 = {'selvage_version': 'selvage 1.2.3'}
        result1 = self.analyzer._extract_version_from_metadata(metadata1)
        self.assertEqual(result1, '1.2.3')
        
        # 다른 필드명
        metadata2 = {'tool_version': 'selvage 2.0.1'}
        result2 = self.analyzer._extract_version_from_metadata(metadata2)
        self.assertEqual(result2, '2.0.1')
        
        # 명령어에서 추출
        metadata3 = {'command': 'run selvage 0.5.0 --config test.yaml'}
        result3 = self.analyzer._extract_version_from_metadata(metadata3)
        self.assertEqual(result3, '0.5.0')
        
        # 버전 정보 없음
        metadata4 = {'other_field': 'no version info'}
        result4 = self.analyzer._extract_version_from_metadata(metadata4)
        self.assertIsNone(result4)
    
    def test_extract_execution_date(self):
        """실행 날짜 추출 테스트"""
        # ISO 형식
        metadata1 = {'execution_date': '2023-01-01T10:00:00Z'}
        result1 = self.analyzer._extract_execution_date(metadata1)
        self.assertIsInstance(result1, datetime)
        
        # 다른 형식
        metadata2 = {'timestamp': '2023-01-01 10:00:00'}
        result2 = self.analyzer._extract_execution_date(metadata2)
        self.assertIsInstance(result2, datetime)
        
        # 날짜 정보 없음
        metadata3 = {'other_field': 'no date info'}
        result3 = self.analyzer._extract_execution_date(metadata3)
        self.assertIsNone(result3)
    
    def test_sort_versions_by_date(self):
        """날짜별 버전 정렬 테스트"""
        version_performance = {
            '0.2.0': {
                'version': '0.2.0',
                'latest_execution_date': datetime(2023, 2, 1),
                'performance': {}
            },
            '0.1.0': {
                'version': '0.1.0',
                'latest_execution_date': datetime(2023, 1, 1),
                'performance': {}
            },
            '0.3.0': {
                'version': '0.3.0',
                'latest_execution_date': datetime(2023, 3, 1),
                'performance': {}
            }
        }
        
        sorted_versions = self.analyzer._sort_versions_by_date(version_performance)
        
        # 날짜 순으로 정렬되어야 함
        self.assertEqual(len(sorted_versions), 3)
        self.assertEqual(sorted_versions[0]['version'], '0.1.0')
        self.assertEqual(sorted_versions[1]['version'], '0.2.0')
        self.assertEqual(sorted_versions[2]['version'], '0.3.0')
    
    def test_calculate_overall_trend(self):
        """전체 트렌드 계산 테스트"""
        trends = {
            'correctness': {'slope': 0.05, 'r_squared': 0.9},
            'clarity': {'slope': 0.03, 'r_squared': 0.8},
            'actionability': {'slope': 0.02, 'r_squared': 0.7},
            'json_correctness': {'slope': 0.01, 'r_squared': 0.6}
        }
        
        overall_trend = self.analyzer._calculate_overall_trend(trends)
        
        self.assertIn('direction', overall_trend)
        self.assertIn('strength', overall_trend)
        self.assertIn('confidence', overall_trend)
        self.assertEqual(overall_trend['direction'], 'improving')
        self.assertGreater(overall_trend['strength'], 0)
        self.assertGreater(overall_trend['confidence'], 0)
    
    def test_assess_stability_stable(self):
        """안정성 평가 테스트 - 안정적인 경우"""
        regressions = []  # 회귀 없음
        
        stability = self.analyzer._assess_stability(regressions)
        
        self.assertEqual(stability['stability_level'], 'stable')
        self.assertIn('No significant regressions', stability['description'])
    
    def test_assess_stability_unstable(self):
        """안정성 평가 테스트 - 불안정한 경우"""
        regressions = [
            {'severity': 'critical'},
            {'severity': 'major'}
        ]
        
        stability = self.analyzer._assess_stability(regressions)
        
        self.assertEqual(stability['stability_level'], 'unstable')
        self.assertIn('Critical regressions', stability['description'])
    
    def test_analyze_affected_metrics(self):
        """영향받은 메트릭 분석 테스트"""
        previous_version = {
            'performance': {
                'correctness': {'mean_score': 0.8},
                'clarity': {'mean_score': 0.7},
                'actionability': {'mean_score': 0.6},
                'json_correctness': {'mean_score': 0.9}
            }
        }
        
        current_version = {
            'performance': {
                'correctness': {'mean_score': 0.85},  # 6.25% 개선
                'clarity': {'mean_score': 0.65},      # 7.14% 회귀
                'actionability': {'mean_score': 0.61}, # 1.67% 개선 (임계값 미만)
                'json_correctness': {'mean_score': 0.88} # 2.22% 회귀 (임계값 미만)
            }
        }
        
        affected_metrics = self.analyzer._analyze_affected_metrics(previous_version, current_version)
        
        # 임계값(3%) 이상 변화한 메트릭만 포함되어야 함
        self.assertEqual(len(affected_metrics), 2)
        
        # correctness는 개선, clarity는 회귀
        metric_names = [m['metric'] for m in affected_metrics]
        self.assertIn('correctness', metric_names)
        self.assertIn('clarity', metric_names)
        
        # 변화 타입 확인
        correctness_metric = next(m for m in affected_metrics if m['metric'] == 'correctness')
        clarity_metric = next(m for m in affected_metrics if m['metric'] == 'clarity')
        
        self.assertEqual(correctness_metric['change_type'], 'improvement')
        self.assertEqual(clarity_metric['change_type'], 'regression')
    
    @patch('selvage_eval.analysis.version_comparison_analyzer.Path')
    def test_collect_version_data_empty_path(self, mock_path):
        """존재하지 않는 경로에 대한 데이터 수집 테스트"""
        mock_path.return_value.exists.return_value = False
        
        result = self.analyzer.collect_version_data('/nonexistent/path')
        
        self.assertEqual(result, {})
    
    def test_analyze_version_progression_empty_data(self):
        """빈 데이터에 대한 버전 발전 분석 테스트"""
        result = self.analyzer.analyze_version_progression({})
        
        self.assertEqual(result['version_timeline'], [])
        self.assertEqual(result['performance_trends'], {})
        self.assertEqual(result['regression_analysis'], {})
        self.assertEqual(result['improvement_highlights'], [])
        self.assertEqual(result['version_recommendations'], {})
    
    def test_analyze_version_progression_with_data(self):
        """데이터가 있는 경우 버전 발전 분석 테스트"""
        # Mock aggregator to return predictable performance data
        with patch.object(self.analyzer.aggregator, 'aggregate_model_performance') as mock_aggregate:
            mock_aggregate.side_effect = [
                {'overall': {'weighted_score': 0.70, 'grade': 'C'}},
                {'overall': {'weighted_score': 0.80, 'grade': 'B'}},
                {'overall': {'weighted_score': 0.85, 'grade': 'A'}}
            ]
            
            result = self.analyzer.analyze_version_progression(self.sample_version_data)
            
            # 기본 구조 확인
            self.assertIn('version_timeline', result)
            self.assertIn('performance_trends', result)
            self.assertIn('regression_analysis', result)
            self.assertIn('improvement_highlights', result)
            self.assertIn('version_recommendations', result)
            
            # 타임라인 확인
            timeline = result['version_timeline']
            self.assertEqual(len(timeline), 3)
            
            # 시간순 정렬 확인
            self.assertEqual(timeline[0]['version'], '0.1.0')
            self.assertEqual(timeline[1]['version'], '0.2.0')
            self.assertEqual(timeline[2]['version'], '0.3.0')
    
    def test_detect_regressions_no_data(self):
        """데이터가 부족한 경우 회귀 탐지 테스트"""
        single_version = [{'version': '0.1.0', 'performance': {'overall': {'weighted_score': 0.8}}}]
        
        result = self.analyzer._detect_regressions(single_version)
        
        self.assertFalse(result['regressions_detected'])
        self.assertIn('Need at least 2 versions', result['reason'])
    
    def test_detect_regressions_with_regression(self):
        """회귀가 있는 경우 회귀 탐지 테스트"""
        versions = [
            {'version': '0.1.0', 'performance': {'overall': {'weighted_score': 0.8}}},
            {'version': '0.2.0', 'performance': {'overall': {'weighted_score': 0.7}}}  # 12.5% 회귀
        ]
        
        with patch.object(self.analyzer, '_analyze_affected_metrics') as mock_analyze:
            mock_analyze.return_value = []
            
            result = self.analyzer._detect_regressions(versions)
            
            self.assertTrue(result['regressions_detected'])
            self.assertEqual(result['total_regressions'], 1)
            
            regression = result['regressions'][0]
            self.assertEqual(regression['from_version'], '0.1.0')
            self.assertEqual(regression['to_version'], '0.2.0')
            self.assertEqual(regression['severity'], 'major')  # 10-15% 범위
    
    def test_identify_improvements_with_improvement(self):
        """개선이 있는 경우 개선사항 식별 테스트"""
        versions = [
            {'version': '0.1.0', 'performance': {'overall': {'weighted_score': 0.7}}},
            {'version': '0.2.0', 'performance': {'overall': {'weighted_score': 0.8}}}  # 14.3% 개선
        ]
        
        with patch.object(self.analyzer, '_analyze_improved_metrics') as mock_analyze:
            mock_analyze.return_value = []
            
            result = self.analyzer._identify_improvements(versions)
            
            self.assertEqual(len(result), 1)
            
            improvement = result[0]
            self.assertEqual(improvement['from_version'], '0.1.0')
            self.assertEqual(improvement['to_version'], '0.2.0')
            self.assertEqual(improvement['improvement_level'], 'significant')  # 8-15% 범위
    
    def test_generate_version_recommendations_empty_data(self):
        """빈 데이터에 대한 버전 권장사항 생성 테스트"""
        result = self.analyzer._generate_version_recommendations([], {})
        
        self.assertIsNone(result['recommended_version'])
        self.assertIn('분석할 버전 데이터가 없습니다', result['recommendations'][0])
    
    def test_generate_version_recommendations_with_data(self):
        """데이터가 있는 경우 버전 권장사항 생성 테스트"""
        versions = [
            {
                'version': '0.1.0',
                'performance': {'overall': {'weighted_score': 0.7, 'grade': 'C'}}
            },
            {
                'version': '0.2.0',
                'performance': {'overall': {'weighted_score': 0.85, 'grade': 'A'}}
            }
        ]
        
        analysis = {
            'regression_analysis': {
                'stability_assessment': {'stability_level': 'stable'}
            },
            'performance_trends': {
                'analysis_performed': True,
                'overall_trend': {'direction': 'improving'}
            },
            'improvement_highlights': [
                {'improvement_level': 'significant'}
            ]
        }
        
        result = self.analyzer._generate_version_recommendations(versions, analysis)
        
        self.assertEqual(result['recommended_version'], '0.2.0')
        self.assertEqual(result['best_performance_version'], '0.2.0')
        self.assertEqual(result['latest_version'], '0.2.0')
        self.assertIsInstance(result['recommendations'], list)
        self.assertGreater(len(result['recommendations']), 0)
        
        # 권장사항 내용 확인
        recommendations_text = ' '.join(result['recommendations'])
        self.assertIn('최고 성능 버전: 0.2.0', recommendations_text)
        self.assertIn('안정성 우수', recommendations_text)
        self.assertIn('성능 개선 트렌드', recommendations_text)


if __name__ == "__main__":
    unittest.main()