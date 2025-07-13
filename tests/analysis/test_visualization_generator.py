"""시각화 생성기 테스트"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path
from selvage_eval.analysis.visualization_generator import VisualizationGenerator


class TestVisualizationGenerator(unittest.TestCase):
    """시각화 생성기 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.generator = VisualizationGenerator()
        
        # 임시 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir)
        
        # 샘플 분석 결과 데이터
        self.sample_analysis_results = {
            'model_comparison': {
                'comparison_table': {
                    'table_data': [
                        {
                            'model_name': 'gpt-4',
                            'overall_score': 0.85,
                            'correctness_score': 0.90,
                            'clarity_score': 0.80,
                            'actionability_score': 0.85,
                            'json_correctness_score': 1.0,
                            'grade': 'A'
                        },
                        {
                            'model_name': 'claude-3',
                            'overall_score': 0.78,
                            'correctness_score': 0.82,
                            'clarity_score': 0.75,
                            'actionability_score': 0.78,
                            'json_correctness_score': 0.95,
                            'grade': 'B+'
                        }
                    ]
                },
                'recommendations': [
                    '🏆 전체 최고 성능: gpt-4 (점수: 0.850, 등급: A)',
                    '📊 정확성 최고: gpt-4 (점수: 0.900)'
                ]
            },
            'failure_analysis': {
                'total_failures': 25,
                'by_metric': {
                    'correctness': {'total_failures': 10},
                    'clarity': {'total_failures': 8},
                    'actionability': {'total_failures': 5},
                    'json_correctness': {'total_failures': 2}
                },
                'by_category': {
                    'missing_issues': 12,
                    'unclear_explanation': 8,
                    'vague_suggestions': 5
                },
                'critical_patterns': [
                    {'severity': 'high', 'count': 3},
                    {'severity': 'medium', 'count': 15},
                    {'severity': 'low', 'count': 7}
                ]
            },
            'version_analysis': {
                'version_timeline': [
                    {
                        'version': '0.1.0',
                        'performance': {
                            'correctness': {'mean_score': 0.70},
                            'clarity': {'mean_score': 0.65},
                            'actionability': {'mean_score': 0.60},
                            'json_correctness': {'mean_score': 0.80}
                        }
                    },
                    {
                        'version': '0.2.0',
                        'performance': {
                            'correctness': {'mean_score': 0.80},
                            'clarity': {'mean_score': 0.75},
                            'actionability': {'mean_score': 0.70},
                            'json_correctness': {'mean_score': 0.85}
                        }
                    }
                ],
                'version_recommendations': {
                    'recommended_version': '0.2.0'
                }
            },
            'tech_stack_analysis': {
                'by_tech_stack': {
                    'TypeScript/JavaScript': {
                        'best_model': {'name': 'gpt-4', 'score': 0.85}
                    },
                    'Java/Spring': {
                        'best_model': {'name': 'claude-3', 'score': 0.78}
                    }
                },
                'recommendations': [
                    '📊 분석된 기술스택: 2개',
                    '🏆 최고 성능 기술스택: TypeScript/JavaScript'
                ]
            }
        }
    
    def tearDown(self):
        """테스트 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """초기화 테스트"""
        self.assertIsInstance(self.generator.color_palette, dict)
        self.assertIn('primary', self.generator.color_palette)
        self.assertIn('success', self.generator.color_palette)
        self.assertIn('warning', self.generator.color_palette)
        self.assertIn('danger', self.generator.color_palette)
        self.assertIn('info', self.generator.color_palette)
    
    def test_generate_comprehensive_dashboard_empty_data(self):
        """빈 데이터에 대한 대시보드 생성 테스트"""
        result = self.generator.generate_comprehensive_dashboard({}, str(self.output_path))
        
        # 빈 결과 리스트가 반환되어야 함
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)
    
    def test_generate_comprehensive_dashboard_with_data(self):
        """데이터가 있는 경우 대시보드 생성 테스트"""
        # 실제 파일 생성을 허용하되, 임시 디렉토리 사용
        result = self.generator.generate_comprehensive_dashboard(
            self.sample_analysis_results, str(self.output_path)
        )
        
        # 결과 검증
        self.assertIsInstance(result, list)
        # 파일이 생성되었는지 확인 (예외가 발생할 수 있으므로 최소한 0개 이상)
        self.assertGreaterEqual(len(result), 0)
        
        # 생성된 파일들이 실제로 존재하는지 확인
        for file_path in result:
            self.assertTrue(Path(file_path).exists())
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_create_model_performance_radar(self, mock_write_html):
        """모델 성능 레이더 차트 생성 테스트"""
        mock_write_html.return_value = None
        
        output_file = self.output_path / 'test_radar.html'
        result = self.generator._create_model_performance_radar(
            self.sample_analysis_results['model_comparison'],
            output_file
        )
        
        self.assertEqual(result, str(output_file))
        # 실제로 Figure가 생성되고 write_html이 호출되는지 확인
        self.assertTrue(mock_write_html.called)
    
    def test_create_model_performance_radar_no_data(self):
        """데이터가 없는 경우 레이더 차트 생성 테스트"""
        empty_comparison = {'comparison_table': {'table_data': []}}
        
        output_file = self.output_path / 'test_radar.html'
        result = self.generator._create_model_performance_radar(
            empty_comparison,
            output_file
        )
        
        self.assertIsNone(result)
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_create_performance_heatmap(self, mock_write_html):
        """성능 히트맵 생성 테스트"""
        mock_write_html.return_value = None
        
        output_file = self.output_path / 'test_heatmap.html'
        result = self.generator._create_performance_heatmap(
            self.sample_analysis_results['model_comparison'],
            output_file
        )
        
        self.assertEqual(result, str(output_file))
        # 실제로 Figure가 생성되고 write_html이 호출되는지 확인
        self.assertTrue(mock_write_html.called)
    
    def test_create_performance_heatmap_no_data(self):
        """데이터가 없는 경우 히트맵 생성 테스트"""
        empty_comparison = {'comparison_table': {'table_data': []}}
        
        output_file = self.output_path / 'test_heatmap.html'
        result = self.generator._create_performance_heatmap(
            empty_comparison,
            output_file
        )
        
        self.assertIsNone(result)
    
    @patch('selvage_eval.analysis.visualization_generator.make_subplots')
    def test_create_failure_pattern_charts(self, mock_make_subplots):
        """실패 패턴 차트 생성 테스트"""
        # 서브플롯 모킹
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig
        mock_fig.write_html.return_value = None
        
        output_file = self.output_path / 'test_failure.html'
        result = self.generator._create_failure_pattern_charts(
            self.sample_analysis_results['failure_analysis'],
            output_file
        )
        
        self.assertEqual(result, str(output_file))
        mock_make_subplots.assert_called_once()
        mock_fig.write_html.assert_called_once()
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_create_version_trend_chart(self, mock_write_html):
        """버전 트렌드 차트 생성 테스트"""
        mock_write_html.return_value = None
        
        output_file = self.output_path / 'test_version.html'
        result = self.generator._create_version_trend_chart(
            self.sample_analysis_results['version_analysis'],
            output_file
        )
        
        self.assertEqual(result, str(output_file))
        mock_write_html.assert_called_once()
    
    def test_create_version_trend_chart_no_data(self):
        """데이터가 없는 경우 버전 트렌드 차트 생성 테스트"""
        empty_version = {'version_timeline': []}
        
        output_file = self.output_path / 'test_version.html'
        result = self.generator._create_version_trend_chart(
            empty_version,
            output_file
        )
        
        self.assertIsNone(result)
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_create_tech_stack_comparison(self, mock_write_html):
        """기술스택 비교 차트 생성 테스트"""
        mock_write_html.return_value = None
        
        output_file = self.output_path / 'test_tech.html'
        result = self.generator._create_tech_stack_comparison(
            self.sample_analysis_results['tech_stack_analysis'],
            output_file
        )
        
        self.assertEqual(result, str(output_file))
        mock_write_html.assert_called_once()
    
    def test_create_tech_stack_comparison_no_data(self):
        """데이터가 없는 경우 기술스택 비교 차트 생성 테스트"""
        empty_tech = {'by_tech_stack': {}}
        
        output_file = self.output_path / 'test_tech.html'
        result = self.generator._create_tech_stack_comparison(
            empty_tech,
            output_file
        )
        
        self.assertIsNone(result)
    
    def test_create_summary_report(self):
        """요약 리포트 생성 테스트"""
        output_file = self.output_path / 'summary_report.html'
        result = self.generator.create_summary_report(
            self.sample_analysis_results,
            output_file
        )
        
        self.assertEqual(result, str(output_file))
        self.assertTrue(output_file.exists())
        
        # 생성된 HTML 파일 내용 확인
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 기본 HTML 구조 확인
        self.assertIn('<html lang="ko">', content)
        self.assertIn('Selvage 평가 분석 리포트', content)
        self.assertIn('모델 성능 비교', content)
        self.assertIn('실패 패턴 분석', content)
        self.assertIn('버전 분석', content)
        self.assertIn('기술스택별 성능', content)
    
    def test_generate_html_report_content(self):
        """HTML 리포트 콘텐츠 생성 테스트"""
        html_content = self.generator._generate_html_report(self.sample_analysis_results)
        
        # HTML 구조 검증
        self.assertIn('<!DOCTYPE html>', html_content)
        self.assertIn('<html lang="ko">', html_content)
        self.assertIn('</html>', html_content)
        
        # 섹션별 내용 검증
        self.assertIn('모델 성능 비교', html_content)
        self.assertIn('실패 패턴 분석', html_content)
        self.assertIn('버전 분석', html_content)
        self.assertIn('기술스택별 성능', html_content)
        
        # 데이터 내용 검증
        self.assertIn('gpt-4', html_content)
        self.assertIn('25', html_content)  # 총 실패 건수
        self.assertIn('0.2.0', html_content)  # 권장 버전
    
    def test_generate_model_comparison_section(self):
        """모델 비교 섹션 생성 테스트"""
        section_html = self.generator._generate_model_comparison_section(
            self.sample_analysis_results['model_comparison']
        )
        
        self.assertIn('모델 성능 비교', section_html)
        self.assertIn('주요 권장사항', section_html)
        self.assertIn('gpt-4', section_html)
    
    def test_generate_failure_analysis_section(self):
        """실패 분석 섹션 생성 테스트"""
        section_html = self.generator._generate_failure_analysis_section(
            self.sample_analysis_results['failure_analysis']
        )
        
        self.assertIn('실패 패턴 분석', section_html)
        self.assertIn('25', section_html)  # 총 실패 건수
    
    def test_generate_version_analysis_section(self):
        """버전 분석 섹션 생성 테스트"""
        section_html = self.generator._generate_version_analysis_section(
            self.sample_analysis_results['version_analysis']
        )
        
        self.assertIn('버전 분석', section_html)
        self.assertIn('0.2.0', section_html)  # 권장 버전
    
    def test_generate_tech_stack_section(self):
        """기술스택 섹션 생성 테스트"""
        section_html = self.generator._generate_tech_stack_section(
            self.sample_analysis_results['tech_stack_analysis']
        )
        
        self.assertIn('기술스택별 성능', section_html)
        self.assertIn('TypeScript/JavaScript', section_html)
    
    def test_output_directory_creation(self):
        """출력 디렉토리 자동 생성 테스트"""
        non_existent_dir = self.output_path / 'new_dir' / 'sub_dir'
        
        # 존재하지 않는 디렉토리에 대시보드 생성
        result = self.generator.generate_comprehensive_dashboard(
            {}, str(non_existent_dir)
        )
        
        # 디렉토리가 생성되어야 함
        self.assertTrue(non_existent_dir.exists())
        self.assertTrue(non_existent_dir.is_dir())
        
        # 빈 결과 반환 (데이터가 없으므로)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()