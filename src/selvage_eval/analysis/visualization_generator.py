"""시각화 생성기

DeepEval 분석 결과를 다양한 형태의 시각화 차트로 변환합니다.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """분석 결과 시각화 생성기"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'info': '#17becf'
        }
        
        # 시각화 스타일 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_comprehensive_dashboard(self, analysis_results: Dict[str, Any], 
                                       output_dir: str) -> List[str]:
        """종합 분석 대시보드 생성
        
        Args:
            analysis_results: 분석 결과 데이터
            output_dir: 출력 디렉토리 경로
            
        Returns:
            List[str]: 생성된 파일 경로 리스트
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        try:
            # 모델 성능 비교 레이더 차트
            if 'model_comparison' in analysis_results:
                radar_file = self._create_model_performance_radar(
                    analysis_results['model_comparison'], 
                    output_path / 'model_performance_radar.html'
                )
                if radar_file:
                    generated_files.append(str(radar_file))
            
            # 성능 히트맵
            if 'model_comparison' in analysis_results:
                heatmap_file = self._create_performance_heatmap(
                    analysis_results['model_comparison'],
                    output_path / 'performance_heatmap.html'
                )
                if heatmap_file:
                    generated_files.append(str(heatmap_file))
            
            # 실패 패턴 분석 차트
            if 'failure_analysis' in analysis_results:
                failure_file = self._create_failure_pattern_charts(
                    analysis_results['failure_analysis'],
                    output_path / 'failure_pattern_analysis.html'
                )
                if failure_file:
                    generated_files.append(str(failure_file))
            
            # 버전별 성능 트렌드
            if 'version_analysis' in analysis_results:
                trend_file = self._create_version_trend_chart(
                    analysis_results['version_analysis'],
                    output_path / 'version_trend_analysis.html'
                )
                if trend_file:
                    generated_files.append(str(trend_file))
            
            # 기술스택별 성능 비교
            if 'tech_stack_analysis' in analysis_results:
                tech_file = self._create_tech_stack_comparison(
                    analysis_results['tech_stack_analysis'],
                    output_path / 'tech_stack_comparison.html'
                )
                if tech_file:
                    generated_files.append(str(tech_file))
            
            logger.info(f"Generated {len(generated_files)} visualization files")
            
        except Exception as e:
            logger.error(f"Error generating dashboard: {e}")
        
        return generated_files
    
    def _create_model_performance_radar(self, model_comparison: Dict[str, Any], 
                                      output_path: Path) -> Optional[str]:
        """모델 성능 레이더 차트 생성
        
        Args:
            model_comparison: 모델 비교 데이터
            output_path: 출력 파일 경로
            
        Returns:
            Optional[str]: 생성된 파일 경로
        """
        try:
            comparison_table = model_comparison.get('comparison_table', {})
            table_data = comparison_table.get('table_data', [])
            
            if not table_data:
                logger.warning("No model comparison data available for radar chart")
                return None
            
            # 메트릭 이름 매핑
            metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
            metric_labels = ['정확성', '명확성', '실행가능성', 'JSON 정확성']
            
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, model_data in enumerate(table_data[:5]):  # 최대 5개 모델
                model_name = model_data['model_name']
                scores = [model_data.get(f'{metric}_score', 0) for metric in metrics]
                
                fig.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=metric_labels,
                    fill='toself',
                    name=model_name,
                    line=dict(color=colors[i % len(colors)])
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="모델별 성능 비교 (레이더 차트)",
                showlegend=True,
                width=800,
                height=600
            )
            
            fig.write_html(str(output_path))
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating radar chart: {e}")
            return None
    
    def _create_performance_heatmap(self, model_comparison: Dict[str, Any], 
                                  output_path: Path) -> Optional[str]:
        """성능 히트맵 생성
        
        Args:
            model_comparison: 모델 비교 데이터
            output_path: 출력 파일 경로
            
        Returns:
            Optional[str]: 생성된 파일 경로
        """
        try:
            comparison_table = model_comparison.get('comparison_table', {})
            table_data = comparison_table.get('table_data', [])
            
            if not table_data:
                logger.warning("No model comparison data available for heatmap")
                return None
            
            # 데이터 준비
            metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
            metric_labels = ['정확성', '명확성', '실행가능성', 'JSON 정확성']
            
            models = []
            scores_matrix = []
            
            for model_data in table_data:
                models.append(model_data['model_name'])
                scores = [model_data.get(f'{metric}_score', 0) for metric in metrics]
                scores_matrix.append(scores)
            
            # 히트맵 생성
            fig = go.Figure(data=go.Heatmap(
                z=scores_matrix,
                x=metric_labels,
                y=models,
                colorscale='RdYlGn',
                zmid=0.7,  # 중간값을 0.7로 설정 (합격선)
                colorbar=dict(
                    title="성능 점수",
                    tickmode="linear",
                    tick0=0,
                    dtick=0.1
                ),
                text=[[f"{score:.3f}" for score in row] for row in scores_matrix],
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="모델별 메트릭 성능 히트맵",
                xaxis_title="메트릭",
                yaxis_title="모델",
                width=800,
                height=500
            )
            
            fig.write_html(str(output_path))
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return None
    
    def _create_failure_pattern_charts(self, failure_analysis: Dict[str, Any], 
                                     output_path: Path) -> Optional[str]:
        """실패 패턴 분석 차트 생성
        
        Args:
            failure_analysis: 실패 패턴 분석 데이터
            output_path: 출력 파일 경로
            
        Returns:
            Optional[str]: 생성된 파일 경로
        """
        try:
            # 2x2 서브플롯 생성
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('메트릭별 실패 분포', '실패 카테고리별 분포', 
                              '심각도별 실패 건수', '개선 우선순위'),
                specs=[[{'type': 'bar'}, {'type': 'pie'}],
                       [{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            # 1. 메트릭별 실패 분포
            by_metric = failure_analysis.get('by_metric', {})
            if by_metric:
                metrics = list(by_metric.keys())
                metric_labels = [m.replace('_', ' ').title() for m in metrics]
                failure_counts = [by_metric[m].get('total_failures', 0) for m in metrics]
                
                fig.add_trace(
                    go.Bar(x=metric_labels, y=failure_counts, 
                          marker_color=self.color_palette['danger'],
                          name='실패 건수'),
                    row=1, col=1
                )
            
            # 2. 실패 카테고리별 분포
            by_category = failure_analysis.get('by_category', {})
            if by_category:
                categories = list(by_category.keys())
                counts = list(by_category.values())
                
                fig.add_trace(
                    go.Pie(labels=categories, values=counts, name="카테고리별 실패"),
                    row=1, col=2
                )
            
            # 3. 심각도별 실패 건수 (가상 데이터 - 실제로는 failure_analysis에서 추출)
            critical_patterns = failure_analysis.get('critical_patterns', [])
            severity_counts = {}
            for pattern in critical_patterns:
                severity = pattern.get('severity', 'medium')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if severity_counts:
                severities = list(severity_counts.keys())
                counts = list(severity_counts.values())
                colors = [self.color_palette['danger'] if s == 'high' 
                         else self.color_palette['warning'] if s == 'medium'
                         else self.color_palette['info'] for s in severities]
                
                fig.add_trace(
                    go.Bar(x=severities, y=counts, marker_color=colors,
                          name='심각도별 건수'),
                    row=2, col=1
                )
            
            # 4. 개선 우선순위 (실패 빈도 기준)
            if by_metric:
                sorted_metrics = sorted(by_metric.items(), 
                                      key=lambda x: x[1].get('total_failures', 0), 
                                      reverse=True)
                priority_metrics = [m[0].replace('_', ' ').title() for m in sorted_metrics]
                priority_scores = [m[1].get('total_failures', 0) for m in sorted_metrics]
                
                fig.add_trace(
                    go.Bar(x=priority_metrics, y=priority_scores,
                          marker_color=self.color_palette['warning'],
                          name='우선순위'),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="실패 패턴 종합 분석",
                showlegend=False,
                width=1200,
                height=800
            )
            
            fig.write_html(str(output_path))
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating failure pattern charts: {e}")
            return None
    
    def _create_version_trend_chart(self, version_analysis: Dict[str, Any], 
                                  output_path: Path) -> Optional[str]:
        """버전별 성능 트렌드 차트 생성
        
        Args:
            version_analysis: 버전 분석 데이터
            output_path: 출력 파일 경로
            
        Returns:
            Optional[str]: 생성된 파일 경로
        """
        try:
            version_timeline = version_analysis.get('version_timeline', [])
            
            if not version_timeline:
                logger.warning("No version timeline data available")
                return None
            
            fig = go.Figure()
            
            # 버전 정보 추출
            versions = [v['version'] for v in version_timeline]
            
            # 메트릭별 트렌드 라인
            metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
            metric_labels = ['정확성', '명확성', '실행가능성', 'JSON 정확성']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                scores = []
                for version_data in version_timeline:
                    performance = version_data.get('performance', {})
                    if metric in performance:
                        scores.append(performance[metric]['mean_score'])
                    else:
                        scores.append(0.0)
                
                # 실제 점수 플롯
                fig.add_trace(go.Scatter(
                    x=versions,
                    y=scores,
                    mode='markers+lines',
                    name=label,
                    line=dict(color=colors[i]),
                    marker=dict(size=8)
                ))
                
                # 트렌드 라인 (선형 회귀)
                if len(scores) >= 2:
                    x_numeric = np.arange(len(scores))
                    z = np.polyfit(x_numeric, scores, 1)
                    trend_line = np.poly1d(z)(x_numeric)
                    
                    fig.add_trace(go.Scatter(
                        x=versions,
                        y=trend_line,
                        mode='lines',
                        name=f'{label} 트렌드',
                        line=dict(color=colors[i], dash='dash'),
                        opacity=0.7
                    ))
            
            fig.update_layout(
                title="버전별 성능 트렌드 분석",
                xaxis_title="버전",
                yaxis_title="성능 점수",
                yaxis=dict(range=[0, 1]),
                width=1000,
                height=600,
                hovermode='x unified'
            )
            
            fig.write_html(str(output_path))
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating version trend chart: {e}")
            return None
    
    def _create_tech_stack_comparison(self, tech_stack_analysis: Dict[str, Any], 
                                    output_path: Path) -> Optional[str]:
        """기술스택별 성능 비교 차트 생성
        
        Args:
            tech_stack_analysis: 기술스택 분석 데이터
            output_path: 출력 파일 경로
            
        Returns:
            Optional[str]: 생성된 파일 경로
        """
        try:
            by_tech_stack = tech_stack_analysis.get('by_tech_stack', {})
            
            if not by_tech_stack:
                logger.warning("No tech stack data available")
                return None
            
            # 데이터 준비
            tech_stacks = []
            best_scores = []
            best_models = []
            
            for tech_stack, data in by_tech_stack.items():
                best_model = data.get('best_model', {})
                if best_model.get('name'):
                    tech_stacks.append(tech_stack)
                    best_scores.append(best_model.get('score', 0))
                    best_models.append(best_model.get('name', 'Unknown'))
            
            if not tech_stacks:
                logger.warning("No valid tech stack data for visualization")
                return None
            
            # 막대 차트 생성
            fig = go.Figure()
            
            # 성능 점수별 색상 지정
            colors = [self.color_palette['success'] if score >= 0.8 
                     else self.color_palette['warning'] if score >= 0.7
                     else self.color_palette['danger'] for score in best_scores]
            
            fig.add_trace(go.Bar(
                x=tech_stacks,
                y=best_scores,
                marker_color=colors,
                text=[f'{model}<br>{score:.3f}' for model, score in zip(best_models, best_scores)],
                textposition='auto',
                name='최고 성능'
            ))
            
            # 합격선 표시
            fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                         annotation_text="합격선 (0.7)")
            
            fig.update_layout(
                title="기술스택별 최고 성능 비교",
                xaxis_title="기술스택",
                yaxis_title="성능 점수",
                yaxis=dict(range=[0, 1]),
                width=800,
                height=500
            )
            
            fig.write_html(str(output_path))
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating tech stack comparison: {e}")
            return None
    
    def create_summary_report(self, analysis_results: Dict[str, Any], 
                            output_path: Path) -> Optional[str]:
        """분석 결과 요약 리포트 생성
        
        Args:
            analysis_results: 분석 결과 데이터
            output_path: 출력 파일 경로
            
        Returns:
            Optional[str]: 생성된 파일 경로
        """
        try:
            # HTML 리포트 생성
            html_content = self._generate_html_report(analysis_results)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating summary report: {e}")
            return None
    
    def _generate_html_report(self, analysis_results: Dict[str, Any]) -> str:
        """HTML 리포트 생성
        
        Args:
            analysis_results: 분석 결과 데이터
            
        Returns:
            str: HTML 콘텐츠
        """
        html = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Selvage 평가 분석 리포트</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: {self.color_palette['primary']}; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f5f5f5; border-radius: 3px; }}
                .success {{ color: {self.color_palette['success']}; }}
                .warning {{ color: {self.color_palette['warning']}; }}
                .danger {{ color: {self.color_palette['danger']}; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Selvage 평가 분석 리포트</h1>
                <p>AI 코드 리뷰 도구 성능 분석 결과</p>
            </div>
        """
        
        # 모델 비교 섹션
        if 'model_comparison' in analysis_results:
            html += self._generate_model_comparison_section(analysis_results['model_comparison'])
        
        # 실패 패턴 섹션
        if 'failure_analysis' in analysis_results:
            html += self._generate_failure_analysis_section(analysis_results['failure_analysis'])
        
        # 버전 분석 섹션
        if 'version_analysis' in analysis_results:
            html += self._generate_version_analysis_section(analysis_results['version_analysis'])
        
        # 기술스택 분석 섹션
        if 'tech_stack_analysis' in analysis_results:
            html += self._generate_tech_stack_section(analysis_results['tech_stack_analysis'])
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_model_comparison_section(self, model_comparison: Dict[str, Any]) -> str:
        """모델 비교 섹션 생성"""
        recommendations = model_comparison.get('recommendations', [])
        
        html = """
        <div class="section">
            <h2>모델 성능 비교</h2>
        """
        
        # 권장사항
        if recommendations:
            html += "<h3>주요 권장사항</h3><ul>"
            for rec in recommendations[:5]:  # 상위 5개만 표시
                html += f"<li>{rec}</li>"
            html += "</ul>"
        
        html += "</div>"
        return html
    
    def _generate_failure_analysis_section(self, failure_analysis: Dict[str, Any]) -> str:
        """실패 분석 섹션 생성"""
        total_failures = failure_analysis.get('total_failures', 0)
        
        html = f"""
        <div class="section">
            <h2>실패 패턴 분석</h2>
            <p><strong>총 실패 건수:</strong> {total_failures}</p>
        </div>
        """
        return html
    
    def _generate_version_analysis_section(self, version_analysis: Dict[str, Any]) -> str:
        """버전 분석 섹션 생성"""
        recommendations = version_analysis.get('version_recommendations', {})
        recommended_version = recommendations.get('recommended_version', 'N/A')
        
        html = f"""
        <div class="section">
            <h2>버전 분석</h2>
            <p><strong>권장 버전:</strong> {recommended_version}</p>
        </div>
        """
        return html
    
    def _generate_tech_stack_section(self, tech_stack_analysis: Dict[str, Any]) -> str:
        """기술스택 분석 섹션 생성"""
        recommendations = tech_stack_analysis.get('recommendations', [])
        
        html = """
        <div class="section">
            <h2>기술스택별 성능</h2>
        """
        
        if recommendations:
            html += "<ul>"
            for rec in recommendations[:3]:  # 상위 3개만 표시
                html += f"<li>{rec}</li>"
            html += "</ul>"
        
        html += "</div>"
        return html