"""DeepEval 분석 엔진 V2

로그 파일 기반 DeepEval 평가 결과를 분석하고 통합된 보고서를 생성하는 엔진입니다.
새로운 분석 클래스들을 통합하여 포괄적인 분석을 제공합니다.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from .deepeval_log_parser import DeepEvalLogParser
from .metric_aggregator import MetricAggregator
from .failure_pattern_analyzer import FailurePatternAnalyzer
from .model_performance_comparator import ModelPerformanceComparator
from .tech_stack_analyzer import TechStackAnalyzer
from .version_comparison_analyzer import VersionComparisonAnalyzer
from .visualization_generator import VisualizationGenerator

logger = logging.getLogger(__name__)


class DeepEvalAnalysisEngine:
    """DeepEval 분석 엔진 V2 - 로그 파일 기반"""
    
    def __init__(self, output_dir: str = "~/Library/selvage-eval/analyze_results"):
        """분석 엔진 초기화
        
        Args:
            output_dir: 분석 결과 출력 디렉토리
        """
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 분석 컴포넌트 초기화
        self.log_parser = DeepEvalLogParser()
        self.metric_aggregator = MetricAggregator()
        self.failure_analyzer = FailurePatternAnalyzer()
        self.model_comparator = ModelPerformanceComparator()
        self.tech_stack_analyzer = TechStackAnalyzer()
        self.version_analyzer = VersionComparisonAnalyzer()
        self.visualizer = VisualizationGenerator()
    
    def analyze_session(self, session_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """세션 분석 실행
        
        Args:
            session_path: DeepEval 로그 파일이 있는 세션 경로
            output_dir: 사용자 지정 출력 디렉토리 (선택사항)
            
        Returns:
            분석 결과 메타데이터
        """
        session_path_obj = Path(session_path).expanduser()
        
        if not session_path_obj.exists():
            raise FileNotFoundError(f"세션 경로가 존재하지 않습니다: {session_path}")
        
        # 출력 디렉토리 설정
        if output_dir:
            final_output_dir = Path(output_dir).expanduser()
        else:
            session_id = session_path_obj.name
            final_output_dir = self.output_dir / session_id
        
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로그 파일 기반 결과 수집
        log_results = self._collect_log_results(session_path_obj)
        
        if not log_results:
            raise ValueError("DeepEval 로그 결과를 찾을 수 없습니다")
        
        # 종합 분석 실행
        analysis_results = self._perform_comprehensive_analysis(log_results)
        
        # 마크다운 보고서 생성
        markdown_report = self._generate_markdown_report(analysis_results)
        report_path = final_output_dir / "analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        # JSON 데이터 저장
        json_path = final_output_dir / "analysis_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 시각화 대시보드 생성
        visualization_files = []
        try:
            viz_files = self.visualizer.generate_comprehensive_dashboard(
                analysis_results, 
                str(final_output_dir / "visualizations")
            )
            visualization_files.extend(viz_files)
            
            # 요약 리포트 생성
            summary_report = self.visualizer.create_summary_report(
                analysis_results,
                final_output_dir / "summary_report.html"
            )
            if summary_report:
                visualization_files.append(summary_report)
                
        except Exception as e:
            logger.error(f"시각화 생성 실패: {e}")
        
        return {
            "analysis_metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "session_path": str(session_path_obj),
                "total_test_cases": analysis_results.get("data_summary", {}).get("total_test_cases", 0),
                "models_analyzed": list(analysis_results.get("model_comparison", {}).get("model_statistics", {}).keys())
            },
            "files_generated": {
                "markdown_report": str(report_path),
                "json_data": str(json_path),
                "visualization_files": visualization_files
            }
        }
    
    def analyze_multiple_sessions(self, base_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """여러 세션 통합 분석 (버전 비교 포함)
        
        Args:
            base_path: 여러 세션이 있는 기본 경로
            output_dir: 사용자 지정 출력 디렉토리
            
        Returns:
            통합 분석 결과 메타데이터
        """
        base_path_obj = Path(base_path).expanduser()
        
        if not base_path_obj.exists():
            raise FileNotFoundError(f"기본 경로가 존재하지 않습니다: {base_path}")
        
        # 출력 디렉토리 설정
        if output_dir:
            final_output_dir = Path(output_dir).expanduser()
        else:
            final_output_dir = self.output_dir / "multi_session_analysis"
        
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 버전별 데이터 수집
        version_data = self.version_analyzer.collect_version_data(str(base_path_obj))
        
        if not version_data:
            raise ValueError("분석할 버전 데이터를 찾을 수 없습니다")
        
        # 버전 비교 분석
        version_analysis = self.version_analyzer.analyze_version_progression(version_data)
        
        # 기술스택별 분석을 위한 데이터 준비
        repo_results = self._prepare_repo_results_from_versions(version_data)
        tech_stack_analysis = self.tech_stack_analyzer.analyze_tech_stack_performance(repo_results)
        
        # 통합 분석 결과
        integrated_results = {
            "analysis_type": "multi_session",
            "version_analysis": version_analysis,
            "tech_stack_analysis": tech_stack_analysis,
            "data_summary": {
                "total_versions": len(version_data),
                "total_sessions": sum(len(data['sessions']) for data in version_data.values()),
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
        # 마크다운 보고서 생성
        markdown_report = self._generate_multi_session_markdown_report(integrated_results)
        report_path = final_output_dir / "multi_session_analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        # JSON 데이터 저장
        json_path = final_output_dir / "multi_session_analysis_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(integrated_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 시각화 대시보드 생성
        visualization_files = []
        try:
            viz_files = self.visualizer.generate_comprehensive_dashboard(
                integrated_results,
                str(final_output_dir / "visualizations")
            )
            visualization_files.extend(viz_files)
            
        except Exception as e:
            logger.error(f"다중 세션 시각화 생성 실패: {e}")
        
        return {
            "analysis_metadata": {
                "analysis_type": "multi_session",
                "analysis_timestamp": datetime.now().isoformat(),
                "base_path": str(base_path_obj),
                "total_versions": len(version_data),
                "total_sessions": sum(len(data['sessions']) for data in version_data.values())
            },
            "files_generated": {
                "markdown_report": str(report_path),
                "json_data": str(json_path),
                "visualization_files": visualization_files
            }
        }
    
    def _collect_log_results(self, session_path: Path) -> Dict[str, List]:
        """로그 파일에서 결과 수집
        
        Args:
            session_path: 세션 디렉토리 경로
            
        Returns:
            모델별 테스트 결과 딕셔너리
        """
        results = {}
        
        # .log 파일들 찾기
        for log_file in session_path.glob("**/*.log"):
            try:
                # 파일명에서 모델명 추출
                model_name = self._extract_model_name_from_path(log_file)
                
                # 로그 파싱
                test_results = list(self.log_parser.parse_log_file(log_file))
                
                if test_results:
                    if model_name not in results:
                        results[model_name] = []
                    results[model_name].extend(test_results)
                    
            except Exception as e:
                logger.warning(f"로그 파일 파싱 실패: {log_file} - {e}")
        
        return results
    
    def _extract_model_name_from_path(self, log_file: Path) -> str:
        """파일 경로에서 모델명 추출
        
        Args:
            log_file: 로그 파일 경로
            
        Returns:
            추출된 모델명
        """
        # 다양한 패턴으로 모델명 추출 시도
        file_name = log_file.stem
        
        # deepeval_results_model_name.log 패턴
        if file_name.startswith('deepeval_results_'):
            parts = file_name.split('_')
            if len(parts) >= 3:
                return '_'.join(parts[2:])
        
        # model_name.log 패턴
        if '_' in file_name:
            return file_name.split('_')[0]
        
        # 기본값
        return file_name or "unknown_model"
    
    def _perform_comprehensive_analysis(self, log_results: Dict[str, List]) -> Dict[str, Any]:
        """종합 분석 수행
        
        Args:
            log_results: 모델별 로그 결과
            
        Returns:
            종합 분석 결과
        """
        if not log_results:
            return {"error": "분석할 데이터가 없습니다"}
        
        try:
            # 모델별 성능 비교
            model_comparison = self.model_comparator.compare_models(log_results)
            
            # 실패 패턴 분석 - 모든 실패 케이스 수집
            all_failed_cases = []
            for model_results in log_results.values():
                for test_case in model_results:
                    # 하나라도 실패한 메트릭이 있으면 실패 케이스로 간주
                    has_failure = (
                        not test_case.correctness.passed or
                        not test_case.clarity.passed or
                        not test_case.actionability.passed or
                        not test_case.json_correctness.passed
                    )
                    if has_failure:
                        all_failed_cases.append(test_case)
            
            failure_analysis = self.failure_analyzer.analyze_failure_patterns(all_failed_cases)
            
            # 데이터 요약
            total_test_cases = sum(len(results) for results in log_results.values())
            successful_cases = total_test_cases - len(all_failed_cases)
            
            return {
                "analysis_type": "single_session", 
                "model_comparison": model_comparison,
                "failure_analysis": failure_analysis,
                "data_summary": {
                    "total_test_cases": total_test_cases,
                    "successful_evaluations": successful_cases,
                    "failed_evaluations": len(all_failed_cases),
                    "models_count": len(log_results),
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"종합 분석 수행 중 오류: {e}")
            return {"error": f"분석 실패: {str(e)}"}
    
    def _prepare_repo_results_from_versions(self, version_data: Dict[str, Any]) -> Dict[str, Dict[str, List]]:
        """버전 데이터에서 저장소별 결과 준비
        
        Args:
            version_data: 버전별 데이터
            
        Returns:
            저장소별 모델 결과
        """
        repo_results = {}
        
        for version, data in version_data.items():
            for session in data['sessions']:
                session_dir = Path(session['session_dir'])
                
                # 세션 경로에서 저장소명 추출 (간단한 휴리스틱)
                repo_name = self._extract_repo_name_from_session_path(session_dir)
                
                if repo_name not in repo_results:
                    repo_results[repo_name] = {}
                
                # 세션의 결과를 모델별로 그룹핑
                session_results = session.get('results', [])
                if session_results:
                    model_name = f"version_{version}"
                    if model_name not in repo_results[repo_name]:
                        repo_results[repo_name][model_name] = []
                    repo_results[repo_name][model_name].extend(session_results)
        
        return repo_results
    
    def _extract_repo_name_from_session_path(self, session_path: Path) -> str:
        """세션 경로에서 저장소명 추출
        
        Args:
            session_path: 세션 디렉토리 경로
            
        Returns:
            추출된 저장소명
        """
        # 경로에서 저장소명으로 보이는 부분 추출
        path_parts = session_path.parts
        
        # 일반적인 저장소명 패턴 찾기
        for part in reversed(path_parts):
            if any(keyword in part.lower() for keyword in ['cline', 'ecommerce', 'kotlin', 'selvage']):
                return part
        
        # 기본값으로 마지막 디렉토리명 사용
        return session_path.name or "unknown_repo"
    
    def _generate_markdown_report(self, analysis_results: Dict[str, Any]) -> str:
        """마크다운 보고서 생성
        
        Args:
            analysis_results: 분석 결과
            
        Returns:
            마크다운 콘텐츠
        """
        if "error" in analysis_results:
            return f"# 분석 오류\n\n{analysis_results['error']}"
        
        lines = []
        
        # 헤더
        lines.extend([
            "# DeepEval 분석 보고서 (V2)",
            "",
            f"**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**분석 유형**: {analysis_results.get('analysis_type', 'single_session')}",
            ""
        ])
        
        # 데이터 요약
        data_summary = analysis_results.get("data_summary", {})
        if data_summary:
            lines.extend([
                "## 📊 데이터 요약",
                "",
                f"- **총 테스트 케이스**: {data_summary.get('total_test_cases', 0)}개",
                f"- **성공한 평가**: {data_summary.get('successful_evaluations', 0)}개",
                f"- **실패한 평가**: {data_summary.get('failed_evaluations', 0)}개",
                f"- **분석된 모델**: {data_summary.get('models_count', 0)}개",
                ""
            ])
        
        # 모델 비교 결과
        model_comparison = analysis_results.get("model_comparison", {})
        if model_comparison and model_comparison.get("recommendations"):
            lines.extend([
                "## 🏆 모델 성능 비교",
                "",
                "### 주요 권장사항",
                ""
            ])
            
            for rec in model_comparison["recommendations"][:10]:  # 상위 10개
                lines.append(f"- {rec}")
            
            lines.append("")
            
            # 비교 표
            comparison_table = model_comparison.get("comparison_table", {})
            table_data = comparison_table.get("table_data", [])
            
            if table_data:
                lines.extend([
                    "### 모델별 성능 요약",
                    "",
                    "| 모델명 | 종합점수 | 등급 | 순위 | 정확성 | 명확성 | 실행가능성 | JSON 정확성 |",
                    "|--------|----------|------|------|--------|--------|------------|-------------|"
                ])
                
                for model_data in table_data:
                    lines.append(
                        f"| {model_data.get('model_name', 'N/A')} | "
                        f"{model_data.get('overall_score', 0):.3f} | "
                        f"{model_data.get('grade', 'N/A')} | "
                        f"{model_data.get('overall_rank', 'N/A')} | "
                        f"{model_data.get('correctness_score', 0):.3f} | "
                        f"{model_data.get('clarity_score', 0):.3f} | "
                        f"{model_data.get('actionability_score', 0):.3f} | "
                        f"{model_data.get('json_correctness_score', 0):.3f} |"
                    )
                
                lines.append("")
        
        # 실패 패턴 분석
        failure_analysis = analysis_results.get("failure_analysis", {})
        if failure_analysis and failure_analysis.get("total_failures", 0) > 0:
            lines.extend([
                "## 🔍 실패 패턴 분석",
                "",
                f"**총 실패 건수**: {failure_analysis['total_failures']}개",
                ""
            ])
            
            # 메트릭별 실패 분석
            by_metric = failure_analysis.get("by_metric", {})
            if by_metric:
                lines.extend([
                    "### 메트릭별 실패 현황",
                    "",
                    "| 메트릭 | 실패 건수 | 실패율 | 평균 신뢰도 |",
                    "|--------|-----------|--------|-------------|"
                ])
                
                for metric, data in by_metric.items():
                    lines.append(
                        f"| {metric.replace('_', ' ').title()} | "
                        f"{data.get('total_failures', 0)} | "
                        f"{data.get('failure_rate', 0):.1%} | "
                        f"{data.get('avg_confidence', 0):.3f} |"
                    )
                
                lines.append("")
            
            # 중요한 패턴
            critical_patterns = failure_analysis.get("critical_patterns", [])
            if critical_patterns:
                lines.extend([
                    "### 중요한 실패 패턴",
                    ""
                ])
                
                for pattern in critical_patterns[:5]:  # 상위 5개
                    lines.append(
                        f"- **{pattern.get('category', 'Unknown')}**: "
                        f"{pattern.get('count', 0)}건 ({pattern.get('percentage', 0):.1f}%) "
                        f"- {pattern.get('reason', '')}"
                    )
                
                lines.append("")
        
        # 결론 및 권장사항
        lines.extend([
            "## 💡 결론 및 권장사항",
            "",
            "### 주요 발견사항",
            ""
        ])
        
        # 전체 권장사항 종합
        all_recommendations = []
        if model_comparison.get("recommendations"):
            all_recommendations.extend(model_comparison["recommendations"][:3])
        
        if failure_analysis.get("critical_patterns"):
            all_recommendations.append(
                f"실패 패턴 개선 필요: {len(failure_analysis['critical_patterns'])}개의 중요한 패턴 발견"
            )
        
        for rec in all_recommendations:
            lines.append(f"1. {rec}")
        
        lines.extend([
            "",
            "### 다음 단계",
            "",
            "- 성능이 낮은 메트릭에 대한 모델 개선",
            "- 실패 패턴이 많은 영역의 프롬프트 최적화",
            "- 최고 성능 모델의 특성을 다른 모델에 적용",
            "- 정기적인 성능 모니터링 및 추적",
            ""
        ])
        
        return "\n".join(lines)
    
    def _generate_multi_session_markdown_report(self, integrated_results: Dict[str, Any]) -> str:
        """다중 세션 마크다운 보고서 생성
        
        Args:
            integrated_results: 통합 분석 결과
            
        Returns:
            마크다운 콘텐츠
        """
        lines = []
        
        # 헤더
        lines.extend([
            "# 다중 세션 분석 보고서",
            "",
            f"**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "**분석 유형**: 버전 비교 및 기술스택 분석",
            ""
        ])
        
        # 데이터 요약
        data_summary = integrated_results.get("data_summary", {})
        lines.extend([
            "## 📊 분석 개요",
            "",
            f"- **분석된 버전**: {data_summary.get('total_versions', 0)}개",
            f"- **총 세션**: {data_summary.get('total_sessions', 0)}개",
            ""
        ])
        
        # 버전 분석
        version_analysis = integrated_results.get("version_analysis", {})
        if version_analysis:
            lines.extend([
                "## 📈 버전별 성능 분석",
                ""
            ])
            
            # 버전 권장사항
            version_recommendations = version_analysis.get("version_recommendations", {})
            if version_recommendations:
                recommended_version = version_recommendations.get("recommended_version")
                if recommended_version:
                    lines.append(f"**권장 버전**: {recommended_version}")
                    lines.append("")
                
                recommendations = version_recommendations.get("recommendations", [])
                if recommendations:
                    lines.append("### 버전 관련 권장사항")
                    lines.append("")
                    for rec in recommendations:
                        lines.append(f"- {rec}")
                    lines.append("")
        
        # 기술스택 분석
        tech_stack_analysis = integrated_results.get("tech_stack_analysis", {})
        if tech_stack_analysis:
            lines.extend([
                "## 🛠 기술스택별 성능 분석",
                ""
            ])
            
            tech_recommendations = tech_stack_analysis.get("recommendations", [])
            if tech_recommendations:
                for rec in tech_recommendations[:10]:  # 상위 10개
                    lines.append(f"- {rec}")
                lines.append("")
        
        lines.extend([
            "## 💡 통합 권장사항",
            "",
            "### 버전 업그레이드",
            "- 최신 안정 버전으로의 업그레이드 검토",
            "- 성능 회귀가 있는 버전 사용 중단",
            "",
            "### 기술스택 최적화", 
            "- 각 기술스택에 최적화된 모델 선택",
            "- 성능이 낮은 기술스택의 프롬프트 개선",
            "",
            "### 지속적 모니터링",
            "- 정기적인 버전별 성능 추적",
            "- 기술스택별 성능 벤치마킹",
            ""
        ])
        
        return "\n".join(lines)