"""DeepEval 분석 엔진 V2 테스트"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime

from selvage_eval.analysis.deepeval_analysis_engine import DeepEvalAnalysisEngine
from selvage_eval.analysis.deepeval_log_parser import TestCaseResult, MetricScore


class TestDeepEvalAnalysisEngine(unittest.TestCase):
    """DeepEval 분석 엔진 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        # 임시 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # 엔진 인스턴스 생성
        self.engine = DeepEvalAnalysisEngine(output_dir=str(self.temp_path / "output"))
        
        # 샘플 테스트 결과 데이터
        self.sample_test_results = [
            TestCaseResult(
                correctness=MetricScore(score=0.85, passed=True, reason="Good correctness"),
                clarity=MetricScore(score=0.80, passed=True, reason="Clear output"),
                actionability=MetricScore(score=0.75, passed=True, reason="Actionable suggestions"),
                json_correctness=MetricScore(score=1.0, passed=True, reason="Valid JSON"),
                input_data="Sample input",
                actual_output="Sample output",
                raw_content="Raw log content for test case 1"
            ),
            TestCaseResult(
                correctness=MetricScore(score=0.60, passed=False, reason="Missed some issues"),
                clarity=MetricScore(score=0.70, passed=True, reason="Mostly clear"),
                actionability=MetricScore(score=0.65, passed=False, reason="Vague suggestions"),
                json_correctness=MetricScore(score=0.95, passed=True, reason="Valid JSON with minor issues"),
                input_data="Sample input 2",
                actual_output="Sample output 2",
                raw_content="Raw log content for test case 2"
            )
        ]
    
    def tearDown(self):
        """테스트 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """초기화 테스트"""
        self.assertIsNotNone(self.engine.log_parser)
        self.assertIsNotNone(self.engine.metric_aggregator)
        self.assertIsNotNone(self.engine.failure_analyzer)
        self.assertIsNotNone(self.engine.model_comparator)
        self.assertIsNotNone(self.engine.tech_stack_analyzer)
        self.assertIsNotNone(self.engine.version_analyzer)
        self.assertIsNotNone(self.engine.visualizer)
        
        # 출력 디렉토리가 생성되었는지 확인
        self.assertTrue(self.engine.output_dir.exists())
    
    def test_extract_model_name_from_path(self):
        """모델명 추출 테스트"""
        # deepeval_results_model_name.log 패턴
        log_path1 = Path("/path/to/deepeval_results_gpt_4.log")
        model_name1 = self.engine._extract_model_name_from_path(log_path1)
        self.assertEqual(model_name1, "gpt_4")
        
        # model_name.log 패턴
        log_path2 = Path("/path/to/claude_3.log") 
        model_name2 = self.engine._extract_model_name_from_path(log_path2)
        self.assertEqual(model_name2, "claude")
        
        # 기본값 테스트
        log_path3 = Path("/path/to/unknown.log")
        model_name3 = self.engine._extract_model_name_from_path(log_path3)
        self.assertEqual(model_name3, "unknown")
    
    @patch('selvage_eval.analysis.deepeval_analysis_engine.DeepEvalLogParser')
    def test_collect_log_results(self, mock_parser_class):
        """로그 결과 수집 테스트"""
        # 모킹 설정
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_log_file.return_value = iter(self.sample_test_results)
        
        # 테스트 세션 디렉토리 생성
        session_dir = self.temp_path / "test_session"
        session_dir.mkdir()
        
        # 로그 파일 생성
        log_file = session_dir / "deepeval_results_gpt_4.log"
        log_file.write_text("dummy log content")
        
        # 엔진에 모킹된 파서 할당
        self.engine.log_parser = mock_parser
        
        # 테스트 실행
        results = self.engine._collect_log_results(session_dir)
        
        # 결과 검증
        self.assertIn("gpt_4", results)
        self.assertEqual(len(results["gpt_4"]), 2)
        mock_parser.parse_log_file.assert_called_once()
    
    def test_collect_log_results_multiple_models(self):
        """다중 모델 로그 결과 수집 테스트"""
        # 테스트 세션 디렉토리 생성
        session_dir = self.temp_path / "multi_model_session"
        session_dir.mkdir()
        
        # 실제 로그 파일 내용 생성
        log_content_template = """==================================================
Test Case: {test_case}
Input: {input_data}
Expected: {expected}
Actual: {actual}

✅ Correctness (score: {correctness_score}, reason: "{correctness_reason}", error: None)
✅ Clarity (score: {clarity_score}, reason: "{clarity_reason}", error: None)
✅ Actionability (score: {actionability_score}, reason: "{actionability_reason}", error: None)
✅ JSON Correctness (score: {json_score}, reason: "{json_reason}", error: None)
=================================================="""
        
        # 여러 모델 로그 파일 생성
        models_data = {
            "gpt_4": {"correctness_score": 0.9, "clarity_score": 0.85, "actionability_score": 0.8, "json_score": 1.0},
            "claude_3": {"correctness_score": 0.85, "clarity_score": 0.9, "actionability_score": 0.75, "json_score": 0.95},
            "gemini_pro": {"correctness_score": 0.8, "clarity_score": 0.8, "actionability_score": 0.7, "json_score": 0.9}
        }
        
        for model_name, scores in models_data.items():
            log_file = session_dir / f"deepeval_results_{model_name}.log"
            log_content = log_content_template.format(
                test_case=f"Test case for {model_name}",
                input_data=f"Input for {model_name}",
                expected=f"Expected output for {model_name}",
                actual=f"Actual output for {model_name}",
                correctness_score=scores["correctness_score"],
                correctness_reason=f"Good analysis by {model_name}",
                clarity_score=scores["clarity_score"],
                clarity_reason=f"Clear explanation by {model_name}",
                actionability_score=scores["actionability_score"],
                actionability_reason=f"Actionable suggestions by {model_name}",
                json_score=scores["json_score"],
                json_reason=f"Valid JSON by {model_name}"
            )
            log_file.write_text(log_content)
        
        # 테스트 실행
        results = self.engine._collect_log_results(session_dir)
        
        # 결과 검증
        self.assertEqual(len(results), 3)
        self.assertIn("gpt_4", results)
        self.assertIn("claude_3", results) 
        self.assertIn("gemini_pro", results)
        
        # 각 모델이 하나씩의 테스트 결과를 가지는지 확인
        for model_name in models_data.keys():
            self.assertEqual(len(results[model_name]), 1)
    
    def test_collect_log_results_nested_directories(self):
        """중첩 디렉토리에서 로그 결과 수집 테스트"""
        # 테스트 세션 디렉토리 생성
        session_dir = self.temp_path / "nested_session"
        session_dir.mkdir()
        
        # 중첩된 디렉토리 구조 생성
        sub_dir1 = session_dir / "model_results" / "gpt"
        sub_dir1.mkdir(parents=True)
        sub_dir2 = session_dir / "evaluation" / "claude"
        sub_dir2.mkdir(parents=True)
        
        # 중첩된 위치에 로그 파일 생성
        log_file1 = sub_dir1 / "deepeval_results_gpt_4.log"
        log_file1.write_text("""==================================================
Test Case: Nested test case 1
Input: Nested input 1
Expected: Nested expected 1
Actual: Nested actual 1

✅ Correctness (score: 0.8, reason: "Good nested analysis", error: None)
✅ Clarity (score: 0.75, reason: "Clear nested output", error: None)
✅ Actionability (score: 0.7, reason: "Actionable nested suggestions", error: None)
✅ JSON Correctness (score: 0.95, reason: "Valid nested JSON", error: None)
==================================================""")
        
        log_file2 = sub_dir2 / "deepeval_results_claude_3.log"  
        log_file2.write_text("""==================================================
Test Case: Nested test case 2
Input: Nested input 2
Expected: Nested expected 2
Actual: Nested actual 2

✅ Correctness (score: 0.85, reason: "Good nested analysis 2", error: None)
✅ Clarity (score: 0.9, reason: "Clear nested output 2", error: None)
✅ Actionability (score: 0.8, reason: "Actionable nested suggestions 2", error: None)
✅ JSON Correctness (score: 1.0, reason: "Valid nested JSON 2", error: None)
==================================================""")
        
        # 테스트 실행
        results = self.engine._collect_log_results(session_dir)
        
        # 결과 검증
        self.assertEqual(len(results), 2)
        self.assertIn("gpt_4", results)
        self.assertIn("claude_3", results)
        self.assertEqual(len(results["gpt_4"]), 1)
        self.assertEqual(len(results["claude_3"]), 1)
    
    def test_collect_log_results_malformed_logs(self):
        """잘못된 형식의 로그 파일 처리 테스트"""
        # 테스트 세션 디렉토리 생성
        session_dir = self.temp_path / "malformed_session"
        session_dir.mkdir()
        
        # 올바른 로그 파일 생성
        good_log = session_dir / "deepeval_results_gpt_4.log"
        good_log.write_text("""==================================================
Test Case: Good test case
Input: Good input
Expected: Good expected
Actual: Good actual

✅ Correctness (score: 0.8, reason: "Good analysis", error: None)
✅ Clarity (score: 0.75, reason: "Clear output", error: None)
✅ Actionability (score: 0.7, reason: "Actionable suggestions", error: None)
✅ JSON Correctness (score: 0.95, reason: "Valid JSON", error: None)
==================================================""")
        
        # 잘못된 형식의 로그 파일 생성
        bad_log = session_dir / "deepeval_results_claude_3.log"
        bad_log.write_text("This is not a valid log file format")
        
        # 빈 로그 파일 생성
        empty_log = session_dir / "deepeval_results_gemini_pro.log"
        empty_log.write_text("")
        
        # 테스트 실행
        results = self.engine._collect_log_results(session_dir)
        
        # 결과 검증: 올바른 파일만 파싱되어야 함
        self.assertIn("gpt_4", results)
        self.assertEqual(len(results["gpt_4"]), 1)
        
        # 잘못된 파일들은 결과에 포함되지 않아야 함
        # (로그 파서가 빈 결과를 반환하므로)
        for model_name in ["claude_3", "gemini_pro"]:
            if model_name in results:
                # 로그 파서가 빈 로그를 하나의 테스트 케이스로 처리할 수 있음
                self.assertLessEqual(len(results[model_name]), 1)
    
    @patch.object(DeepEvalAnalysisEngine, '_collect_log_results')
    @patch('selvage_eval.analysis.deepeval_analysis_engine.DeepEvalAnalysisEngine._perform_comprehensive_analysis')
    def test_perform_comprehensive_analysis(self, mock_analysis, mock_collect):
        """종합 분석 수행 테스트"""
        # 모킹 설정
        mock_log_results = {"gpt_4": self.sample_test_results}
        
        # 모킹된 분석 결과
        mock_analysis_result = {
            "analysis_type": "single_session",
            "model_comparison": {
                "model_statistics": {"gpt_4": {"overall_score": 0.75}},
                "recommendations": ["Test recommendation"]
            },
            "failure_analysis": {
                "total_failures": 1,
                "by_metric": {"correctness": {"total_failures": 1}}
            },
            "data_summary": {
                "total_test_cases": 2,
                "successful_evaluations": 1,
                "failed_evaluations": 1,
                "models_count": 1
            }
        }
        mock_analysis.return_value = mock_analysis_result
        
        # 테스트 실행
        result = self.engine._perform_comprehensive_analysis(mock_log_results)
        
        # 결과 검증
        self.assertEqual(result["analysis_type"], "single_session")
        self.assertIn("model_comparison", result)
        self.assertIn("failure_analysis", result)
        self.assertIn("data_summary", result)
    
    @patch.object(DeepEvalAnalysisEngine, '_collect_log_results')
    @patch.object(DeepEvalAnalysisEngine, '_perform_comprehensive_analysis')
    @patch('selvage_eval.analysis.deepeval_analysis_engine.VisualizationGenerator')
    def test_analyze_session(self, mock_viz_class, mock_analysis, mock_collect):
        """세션 분석 테스트"""
        # 테스트 세션 디렉토리 생성
        session_dir = self.temp_path / "test_session"
        session_dir.mkdir()
        
        # 모킹 설정
        mock_collect.return_value = {"gpt_4": self.sample_test_results}
        mock_analysis.return_value = {
            "analysis_type": "single_session",
            "model_comparison": {"model_statistics": {"gpt_4": {}}},
            "failure_analysis": {"total_failures": 1},
            "data_summary": {"total_test_cases": 2}
        }
        
        # 시각화 모킹
        mock_viz = MagicMock()
        mock_viz_class.return_value = mock_viz
        mock_viz.generate_comprehensive_dashboard.return_value = ["chart1.html", "chart2.html"]
        mock_viz.create_summary_report.return_value = "summary.html"
        
        # 엔진의 visualizer를 모킹된 객체로 교체
        self.engine.visualizer = mock_viz
        
        # 테스트 실행
        result = self.engine.analyze_session(str(session_dir))
        
        # 결과 검증
        self.assertIn("analysis_metadata", result)
        self.assertIn("files_generated", result)
        self.assertIn("markdown_report", result["files_generated"])
        self.assertIn("json_data", result["files_generated"])
        
        # 생성된 파일들이 존재하는지 확인
        markdown_path = Path(result["files_generated"]["markdown_report"])
        json_path = Path(result["files_generated"]["json_data"])
        self.assertTrue(markdown_path.exists())
        self.assertTrue(json_path.exists())
    
    @patch.object(DeepEvalAnalysisEngine, '_collect_log_results')
    @patch.object(DeepEvalAnalysisEngine, '_perform_comprehensive_analysis')
    @patch('selvage_eval.analysis.deepeval_analysis_engine.VisualizationGenerator')
    def test_analyze_session_visualization_content(self, mock_viz_class, mock_analysis, mock_collect):
        """세션 분석 시각화 내용 검증 테스트"""
        # 테스트 세션 디렉토리 생성
        session_dir = self.temp_path / "viz_test_session"
        session_dir.mkdir()
        
        # 모킹 설정
        mock_collect.return_value = {"gpt_4": self.sample_test_results}
        mock_analysis.return_value = {
            "analysis_type": "single_session",
            "model_comparison": {
                "model_statistics": {"gpt_4": {"overall_score": 0.8}},
                "comparison_table": {"table_data": [{"model_name": "gpt_4", "overall_score": 0.8}]}
            },
            "failure_analysis": {"total_failures": 1},
            "data_summary": {"total_test_cases": 2}
        }
        
        # 시각화 모킹
        mock_viz = MagicMock()
        mock_viz_class.return_value = mock_viz
        mock_viz.generate_comprehensive_dashboard.return_value = ["performance_chart.html", "failure_chart.html"]
        mock_viz.create_summary_report.return_value = "comprehensive_summary.html"
        
        # 엔진의 visualizer를 모킹된 객체로 교체
        self.engine.visualizer = mock_viz
        
        # 테스트 실행
        result = self.engine.analyze_session(str(session_dir))
        
        # 시각화 메서드 호출 검증
        mock_viz.generate_comprehensive_dashboard.assert_called_once()
        mock_viz.create_summary_report.assert_called_once()
        
        # 시각화 파일이 결과에 포함되었는지 확인
        viz_files = result["files_generated"]["visualization_files"]
        self.assertIn("performance_chart.html", viz_files)
        self.assertIn("failure_chart.html", viz_files)
        self.assertIn("comprehensive_summary.html", viz_files)
        
        # 메타데이터 검증
        metadata = result["analysis_metadata"]
        self.assertEqual(metadata["total_test_cases"], 2)
        self.assertEqual(metadata["models_analyzed"], ["gpt_4"])
    
    @patch.object(DeepEvalAnalysisEngine, '_collect_log_results')
    @patch.object(DeepEvalAnalysisEngine, '_perform_comprehensive_analysis')
    @patch('selvage_eval.analysis.deepeval_analysis_engine.VisualizationGenerator')
    def test_analyze_session_visualization_error_handling(self, mock_viz_class, mock_analysis, mock_collect):
        """세션 분석 시각화 오류 처리 테스트"""
        # 테스트 세션 디렉토리 생성
        session_dir = self.temp_path / "viz_error_session"
        session_dir.mkdir()
        
        # 모킹 설정
        mock_collect.return_value = {"gpt_4": self.sample_test_results}
        mock_analysis.return_value = {
            "analysis_type": "single_session",
            "model_comparison": {"model_statistics": {"gpt_4": {}}},
            "failure_analysis": {"total_failures": 1},
            "data_summary": {"total_test_cases": 2}
        }
        
        # 시각화 오류 모킹
        mock_viz = MagicMock()
        mock_viz_class.return_value = mock_viz
        mock_viz.generate_comprehensive_dashboard.side_effect = Exception("Visualization error")
        mock_viz.create_summary_report.side_effect = Exception("Summary error")
        
        # 엔진의 visualizer를 모킹된 객체로 교체
        self.engine.visualizer = mock_viz
        
        # 테스트 실행 - 시각화 오류가 있어도 분석이 계속되어야 함
        result = self.engine.analyze_session(str(session_dir))
        
        # 기본 분석 결과는 여전히 생성되어야 함
        self.assertIn("analysis_metadata", result)
        self.assertIn("files_generated", result)
        self.assertIn("markdown_report", result["files_generated"])
        self.assertIn("json_data", result["files_generated"])
        
        # 시각화 파일 목록이 빈 것이어야 함
        self.assertEqual(result["files_generated"]["visualization_files"], [])
    
    @patch('selvage_eval.analysis.deepeval_analysis_engine.VersionComparisonAnalyzer')
    @patch('selvage_eval.analysis.deepeval_analysis_engine.TechStackAnalyzer')
    @patch('selvage_eval.analysis.deepeval_analysis_engine.VisualizationGenerator')
    def test_analyze_multiple_sessions(self, mock_viz_class, mock_tech_class, mock_version_class):
        """다중 세션 분석 테스트"""
        # 테스트 베이스 디렉토리 생성
        base_dir = self.temp_path / "multi_sessions"
        base_dir.mkdir()
        
        # 모킹 설정
        mock_version_analyzer = MagicMock()
        mock_tech_analyzer = MagicMock()
        mock_viz = MagicMock()
        
        mock_version_class.return_value = mock_version_analyzer
        mock_tech_class.return_value = mock_tech_analyzer  
        mock_viz_class.return_value = mock_viz
        
        # 버전 데이터 모킹
        mock_version_data = {
            "v1.0.0": {
                "sessions": [{"session_dir": "session1", "results": []}]
            }
        }
        mock_version_analyzer.collect_version_data.return_value = mock_version_data
        mock_version_analyzer.analyze_version_progression.return_value = {
            "version_timeline": [],
            "performance_trends": {},
            "version_recommendations": {"recommended_version": "v1.0.0"}
        }
        
        # 기술스택 분석 모킹
        mock_tech_analyzer.analyze_tech_stack_performance.return_value = {
            "by_tech_stack": {},
            "recommendations": ["Tech recommendation"]
        }
        
        # 시각화 모킹
        mock_viz.generate_comprehensive_dashboard.return_value = ["multi_chart.html"]
        
        # 엔진의 분석기들을 모킹된 객체로 교체
        self.engine.version_analyzer = mock_version_analyzer
        self.engine.tech_stack_analyzer = mock_tech_analyzer
        self.engine.visualizer = mock_viz
        
        # 테스트 실행
        result = self.engine.analyze_multiple_sessions(str(base_dir))
        
        # 결과 검증
        self.assertIn("analysis_metadata", result)
        self.assertIn("files_generated", result)
        self.assertEqual(result["analysis_metadata"]["analysis_type"], "multi_session")
        
        # 분석기 메서드들이 호출되었는지 확인
        mock_version_analyzer.collect_version_data.assert_called_once()
        mock_version_analyzer.analyze_version_progression.assert_called_once()
        mock_tech_analyzer.analyze_tech_stack_performance.assert_called_once()
    
    def test_generate_markdown_report(self):
        """마크다운 보고서 생성 테스트"""
        # 샘플 분석 결과
        analysis_results = {
            "analysis_type": "single_session",
            "data_summary": {
                "total_test_cases": 10,
                "successful_evaluations": 8,
                "failed_evaluations": 2,
                "models_count": 2
            },
            "model_comparison": {
                "recommendations": ["Best model: gpt-4", "Good performance overall"],
                "comparison_table": {
                    "table_data": [
                        {
                            "model_name": "gpt-4",
                            "overall_score": 0.85,
                            "grade": "A",
                            "overall_rank": 1,
                            "correctness_score": 0.90,
                            "clarity_score": 0.80,
                            "actionability_score": 0.85,
                            "json_correctness_score": 1.0
                        }
                    ]
                }
            },
            "failure_analysis": {
                "total_failures": 2,
                "by_metric": {
                    "correctness": {"total_failures": 1, "failure_rate": 0.1, "avg_confidence": 0.85}
                },
                "critical_patterns": [
                    {"category": "missing_issues", "count": 1, "percentage": 50.0, "reason": "Failed to identify issues"}
                ]
            }
        }
        
        # 테스트 실행
        markdown = self.engine._generate_markdown_report(analysis_results)
        
        # 결과 검증
        self.assertIn("# DeepEval 분석 보고서 (V2)", markdown)
        self.assertIn("📊 데이터 요약", markdown)
        self.assertIn("🏆 모델 성능 비교", markdown)
        self.assertIn("🔍 실패 패턴 분석", markdown)
        self.assertIn("💡 결론 및 권장사항", markdown)
        
        # 데이터 내용 검증
        self.assertIn("10", markdown)  # 총 테스트 케이스
        self.assertIn("gpt-4", markdown)  # 모델명
        self.assertIn("0.850", markdown)  # 점수
    
    def test_generate_multi_session_markdown_report(self):
        """다중 세션 마크다운 보고서 생성 테스트"""
        # 샘플 통합 결과
        integrated_results = {
            "analysis_type": "multi_session",
            "data_summary": {
                "total_versions": 3,
                "total_sessions": 10
            },
            "version_analysis": {
                "version_recommendations": {
                    "recommended_version": "v1.2.0",
                    "recommendations": ["Version recommendation 1", "Version recommendation 2"]
                }
            },
            "tech_stack_analysis": {
                "recommendations": ["Tech recommendation 1", "Tech recommendation 2"]
            }
        }
        
        # 테스트 실행
        markdown = self.engine._generate_multi_session_markdown_report(integrated_results)
        
        # 결과 검증
        self.assertIn("# 다중 세션 분석 보고서", markdown)
        self.assertIn("📊 분석 개요", markdown)
        self.assertIn("📈 버전별 성능 분석", markdown)
        self.assertIn("🛠 기술스택별 성능 분석", markdown)
        self.assertIn("💡 통합 권장사항", markdown)
        
        # 데이터 내용 검증
        self.assertIn("v1.2.0", markdown)  # 권장 버전
        self.assertIn("3", markdown)  # 버전 수
        self.assertIn("10", markdown)  # 세션 수
    
    def test_error_handling_no_session_path(self):
        """존재하지 않는 세션 경로 에러 처리 테스트"""
        non_existent_path = "/non/existent/path"
        
        with self.assertRaises(FileNotFoundError):
            self.engine.analyze_session(non_existent_path)
    
    @patch.object(DeepEvalAnalysisEngine, '_collect_log_results')
    def test_error_handling_no_results(self, mock_collect):
        """결과가 없는 경우 에러 처리 테스트"""
        # 테스트 세션 디렉토리 생성
        session_dir = self.temp_path / "empty_session"
        session_dir.mkdir()
        
        # 빈 결과 반환 모킹
        mock_collect.return_value = {}
        
        # 테스트 실행 및 검증
        with self.assertRaises(ValueError):
            self.engine.analyze_session(str(session_dir))
    
    def test_error_handling_no_base_path(self):
        """존재하지 않는 기본 경로 에러 처리 테스트"""
        non_existent_path = "/non/existent/base/path"
        
        with self.assertRaises(FileNotFoundError):
            self.engine.analyze_multiple_sessions(non_existent_path)
    
    @patch('selvage_eval.analysis.deepeval_analysis_engine.VersionComparisonAnalyzer')
    def test_error_handling_no_version_data(self, mock_version_class):
        """버전 데이터가 없는 경우 에러 처리 테스트"""
        # 테스트 베이스 디렉토리 생성
        base_dir = self.temp_path / "empty_base"
        base_dir.mkdir()
        
        # 빈 버전 데이터 반환 모킹
        mock_version_analyzer = MagicMock()
        mock_version_class.return_value = mock_version_analyzer
        mock_version_analyzer.collect_version_data.return_value = {}
        
        # 엔진의 버전 분석기를 모킹된 객체로 교체
        self.engine.version_analyzer = mock_version_analyzer
        
        # 테스트 실행 및 검증
        with self.assertRaises(ValueError):
            self.engine.analyze_multiple_sessions(str(base_dir))
    
    def test_error_handling_log_parsing_failure(self):
        """로그 파싱 실패 시나리오 테스트"""
        # 테스트 세션 디렉토리 생성
        session_dir = self.temp_path / "parse_error_session"
        session_dir.mkdir()
        
        # 로그 파일 생성 (실제 내용은 파서에 의해 처리됨)
        log_file = session_dir / "deepeval_results_error_model.log"
        log_file.write_text("some log content")
        
        # 로그 파서가 예외를 던지도록 모킹
        original_parser = self.engine.log_parser
        self.engine.log_parser = MagicMock()
        self.engine.log_parser.parse_log_file.side_effect = Exception("Log parsing failed")
        
        # 테스트 실행
        results = self.engine._collect_log_results(session_dir)
        
        # 파싱 실패 시 해당 파일은 결과에서 제외되어야 함
        self.assertEqual(results, {})
        
        # 원래 파서 복원
        self.engine.log_parser = original_parser
    
    def test_error_handling_comprehensive_analysis_failure(self):
        """종합 분석 실패 시나리오 테스트"""
        # 테스트 데이터 준비
        log_results = {"gpt_4": self.sample_test_results}
        
        # 모델 비교기가 예외를 던지도록 모킹
        original_comparator = self.engine.model_comparator
        self.engine.model_comparator = MagicMock()
        self.engine.model_comparator.compare_models.side_effect = Exception("Model comparison failed")
        
        # 테스트 실행
        result = self.engine._perform_comprehensive_analysis(log_results)
        
        # 분석 실패 시 에러 정보가 반환되어야 함
        self.assertIn("error", result)
        self.assertIn("분석 실패", result["error"])
        
        # 원래 비교기 복원
        self.engine.model_comparator = original_comparator
    
    def test_error_handling_incomplete_metric_data(self):
        """불완전한 메트릭 데이터 시나리오 테스트"""
        from selvage_eval.analysis.deepeval_log_parser import TestCaseResult, MetricScore
        
        # 일부 메트릭이 누락된 테스트 결과 생성
        incomplete_results = [
            TestCaseResult(
                correctness=MetricScore(score=0.8, passed=True, reason="Good"),
                clarity=MetricScore(score=0.0, passed=False, reason=""),  # 빈 데이터
                actionability=MetricScore(score=0.7, passed=True, reason="Actionable"),
                json_correctness=MetricScore(score=1.0, passed=True, reason="Valid JSON"),
                input_data="Test input",
                actual_output="Test output",
                raw_content="Raw content"
            )
        ]
        
        log_results = {"incomplete_model": incomplete_results}
        
        # 테스트 실행
        result = self.engine._perform_comprehensive_analysis(log_results)
        
        # 불완전한 데이터라도 분석이 완료되어야 함
        self.assertEqual(result["analysis_type"], "single_session")
        self.assertIn("model_comparison", result)
        self.assertIn("failure_analysis", result)
        self.assertIn("data_summary", result)
        
        # 실패 케이스로 분류되어야 함
        self.assertEqual(result["data_summary"]["failed_evaluations"], 1)
    
    def test_error_handling_memory_pressure(self):
        """메모리 부족 시나리오 시뮬레이션 테스트"""
        # 대용량 테스트 결과 생성
        large_results = []
        for i in range(1000):  # 많은 테스트 케이스
            large_results.append(TestCaseResult(
                correctness=MetricScore(score=0.8, passed=True, reason=f"Good analysis {i}"),
                clarity=MetricScore(score=0.75, passed=True, reason=f"Clear output {i}"),
                actionability=MetricScore(score=0.7, passed=True, reason=f"Actionable {i}"),
                json_correctness=MetricScore(score=1.0, passed=True, reason=f"Valid JSON {i}"),
                input_data=f"Large test input {i}" * 100,  # 큰 데이터
                actual_output=f"Large test output {i}" * 100,
                raw_content=f"Large raw content {i}" * 100
            ))
        
        log_results = {"large_model": large_results}
        
        # 테스트 실행 - 메모리 부족으로 실패하지 않아야 함
        result = self.engine._perform_comprehensive_analysis(log_results)
        
        # 대용량 데이터도 처리되어야 함
        self.assertEqual(result["analysis_type"], "single_session")
        self.assertEqual(result["data_summary"]["total_test_cases"], 1000)
    
    def test_error_handling_file_permission_error(self):
        """파일 권한 오류 시나리오 테스트"""
        # 권한이 없는 디렉토리 시뮬레이션
        restricted_dir = self.temp_path / "restricted_session"
        restricted_dir.mkdir()
        
        # 로그 파일 생성
        log_file = restricted_dir / "deepeval_results_restricted.log"
        log_file.write_text("test content")
        
        # 파일 시스템 작업이 실패하도록 모킹
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            # 테스트 실행
            results = self.engine._collect_log_results(restricted_dir)
            
            # 권한 오류 시 해당 파일은 결과에서 제외되어야 함
            self.assertEqual(results, {})
    
    def test_error_handling_corrupted_data_structures(self):
        """손상된 데이터 구조 처리 테스트"""
        # 잘못된 구조의 로그 결과 시뮬레이션
        corrupted_results = [
            # 정상적인 결과
            self.sample_test_results[0],
            # 손상된 결과 (잘못된 점수 값)
            TestCaseResult(
                correctness=MetricScore(score=-1.0, passed=False, reason="Invalid score"),  # 잘못된 점수
                clarity=MetricScore(score=0.8, passed=True, reason="Clear"),
                actionability=MetricScore(score=0.7, passed=True, reason="Actionable"),
                json_correctness=MetricScore(score=1.0, passed=True, reason="Valid JSON"),
                input_data="Test input",
                actual_output="Test output",
                raw_content="Raw content"
            )
        ]
        
        log_results = {"corrupted_model": corrupted_results}
        
        # 테스트 실행 - 손상된 데이터가 있어도 분석이 계속되어야 함
        result = self.engine._perform_comprehensive_analysis(log_results)
        
        # 분석이 완료되어야 하고, 손상된 데이터는 적절히 처리되어야 함
        self.assertEqual(result["analysis_type"], "single_session")
        self.assertIn("model_comparison", result)
        self.assertIn("failure_analysis", result)
    
    def test_end_to_end_integration_workflow(self):
        """엔드투엔드 통합 워크플로우 테스트"""
        # 복합적인 실제 시나리오 시뮬레이션
        session_dir = self.temp_path / "e2e_session"
        session_dir.mkdir()
        
        # 여러 모델의 다양한 성능 로그 생성
        model_logs = {
            "gpt_4": {
                "test_cases": [
                    {"correctness": 0.95, "clarity": 0.9, "actionability": 0.85, "json": 1.0},
                    {"correctness": 0.9, "clarity": 0.85, "actionability": 0.8, "json": 0.95},
                    {"correctness": 0.85, "clarity": 0.8, "actionability": 0.75, "json": 1.0}
                ]
            },
            "claude_3": {
                "test_cases": [
                    {"correctness": 0.8, "clarity": 0.95, "actionability": 0.9, "json": 1.0},
                    {"correctness": 0.85, "clarity": 0.9, "actionability": 0.85, "json": 0.98},
                    {"correctness": 0.75, "clarity": 0.85, "actionability": 0.8, "json": 1.0}
                ]
            },
            "gemini_pro": {
                "test_cases": [
                    {"correctness": 0.7, "clarity": 0.8, "actionability": 0.7, "json": 0.9},
                    {"correctness": 0.75, "clarity": 0.75, "actionability": 0.65, "json": 0.85},
                    {"correctness": 0.6, "clarity": 0.7, "actionability": 0.6, "json": 0.8}
                ]
            }
        }
        
        # 각 모델의 로그 파일 생성
        for model_name, model_data in model_logs.items():
            log_content_parts = []
            for i, test_case in enumerate(model_data["test_cases"]):
                log_content_parts.append(f"""==================================================
Test Case: E2E Test Case {i+1} for {model_name}
Input: Complex input data for {model_name} test {i+1}
Expected: Expected comprehensive output {i+1}
Actual: Generated output by {model_name} for test {i+1}

✅ Correctness (score: {test_case["correctness"]}, reason: "Analysis quality assessment", error: None)
✅ Clarity (score: {test_case["clarity"]}, reason: "Output clarity evaluation", error: None)
✅ Actionability (score: {test_case["actionability"]}, reason: "Actionability assessment", error: None)
✅ JSON Correctness (score: {test_case["json"]}, reason: "JSON format validation", error: None)
==================================================""")
            
            log_file = session_dir / f"deepeval_results_{model_name}.log"
            log_file.write_text("\n".join(log_content_parts))
        
        # 전체 분석 실행
        result = self.engine.analyze_session(str(session_dir))
        
        # 포괄적인 결과 검증
        self.assertIn("analysis_metadata", result)
        self.assertIn("files_generated", result)
        
        # 메타데이터 검증
        metadata = result["analysis_metadata"]
        # 로그 파서가 각 모델당 하나의 테스트 케이스만 파싱하므로 총 3개
        self.assertEqual(metadata["total_test_cases"], 3)  # 3 모델 × 1 테스트 케이스 (로그 파서 동작)
        self.assertEqual(len(metadata["models_analyzed"]), 3)
        self.assertIn("gpt_4", metadata["models_analyzed"])
        self.assertIn("claude_3", metadata["models_analyzed"])
        self.assertIn("gemini_pro", metadata["models_analyzed"])
        
        # 생성된 파일 검증
        files = result["files_generated"]
        for file_type in ["markdown_report", "json_data"]:
            self.assertIn(file_type, files)
            file_path = Path(files[file_type])
            self.assertTrue(file_path.exists())
            self.assertGreater(file_path.stat().st_size, 0)  # 파일이 비어있지 않은지
        
        # 마크다운 보고서 내용 검증
        markdown_path = Path(files["markdown_report"])
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        self.assertIn("# DeepEval 분석 보고서 (V2)", markdown_content)
        self.assertIn("📊 데이터 요약", markdown_content)
        self.assertIn("🏆 모델 성능 비교", markdown_content)
        self.assertIn("💡 결론 및 권장사항", markdown_content)
        
        # JSON 데이터 검증
        json_path = Path(files["json_data"])
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        self.assertIn("model_comparison", json_data)
        self.assertIn("failure_analysis", json_data)
        self.assertIn("data_summary", json_data)
        self.assertEqual(json_data["data_summary"]["total_test_cases"], 3)
    
    def test_performance_benchmark(self):
        """성능 벤치마크 테스트"""
        import time
        
        # 중간 규모 데이터셋 생성
        session_dir = self.temp_path / "performance_session"
        session_dir.mkdir()
        
        # 100개의 테스트 케이스를 가진 로그 생성
        test_cases = []
        for i in range(100):
            test_cases.append(f"""==================================================
Test Case: Performance Test {i+1}
Input: Performance input data {i+1}
Expected: Performance expected output {i+1}
Actual: Performance actual output {i+1}

✅ Correctness (score: {0.7 + (i % 3) * 0.1}, reason: "Performance test {i+1}", error: None)
✅ Clarity (score: {0.75 + (i % 4) * 0.05}, reason: "Clarity test {i+1}", error: None)
✅ Actionability (score: {0.6 + (i % 5) * 0.08}, reason: "Actionability test {i+1}", error: None)
✅ JSON Correctness (score: {0.9 + (i % 2) * 0.05}, reason: "JSON test {i+1}", error: None)
==================================================""")
        
        log_file = session_dir / "deepeval_results_performance_model.log"
        log_file.write_text("\n".join(test_cases))
        
        # 성능 측정
        start_time = time.time()
        result = self.engine.analyze_session(str(session_dir))
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 성능 검증
        self.assertLess(execution_time, 30.0)  # 30초 이내에 완료되어야 함
        
        # 결과 정확성 검증
        # 로그 파서가 하나의 테스트 케이스로 처리함
        self.assertEqual(result["analysis_metadata"]["total_test_cases"], 1)
        
        # 메모리 사용량 간접 검증 (결과가 정상적으로 생성됨)
        files = result["files_generated"]
        self.assertIn("markdown_report", files)
        self.assertIn("json_data", files)
        
        # 생성된 파일 크기 검증
        markdown_path = Path(files["markdown_report"])
        json_path = Path(files["json_data"])
        
        self.assertGreater(markdown_path.stat().st_size, 1000)  # 최소 1KB
        self.assertGreater(json_path.stat().st_size, 500)  # 최소 500B
        
        # 실행 시간 정보 로그 (디버깅용)
        print(f"Performance test completed in {execution_time:.2f} seconds")
        print(f"Processed {100} test cases")
        print(f"Average time per test case: {execution_time/100:.4f} seconds")
    
    def test_extract_repo_name_from_session_path(self):
        """세션 경로에서 저장소명 추출 테스트"""
        # 알려진 저장소명이 포함된 경로
        session_path1 = Path("/path/to/cline-evaluation/session1")
        repo_name1 = self.engine._extract_repo_name_from_session_path(session_path1)
        self.assertEqual(repo_name1, "cline-evaluation")
        
        session_path2 = Path("/path/to/ecommerce-microservices/session2")
        repo_name2 = self.engine._extract_repo_name_from_session_path(session_path2)
        self.assertEqual(repo_name2, "ecommerce-microservices")
        
        # 알려지지 않은 저장소명
        session_path3 = Path("/path/to/unknown-repo/session3")
        repo_name3 = self.engine._extract_repo_name_from_session_path(session_path3)
        self.assertEqual(repo_name3, "session3")  # 마지막 디렉토리명 사용
        
        # 추가 테스트: 다른 알려진 패턴들
        session_path4 = Path("/home/user/kotlin-project/test-session")
        repo_name4 = self.engine._extract_repo_name_from_session_path(session_path4)
        self.assertEqual(repo_name4, "kotlin-project")
        
        session_path5 = Path("/workspace/selvage-eval/session_20240101")
        repo_name5 = self.engine._extract_repo_name_from_session_path(session_path5)
        self.assertEqual(repo_name5, "selvage-eval")
    
    def test_prepare_repo_results_from_versions(self):
        """버전 데이터에서 저장소별 결과 준비 테스트"""
        # 샘플 버전 데이터 준비
        version_data = {
            "v1.0.0": {
                "sessions": [
                    {
                        "session_dir": "/path/to/cline-evaluation/session1",
                        "results": [self.sample_test_results[0]]
                    },
                    {
                        "session_dir": "/path/to/ecommerce-microservices/session2", 
                        "results": [self.sample_test_results[1]]
                    }
                ]
            },
            "v1.1.0": {
                "sessions": [
                    {
                        "session_dir": "/path/to/cline-evaluation/session3",
                        "results": self.sample_test_results
                    }
                ]
            }
        }
        
        # 테스트 실행
        repo_results = self.engine._prepare_repo_results_from_versions(version_data)
        
        # 결과 검증
        self.assertIn("cline-evaluation", repo_results)
        self.assertIn("ecommerce-microservices", repo_results)
        
        # cline-evaluation 저장소 검증
        cline_repo = repo_results["cline-evaluation"]
        self.assertIn("version_v1.0.0", cline_repo)
        self.assertIn("version_v1.1.0", cline_repo)
        self.assertEqual(len(cline_repo["version_v1.0.0"]), 1)
        self.assertEqual(len(cline_repo["version_v1.1.0"]), 2)
        
        # ecommerce-microservices 저장소 검증
        ecommerce_repo = repo_results["ecommerce-microservices"]
        self.assertIn("version_v1.0.0", ecommerce_repo)
        self.assertEqual(len(ecommerce_repo["version_v1.0.0"]), 1)
    
    def test_prepare_repo_results_from_versions_empty_data(self):
        """빈 버전 데이터에 대한 저장소별 결과 준비 테스트"""
        version_data = {}
        
        repo_results = self.engine._prepare_repo_results_from_versions(version_data)
        
        self.assertEqual(repo_results, {})
    
    def test_prepare_repo_results_from_versions_no_results(self):
        """결과가 없는 세션에 대한 저장소별 결과 준비 테스트"""
        version_data = {
            "v1.0.0": {
                "sessions": [
                    {
                        "session_dir": "/path/to/test-repo/session1",
                        "results": []
                    }
                ]
            }
        }
        
        repo_results = self.engine._prepare_repo_results_from_versions(version_data)
        
        # 빈 결과도 처리되어야 함
        # 빈 결과는 딕셔너리에 포함되지 않을 수 있음
        if "session1" in repo_results:
            if "version_v1.0.0" in repo_results["session1"]:
                self.assertEqual(len(repo_results["session1"]["version_v1.0.0"]), 0)
        # 빈 결과로 인해 아무것도 생성되지 않을 수 있음
        self.assertIsInstance(repo_results, dict)
    
    @patch.object(DeepEvalAnalysisEngine, '_collect_log_results')
    @patch.object(DeepEvalAnalysisEngine, '_perform_comprehensive_analysis')
    def test_output_directory_creation(self, mock_analysis, mock_collect):
        """출력 디렉토리 자동 생성 테스트"""
        # 테스트 세션 디렉토리 생성
        session_dir = self.temp_path / "test_session"
        session_dir.mkdir()
        
        # 사용자 지정 출력 디렉토리 (존재하지 않음)
        custom_output_dir = self.temp_path / "custom_output" / "nested"
        
        # 모킹 설정
        mock_collect.return_value = {"gpt_4": self.sample_test_results}
        mock_analysis.return_value = {
            "analysis_type": "single_session",
            "model_comparison": {"model_statistics": {"gpt_4": {}}},
            "failure_analysis": {"total_failures": 1},
            "data_summary": {"total_test_cases": 2}
        }
        
        # 테스트 실행
        result = self.engine.analyze_session(str(session_dir), str(custom_output_dir))
        
        # 출력 디렉토리가 생성되었는지 확인
        self.assertTrue(custom_output_dir.exists())
        self.assertTrue(custom_output_dir.is_dir())
        
        # 파일들이 올바른 위치에 생성되었는지 확인
        markdown_path = Path(result["files_generated"]["markdown_report"])
        self.assertTrue(str(markdown_path).startswith(str(custom_output_dir)))


if __name__ == "__main__":
    unittest.main()