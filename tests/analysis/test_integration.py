"""분석 엔진 통합 테스트"""

import json
import tempfile
import unittest
from pathlib import Path

from selvage_eval.analysis import DeepEvalAnalysisEngine


class TestAnalysisIntegration(unittest.TestCase):
    """분석 엔진 통합 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.engine = DeepEvalAnalysisEngine(output_dir=self.temp_dir)
        
        # 임시 세션 디렉토리 생성
        self.session_dir = Path(self.temp_dir) / "test_session"
        self.session_dir.mkdir(exist_ok=True)
        
        # 샘플 DeepEval 로그 파일 생성 (V2 엔진용)
        sample_log_content = """==================================================
Test Case: Sample test case
Input: Sample input data
Expected: Expected output
Actual: Actual output

✅ Correctness (score: 0.85, reason: "Good analysis", error: None)
✅ Clarity (score: 0.90, reason: "Clear explanations", error: None)
✅ Actionability (score: 0.78, reason: "Actionable suggestions", error: None)
✅ JSON Correctness (score: 1.0, reason: "Valid JSON format", error: None)
=================================================="""
        
        # 여러 모델 로그 파일 생성
        models = ["gemini-2.5-pro", "claude-sonnet-4", "gpt-4"]
        for model in models:
            file_path = self.session_dir / f"deepeval_results_{model}.log"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(sample_log_content)
    
    def test_full_analysis_workflow(self):
        """전체 분석 워크플로우 테스트"""
        # 분석 실행
        result = self.engine.analyze_session(str(self.session_dir))
        
        # 결과 검증
        self.assertIn("analysis_metadata", result)
        self.assertIn("files_generated", result)
        
        # 생성된 파일들 확인
        files_generated = result["files_generated"]
        
        # 마크다운 보고서 확인
        markdown_path = Path(files_generated["markdown_report"])
        self.assertTrue(markdown_path.exists())
        
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        self.assertIn("# DeepEval 분석 보고서 (V2)", markdown_content)
        self.assertIn("gemini-2.5-pro", markdown_content)
        self.assertIn("claude-sonnet-4", markdown_content)
        self.assertIn("gpt-4", markdown_content)
        
        # JSON 데이터 확인
        json_path = Path(files_generated["json_data"])
        self.assertTrue(json_path.exists())
        
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        self.assertIn("model_comparison", json_data)
        self.assertIn("data_summary", json_data)
        
        # 메타데이터 확인
        metadata = result["analysis_metadata"]
        self.assertEqual(len(metadata["models_analyzed"]), 3)
        self.assertIn("gemini-2.5-pro", metadata["models_analyzed"])
    
    def test_empty_session_handling(self):
        """빈 세션 디렉토리 처리 테스트"""
        empty_dir = Path(self.temp_dir) / "empty_session"
        empty_dir.mkdir(exist_ok=True)
        
        # 빈 디렉토리에서 분석 시도
        with self.assertRaises(ValueError) as context:
            self.engine.analyze_session(str(empty_dir))
        
        self.assertIn("DeepEval 로그 결과를 찾을 수 없습니다", str(context.exception))
    
    def test_nonexistent_session_handling(self):
        """존재하지 않는 세션 디렉토리 처리 테스트"""
        nonexistent_path = "/nonexistent/path/session"
        
        with self.assertRaises(FileNotFoundError) as context:
            self.engine.analyze_session(nonexistent_path)
        
        self.assertIn("세션 경로가 존재하지 않습니다", str(context.exception))


if __name__ == "__main__":
    unittest.main()