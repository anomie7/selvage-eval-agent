"""
DeepEvalAnalysisEngine 통합 테스트
"""
import pytest
import os
from pathlib import Path

from selvage_eval.analysis.deepeval_analysis_engine import DeepEvalAnalysisEngine


class TestDeepEvalAnalysisEngineIntegration:
    """DeepEvalAnalysisEngine 통합 테스트 클래스"""
    
    def test_analyze_session_with_default_paths(self):
        """DeepEval 분석 엔진 기본 경로 테스트"""
        # DeepEval 텔레메트리 디렉토리 생성
        deepeval_dir = Path(".deepeval")
        deepeval_dir.mkdir(exist_ok=True)
        telemetry_file = deepeval_dir / ".deepeval_telemetry.txt"
        if not telemetry_file.exists():
            telemetry_file.touch()
        
        try:
            # DeepEvalAnalysisEngine 인스턴스 생성
            analysis_engine = DeepEvalAnalysisEngine()
            
            # 지정된 session_id로 analyze_session 호출 (기본 경로 사용)
            session_id = 'eval_20250708_012833_39bb3ebc'
            
            # 기본 경로 확인
            expected_deepeval_path = Path("/Users/demin_coder/Library/selvage-eval/deepeval_results") / session_id
            assert expected_deepeval_path.exists(), f"DeepEval 결과 경로가 존재하지 않습니다: {expected_deepeval_path}"
            
            # 분석 실행 (deepeval_results_path와 output_dir 모두 기본값 사용)
            result = analysis_engine.analyze_session(session_id)
            
            # 결과 검증
            assert result is not None, "분석 결과가 None입니다"
            assert "analysis_metadata" in result, "분석 메타데이터가 없습니다"
            assert "files_generated" in result, "생성된 파일 정보가 없습니다"
            
            # 메타데이터 검증
            metadata = result["analysis_metadata"]
            assert "analysis_timestamp" in metadata, "분석 시간이 없습니다"
            assert "session_path" in metadata, "세션 경로가 없습니다"
            assert "total_test_cases" in metadata, "총 테스트 케이스 수가 없습니다"
            assert "models_analyzed" in metadata, "분석된 모델 목록이 없습니다"
            
            # 생성된 파일 검증
            files_generated = result["files_generated"]
            assert "markdown_report" in files_generated, "마크다운 보고서 경로가 없습니다"
            assert "json_data" in files_generated, "JSON 데이터 경로가 없습니다"
            assert "visualization_files" in files_generated, "시각화 파일 목록이 없습니다"
            
            # 실제 파일 존재 확인
            markdown_path = Path(files_generated["markdown_report"])
            json_path = Path(files_generated["json_data"])
            
            assert markdown_path.exists(), f"마크다운 보고서 파일이 존재하지 않습니다: {markdown_path}"
            assert json_path.exists(), f"JSON 데이터 파일이 존재하지 않습니다: {json_path}"
            
            # 기본 출력 디렉토리 확인
            expected_output_dir = Path("~/Library/selvage-eval/analyze_results").expanduser() / session_id
            assert markdown_path.parent == expected_output_dir, f"출력 디렉토리가 예상과 다릅니다: {markdown_path.parent} vs {expected_output_dir}"
            
            # 파일 내용 기본 검증
            assert markdown_path.stat().st_size > 0, "마크다운 보고서가 비어있습니다"
            assert json_path.stat().st_size > 0, "JSON 데이터가 비어있습니다"
            
            # 마크다운 보고서 기본 구조 확인
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
                assert "DeepEval 분석 보고서" in markdown_content, "마크다운 보고서 제목이 없습니다"
                assert "데이터 요약" in markdown_content, "데이터 요약 섹션이 없습니다"
                assert "분석 시간" in markdown_content, "분석 시간 정보가 없습니다"
            
            # 시각화 디렉토리 확인 (생성 시도되었는지)
            visualization_dir = expected_output_dir / "visualizations"
            if visualization_dir.exists():
                assert visualization_dir.is_dir(), "시각화 디렉토리가 디렉토리가 아닙니다"
            
            print(f"✅ 분석 성공!")
            print(f"📊 총 테스트 케이스: {metadata['total_test_cases']}")
            print(f"🤖 분석된 모델: {metadata['models_analyzed']}")
            print(f"📝 마크다운 보고서: {markdown_path}")
            print(f"📄 JSON 데이터: {json_path}")
            print(f"📂 출력 디렉토리: {expected_output_dir}")
            
        finally:
            # 임시 파일 정리
            if telemetry_file.exists():
                telemetry_file.unlink()
            if deepeval_dir.exists() and not any(deepeval_dir.iterdir()):
                deepeval_dir.rmdir()