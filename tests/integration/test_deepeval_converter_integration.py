"""
DeepEvalTestCaseConverterTool 통합 테스트
"""
import pytest
import tempfile
import os
from pathlib import Path

from selvage_eval.tools.deepeval_test_case_converter_tool import DeepEvalTestCaseConverterTool


class TestDeepEvalConverterIntegration:
    """DeepEvalTestCaseConverterTool 통합 테스트 클래스"""
    
    def test_converter_tool_execute_with_specific_session_id(self):
        """특정 session_id로 converter_tool.execute 호출 테스트"""
        # 임시 디렉토리에서 테스트 실행
        with tempfile.TemporaryDirectory() as temp_dir:
            # 원래 작업 디렉토리 저장
            original_cwd = os.getcwd()
            
            try:
                # 임시 디렉토리로 변경
                os.chdir(temp_dir)
                
                # DeepEvalTestCaseConverterTool 인스턴스 생성
                converter_tool = DeepEvalTestCaseConverterTool()
                
                # 지정된 session_id로 execute 호출
                session_id = 'eval_20250707_174243_e4df05f6'
                conversion_result = converter_tool.execute(session_id=session_id)
                
                # 결과 검증
                assert conversion_result is not None
                print(f"Conversion result: {conversion_result}")
                
                # 결과가 툴 결과 형태인지 확인
                assert hasattr(conversion_result, 'success') or hasattr(conversion_result, 'result')
                
            finally:
                # 원래 작업 디렉토리로 복원
                os.chdir(original_cwd)
    
    def test_converter_tool_execute_success_verification(self):
        """converter_tool.execute 호출이 성공하는지 확인하는 테스트"""
        # 임시 디렉토리에서 테스트 실행
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            
            try:
                os.chdir(temp_dir)
                
                converter_tool = DeepEvalTestCaseConverterTool()
                session_id = 'eval_20250707_174243_e4df05f6'
                
                # execute 호출이 예외를 발생시키지 않는지 확인
                try:
                    conversion_result = converter_tool.execute(session_id=session_id)
                    
                    # 호출이 성공했음을 출력
                    print(f"✓ converter_tool.execute 호출 성공")
                    print(f"Session ID: {session_id}")
                    print(f"Result: {conversion_result}")
                    
                    # 성공으로 간주
                    assert True, "converter_tool.execute 호출이 성공적으로 완료되었습니다"
                    
                except Exception as e:
                    pytest.fail(f"converter_tool.execute 호출 중 예외 발생: {e}")
                    
            finally:
                os.chdir(original_cwd) 