"""
DeepEvalExecutorTool 통합 테스트
"""
import pytest
import tempfile
import os
from pathlib import Path

from selvage_eval.tools.deepeval_executor_tool import DeepEvalExecutorTool


class TestDeepEvalExecutorIntegration:
    """DeepEvalExecutorTool 통합 테스트 클래스"""
    
    def test_executor_tool_execute(self):
        """DeepEval executor tool 실행 테스트"""
        # DeepEval 텔레메트리 디렉토리 생성
        deepeval_dir = Path(".deepeval")
        deepeval_dir.mkdir(exist_ok=True)
        telemetry_file = deepeval_dir / ".deepeval_telemetry.txt"
        if not telemetry_file.exists():
            telemetry_file.touch()
        
        try:
            # DeepEvalExecutorTool 인스턴스 생성
            executor_tool = DeepEvalExecutorTool()
            
            # 지정된 session_id로 execute 호출
            session_id = 'eval_20250707_174243_e4df05f6'
            execution_result = executor_tool.execute(
                session_id=session_id,
                parallel_workers=1,
                display_filter="all"
            )
            
            # 결과 검증
            assert execution_result is not None
            print(f"Execution result: {execution_result}")
            
            # 결과가 툴 결과 형태인지 확인
            assert hasattr(execution_result, 'success')
            
            # 성공 여부 출력
            if execution_result.success:
                print("✓ DeepEval 평가 실행 성공")
                
                # 개별 평가 결과 확인
                if execution_result.data and 'evaluation_results' in execution_result.data:
                    eval_results = execution_result.data['evaluation_results']
                    for model_name, result in eval_results.items():
                        if result['success']:
                            print(f"  ✓ {model_name}: 평가 성공")
                        else:
                            print(f"  ✗ {model_name}: {result.get('error', '알 수 없는 오류')}")
            else:
                print(f"✗ DeepEval 평가 실행 실패: {execution_result.error_message}")
                
        except Exception as e:
            pytest.fail(f"테스트 실행 중 예외 발생: {e}")
    
 