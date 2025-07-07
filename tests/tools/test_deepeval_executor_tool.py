"""DeepEvalExecutorTool 단위 테스트

DeepEval 평가 실행 도구를 테스트합니다.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch
from pathlib import Path

from selvage_eval.tools.deepeval_executor_tool import (
    DeepEvalExecutorTool, IssueSeverityEnum
)


# 테스트 마커 정의
pytestmark = [
    pytest.mark.unit,
]


@pytest.fixture
def temp_base_dir():
    """모든 테스트용 기본 임시 디렉토리"""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)
        
        # 표준 디렉토리 구조 생성
        (base_path / "deep_eval_test_case").mkdir(exist_ok=True)
        (base_path / "deepeval_results").mkdir(exist_ok=True)
        
        yield base_path


@pytest.fixture
def sample_session_id():
    """샘플 세션 ID"""
    return "test-session-2024-01-15"


@pytest.fixture
def sample_test_cases_data():
    """샘플 테스트 케이스 데이터"""
    return [
        {
            "input": '["test prompt"]',
            "actual_output": '{"issues": [], "summary": "No issues found"}',
            "expected_output": None,
            "repo_name": "test-repo"
        },
        {
            "input": '["another prompt"]',
            "actual_output": '{"issues": [{"type": "bug", "severity": "error"}], "summary": "Found issues"}',
            "expected_output": None,
            "repo_name": "test-repo"
        }
    ]


@pytest.fixture
def test_cases_structure(temp_base_dir, sample_session_id, sample_test_cases_data):
    """테스트 케이스 디렉토리 구조 생성"""
    base_dir = temp_base_dir / "deep_eval_test_case" / sample_session_id
    
    # 모델별 디렉토리 및 테스트 케이스 파일 생성
    models = ["gemini-2.5-pro", "claude-3-sonnet"]
    
    for model_name in models:
        model_dir = base_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # test_cases.json 파일 생성
        test_cases_file = model_dir / "test_cases.json"
        with open(test_cases_file, 'w', encoding='utf-8') as f:
            json.dump(sample_test_cases_data, f, ensure_ascii=False, indent=2)
    
    # 메타데이터 파일 생성
    metadata = {
        "selvage_version": "0.1.2",
        "execution_date": "2024-01-15T10:30:00.123456",
        "tool_name": "deepeval_test_case_converter",
        "created_by": "DeepEvalTestCaseConverterTool",
        "evaluation_framework": "DeepEval"
    }
    metadata_file = base_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return base_dir


@pytest.mark.unit
class TestDeepEvalExecutorTool:
    """DeepEvalExecutorTool 테스트"""
    
    def test_tool_properties(self):
        """Tool 기본 속성 테스트"""
        tool = DeepEvalExecutorTool()
        
        assert tool.name == "deepeval_executor"
        assert "DeepEval을 사용하여 코드 리뷰 품질을 평가합니다" in tool.description
        
        schema = tool.parameters_schema
        assert schema["type"] == "object"
        assert "session_id" in schema["properties"]
        assert "session_id" in schema["required"]
    
    def test_validate_parameters_valid(self):
        """유효한 파라미터 검증 테스트"""
        tool = DeepEvalExecutorTool()
        
        # 필수 파라미터만
        params = {"session_id": "test-session"}
        assert tool.validate_parameters(params) is True
        
        # 모든 파라미터
        params = {
            "session_id": "test-session",
            "test_case_path": "/path/to/test/cases",
            "parallel_workers": 2,
            "display_filter": "all"
        }
        assert tool.validate_parameters(params) is True
    
    def test_validate_parameters_invalid(self):
        """유효하지 않은 파라미터 검증 테스트"""
        tool = DeepEvalExecutorTool()
        
        # session_id 누락
        params = {}
        assert tool.validate_parameters(params) is False
        
        # session_id 타입 오류
        params = {"session_id": 123}
        assert tool.validate_parameters(params) is False
        
        # session_id 빈 문자열
        params = {"session_id": ""}
        assert tool.validate_parameters(params) is False
        
        # parallel_workers 타입 오류
        params = {"session_id": "test", "parallel_workers": "invalid"}
        assert tool.validate_parameters(params) is False
        
        # parallel_workers 음수
        params = {"session_id": "test", "parallel_workers": -1}
        assert tool.validate_parameters(params) is False
    
    def test_check_environment_missing_vars(self):
        """환경 변수 누락 테스트"""
        tool = DeepEvalExecutorTool()
        
        with patch.dict(os.environ, {}, clear=True):
            result = tool._check_environment()
            assert result["valid"] is False
            assert "OPENAI_API_KEY" in result["message"]
            assert "GEMINI_API_KEY" in result["message"]
    
    def test_check_environment_valid(self):
        """유효한 환경 변수 테스트"""
        tool = DeepEvalExecutorTool()
        
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-openai-key",
            "GEMINI_API_KEY": "test-gemini-key"
        }):
            result = tool._check_environment()
            assert result["valid"] is True
            assert result["message"] == "환경 설정 완료"
    
    @patch('selvage_eval.tools.deepeval_executor_tool.get_selvage_version')
    def test_create_metadata(self, mock_get_version, sample_session_id):
        """메타데이터 생성 테스트"""
        mock_get_version.return_value = "0.1.2"
        
        tool = DeepEvalExecutorTool()
        test_case_path = "/test/path"
        
        metadata = tool._create_metadata(sample_session_id, test_case_path)
        
        assert metadata["selvage_version"] == "0.1.2"
        assert metadata["session_id"] == sample_session_id
        assert metadata["deep_eval_test_case_path"] == test_case_path
        assert metadata["tool_name"] == "deepeval_executor"
        assert metadata["created_by"] == "DeepEvalExecutorTool"
        assert "execution_date" in metadata
    
    def test_load_test_cases_no_deepeval(self, temp_base_dir, sample_test_cases_data):
        """DeepEval이 없는 경우 테스트 케이스 로드 테스트"""
        tool = DeepEvalExecutorTool()
        
        # 테스트 케이스 파일 생성
        test_file = temp_base_dir / "test_cases.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(sample_test_cases_data, f)
        
        # _load_test_cases 메서드 내에서 ImportError가 발생하는 상황을 시뮬레이션
        with patch('builtins.__import__', side_effect=ImportError("No module named 'deepeval'")):
            result = tool._load_test_cases(test_file)
            assert result == []
    
    def test_load_test_cases_invalid_file(self, temp_base_dir):
        """유효하지 않은 파일로 테스트 케이스 로드 테스트"""
        tool = DeepEvalExecutorTool()
        
        # 존재하지 않는 파일
        non_existent_file = temp_base_dir / "non_existent.json"
        
        result = tool._load_test_cases(non_existent_file)
        assert result == []
    
    def test_create_metrics_no_deepeval(self):
        """DeepEval이 없는 경우 메트릭 생성 테스트"""
        tool = DeepEvalExecutorTool()
        
        # _create_metrics 메서드 내에서 ImportError가 발생하는 상황을 시뮬레이션
        with patch('builtins.__import__', side_effect=ImportError("No module named 'deepeval'")):
            metrics = tool._create_metrics()
            assert metrics == []
    
    def test_convert_display_filter_to_enum(self):
        """display_filter 문자열을 enum으로 변환 테스트"""
        tool = DeepEvalExecutorTool()
        
        # Mock TestRunResultDisplay enum
        class MockTestRunResultDisplay:
            ALL = "all"
            FAILING = "failing"
            PASSING = "passing"
        
        # 정상적인 변환 테스트
        result = tool._convert_display_filter_to_enum("all", MockTestRunResultDisplay)
        assert result == MockTestRunResultDisplay.ALL
        
        result = tool._convert_display_filter_to_enum("failing", MockTestRunResultDisplay)
        assert result == MockTestRunResultDisplay.FAILING
        
        result = tool._convert_display_filter_to_enum("passing", MockTestRunResultDisplay)
        assert result == MockTestRunResultDisplay.PASSING
        
        # 대소문자 무관 테스트
        result = tool._convert_display_filter_to_enum("ALL", MockTestRunResultDisplay)
        assert result == MockTestRunResultDisplay.ALL
        
        result = tool._convert_display_filter_to_enum("Failing", MockTestRunResultDisplay)
        assert result == MockTestRunResultDisplay.FAILING
        
        # 알 수 없는 값은 기본값 반환
        result = tool._convert_display_filter_to_enum("unknown", MockTestRunResultDisplay)
        assert result == MockTestRunResultDisplay.ALL
    
    def test_run_evaluation_no_test_cases(self, temp_base_dir):
        """테스트 케이스가 없는 경우 평가 실행 테스트"""
        tool = DeepEvalExecutorTool()
        
        test_cases_file = temp_base_dir / "empty_test_cases.json"
        output_path = temp_base_dir / "output"
        output_path.mkdir()
        
        # 빈 테스트 케이스 파일 생성
        with open(test_cases_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        
        with patch.object(tool, '_load_test_cases', return_value=[]):
            result = tool._run_evaluation(test_cases_file, output_path, 1, "all")
            assert result["success"] is False
            assert "테스트 케이스를 로드할 수 없습니다" in result["error"]
    
    def test_run_evaluation_no_deepeval_import(self, temp_base_dir, sample_test_cases_data):
        """DeepEval import 실패 시 평가 실행 테스트"""
        tool = DeepEvalExecutorTool()
        
        test_cases_file = temp_base_dir / "test_cases.json"
        output_path = temp_base_dir / "output"
        output_path.mkdir()
        
        with open(test_cases_file, 'w', encoding='utf-8') as f:
            json.dump(sample_test_cases_data, f)
        
        # 테스트 케이스 로드는 성공하지만 evaluate import가 실패하는 상황
        mock_test_cases = [Mock()]
        with patch.object(tool, '_load_test_cases', return_value=mock_test_cases), \
             patch('builtins.__import__', side_effect=ImportError("No module named 'deepeval'")):
            result = tool._run_evaluation(test_cases_file, output_path, 1, "all")
            assert result["success"] is False
            assert "DeepEval 라이브러리를 가져올 수 없습니다" in result["error"]
    
    def test_execute_missing_environment(self, sample_session_id):
        """환경 변수 누락 시 실행 테스트"""
        tool = DeepEvalExecutorTool()
        
        with patch.dict(os.environ, {}, clear=True):
            result = tool.execute(session_id=sample_session_id)
            
            assert result.success is False
            assert result.error_message is not None
            assert "환경 설정 오류" in result.error_message
            assert "OPENAI_API_KEY" in result.error_message
    
    def test_execute_missing_test_cases_directory(self, temp_base_dir, sample_session_id):
        """테스트 케이스 디렉토리 누락 시 실행 테스트"""
        tool = DeepEvalExecutorTool()
        
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "GEMINI_API_KEY": "test-key"
        }):
            # 존재하지 않는 경로 설정
            non_existent_path = str(temp_base_dir / "non_existent_path")
            
            result = tool.execute(
                session_id=sample_session_id,
                test_case_path=non_existent_path
            )
            
            assert result.success is False
            assert result.error_message is not None
            assert "테스트 케이스 디렉토리를 찾을 수 없습니다" in result.error_message
    
    @patch('selvage_eval.tools.deepeval_executor_tool.get_selvage_version')
    def test_execute_successful_evaluation(self, mock_get_version, temp_base_dir, 
                                         sample_session_id):
        """성공적인 평가 실행 테스트"""
        mock_get_version.return_value = "0.1.2"
        
        tool = DeepEvalExecutorTool()
        
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "GEMINI_API_KEY": "test-key"
        }):
            test_case_path = str(temp_base_dir / "deep_eval_test_case")
            results_path = str(temp_base_dir / "deepeval_results")
            
            # 테스트 케이스 디렉토리 구조 생성
            session_dir = Path(test_case_path) / sample_session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # 더미 모델 디렉토리와 테스트 케이스 파일 생성
            model_dir = session_dir / "gemini-2.5-pro"
            model_dir.mkdir(parents=True, exist_ok=True)
            test_cases_file = model_dir / "test_cases.json"
            with open(test_cases_file, 'w', encoding='utf-8') as f:
                json.dump([{"input": "test", "actual_output": "test"}], f)
            
            # _run_evaluation 메서드를 모킹하여 성공적인 실행 시뮬레이션
            with patch.object(tool, '_run_evaluation', return_value={
                "success": True,
                "data": {"executed": True, "test_cases_count": 2}
            }):
                # 경로 설정을 위해 패치
                with patch.object(tool, 'test_case_base_path', test_case_path), \
                     patch.object(tool, 'results_base_path', results_path):
                    
                    result = tool.execute(session_id=sample_session_id)
                    
                    assert result.success is True
                    assert result.data["session_id"] == sample_session_id
                    assert result.data["total_evaluations"] == 1  # 실제로는 gemini-2.5-pro 하나만 생성
                    assert "evaluation_results" in result.data
                    assert "metadata_path" in result.data
                    
                    # 결과 디렉토리가 생성되었는지 확인
                    results_dir = Path(results_path) / sample_session_id
                    assert results_dir.exists()
                    assert (results_dir / "metadata.json").exists()


@pytest.mark.unit
class TestIssueSeverityEnum:
    """IssueSeverityEnum 테스트"""
    
    def test_severity_enum_values(self):
        """심각도 열거형 값 테스트"""
        assert IssueSeverityEnum.INFO == "info"
        assert IssueSeverityEnum.WARNING == "warning"
        assert IssueSeverityEnum.ERROR == "error"
    
    def test_severity_enum_usage(self):
        """심각도 열거형 사용 테스트"""
        severity = IssueSeverityEnum.ERROR
        assert severity.value == "error"
        assert severity == "error"


@pytest.mark.unit
class TestStructuredModels:
    """Structured 모델 클래스들 테스트"""
    
    @pytest.mark.skipif(True, reason="pydantic이 없는 경우를 대비한 더미 테스트")
    def test_structured_models_import(self):
        """Structured 모델 import 테스트"""
        from selvage_eval.tools.deepeval_executor_tool import (
            StructuredReviewIssue, StructuredReviewResponse
        )
        
        # 클래스가 정의되어 있는지만 확인
        assert StructuredReviewIssue is not None
        assert StructuredReviewResponse is not None