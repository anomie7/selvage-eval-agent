"""DeepEvalTestCaseConverterTool 단위 테스트

DeepEval 테스트 케이스 변환 도구를 테스트합니다.
"""

import pytest
import json
import tempfile
import os
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from selvage_eval.tools.deepeval_test_case_converter_tool import (
    DeepEvalTestCaseConverterTool, ReviewLogInfo, DeepEvalTestCase, get_selvage_version
)
from selvage_eval.tools.tool_result import ToolResult


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
        (base_path / "review_logs").mkdir(exist_ok=True)
        (base_path / "deep_eval_test_case").mkdir(exist_ok=True)
        
        yield base_path


@pytest.fixture
def sample_session_id():
    """샘플 세션 ID"""
    return "test-session-2024-01-15"


@pytest.fixture
def sample_review_log_data():
    """샘플 리뷰 로그 데이터"""
    return {
        "prompt": [
            {"role": "system", "content": "You are a code reviewer."},
            {"role": "user", "content": "Please review this code change."}
        ],
        "review_response": {
            "issues": [
                {
                    "type": "bug",
                    "line_number": 15,
                    "file": "src/main.py",
                    "description": "Potential null pointer exception",
                    "suggestion": "Add null check before accessing object",
                    "severity": "error",
                    "target_code": "obj.method()",
                    "suggested_code": "if obj: obj.method()"
                }
            ],
            "summary": "Found 1 critical issue that needs attention",
            "score": 6.5,
            "recommendations": ["Add input validation", "Improve error handling"]
        },
        "metadata": {
            "model": "gemini-2.5-pro",
            "timestamp": "2024-01-15T10:30:00Z"
        }
    }


@pytest.fixture
def review_logs_structure(temp_base_dir, sample_review_log_data):
    """리뷰 로그 디렉토리 구조 생성"""
    base_dir = temp_base_dir / "review_logs"
    
    # 디렉토리 구조: repo_name/commit_id/model_name/
    repos_data = [
        ("test-repo-1", "abc123", "gemini-2.5-pro"),
        ("test-repo-1", "abc123", "claude-3-sonnet"),
        ("test-repo-2", "def456", "gemini-2.5-pro"),
    ]
    
    created_files = []
    
    for repo_name, commit_id, model_name in repos_data:
        model_dir = base_dir / repo_name / commit_id / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 리뷰 로그 파일 생성
        log_file = model_dir / f"review_log_{commit_id}_{model_name}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(sample_review_log_data, f, ensure_ascii=False, indent=2)
        
        created_files.append(ReviewLogInfo(
            repo_name=repo_name,
            commit_id=commit_id,
            model_name=model_name,
            file_path=log_file,
            file_name=log_file.name
        ))
    
    return created_files


@pytest.mark.unit
class TestDeepEvalTestCaseConverterTool:
    """DeepEvalTestCaseConverterTool 테스트"""
    
    def test_tool_properties(self):
        """Tool 기본 속성 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        assert tool.name == "deepeval_test_case_converter"
        assert "리뷰 로그를 DeepEval 테스트 케이스 형식으로 변환합니다" in tool.description
        
        schema = tool.parameters_schema
        assert schema["type"] == "object"
        assert "session_id" in schema["properties"]
        assert "session_id" in schema["required"]
    
    def test_validate_parameters_valid(self):
        """유효한 파라미터 검증 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # 필수 파라미터만
        params = {"session_id": "test-session"}
        assert tool.validate_parameters(params) is True
        
        # 모든 파라미터
        params = {
            "session_id": "test-session",
            "review_logs_path": "/path/to/logs",
            "output_path": "/path/to/output"
        }
        assert tool.validate_parameters(params) is True
    
    def test_validate_parameters_invalid(self):
        """유효하지 않은 파라미터 검증 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # session_id 누락
        params = {}
        assert tool.validate_parameters(params) is False
        
        # session_id 타입 오류
        params = {"session_id": 123}
        assert tool.validate_parameters(params) is False
        
        # session_id 빈 문자열
        params = {"session_id": ""}
        assert tool.validate_parameters(params) is False
        
        # 선택적 파라미터 타입 오류
        params = {"session_id": "test", "review_logs_path": 123}
        assert tool.validate_parameters(params) is False
    
    def test_scan_review_logs_empty_directory(self, temp_base_dir):
        """빈 디렉토리 스캔 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # 존재하지 않는 경로
        non_existent_path = temp_base_dir / "non_existent"
        result = tool._scan_review_logs(non_existent_path)
        assert result == []
        
        # 빈 디렉토리
        empty_dir = temp_base_dir / "empty"
        empty_dir.mkdir()
        result = tool._scan_review_logs(empty_dir)
        assert result == []
    
    def test_scan_review_logs_with_files(self, temp_base_dir, review_logs_structure):
        """파일이 있는 디렉토리 스캔 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        logs_path = temp_base_dir / "review_logs"
        result = tool._scan_review_logs(logs_path)
        
        assert len(result) == 3
        
        # 결과 검증
        repo_names = {log.repo_name for log in result}
        model_names = {log.model_name for log in result}
        
        assert repo_names == {"test-repo-1", "test-repo-2"}
        assert model_names == {"gemini-2.5-pro", "claude-3-sonnet"}
    
    def test_group_logs_by_model(self, review_logs_structure):
        """모델별 로그 그룹화 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        grouped = tool._group_logs_by_model(review_logs_structure)
        
        assert len(grouped) == 2
        assert "gemini-2.5-pro" in grouped
        assert "claude-3-sonnet" in grouped
        
        # gemini-2.5-pro는 2개 파일 (repo-1, repo-2)
        assert len(grouped["gemini-2.5-pro"]) == 2
        # claude-3-sonnet는 1개 파일 (repo-1만)
        assert len(grouped["claude-3-sonnet"]) == 1
    
    def test_extract_prompt_and_response(self, temp_base_dir, sample_review_log_data):
        """prompt와 response 추출 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # 정상 파일
        test_file = temp_base_dir / "test_log.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(sample_review_log_data, f)
        
        result = tool._extract_prompt_and_response(test_file)
        
        assert result is not None
        assert "prompt" in result
        assert "review_response" in result
        assert "original_data" in result
        assert result["prompt"] == sample_review_log_data["prompt"]
        assert result["review_response"] == sample_review_log_data["review_response"]
    
    def test_extract_prompt_and_response_invalid_file(self, temp_base_dir):
        """유효하지 않은 파일 처리 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # 존재하지 않는 파일
        non_existent_file = temp_base_dir / "non_existent.json"
        result = tool._extract_prompt_and_response(non_existent_file)
        assert result is None
        
        # 잘못된 JSON 파일
        invalid_json_file = temp_base_dir / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write("invalid json content")
        
        result = tool._extract_prompt_and_response(invalid_json_file)
        assert result is None
        
        # prompt/review_response 누락 파일
        incomplete_data = {"metadata": {"model": "test"}}
        incomplete_file = temp_base_dir / "incomplete.json"
        with open(incomplete_file, 'w', encoding='utf-8') as f:
            json.dump(incomplete_data, f)
        
        result = tool._extract_prompt_and_response(incomplete_file)
        assert result is None
    
    def test_convert_to_deepeval_format(self, sample_review_log_data):
        """DeepEval 형식 변환 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        extracted_data = {
            "prompt": sample_review_log_data["prompt"],
            "review_response": sample_review_log_data["review_response"],
            "original_data": sample_review_log_data
        }
        
        test_case = tool._convert_to_deepeval_format(extracted_data)
        
        assert isinstance(test_case, DeepEvalTestCase)
        assert test_case.input == json.dumps(sample_review_log_data["prompt"], ensure_ascii=False)
        assert test_case.actual_output == json.dumps(sample_review_log_data["review_response"], ensure_ascii=False)
        assert test_case.expected_output is None
    
    def test_test_case_to_dict(self):
        """테스트 케이스 딕셔너리 변환 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        test_case = DeepEvalTestCase(
            input="test input",
            actual_output="test output",
            expected_output="expected output"
        )
        
        result = tool._test_case_to_dict(test_case)
        
        assert result == {
            "input": "test input",
            "actual_output": "test output",
            "expected_output": "expected output"
        }
    
    @patch('selvage_eval.tools.deepeval_test_case_converter_tool.get_selvage_version')
    def test_create_metadata(self, mock_get_version):
        """메타데이터 생성 테스트"""
        mock_get_version.return_value = "0.1.2"
        
        tool = DeepEvalTestCaseConverterTool()
        metadata = tool._create_metadata()
        
        assert metadata["selvage_version"] == "0.1.2"
        assert metadata["tool_name"] == "deepeval_test_case_converter"
        assert metadata["created_by"] == "DeepEvalTestCaseConverterTool"
        assert metadata["evaluation_framework"] == "DeepEval"
        assert "execution_date" in metadata
    
    def test_execute_no_review_logs(self, temp_base_dir, sample_session_id):
        """리뷰 로그가 없는 경우 실행 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # 빈 디렉토리로 설정
        logs_path = str(temp_base_dir / "empty_logs")
        output_path = str(temp_base_dir / "deep_eval_test_case")
        
        result = tool.execute(
            session_id=sample_session_id,
            review_logs_path=logs_path,
            output_path=output_path
        )
        
        assert result.success is False
        assert result.error_message is not None
        assert "리뷰 로그 파일을 찾을 수 없습니다" in result.error_message
    
    @patch('selvage_eval.tools.deepeval_test_case_converter_tool.get_selvage_version')
    def test_execute_successful_conversion(self, mock_get_version, temp_base_dir, sample_session_id, review_logs_structure):
        """성공적인 변환 실행 테스트"""
        mock_get_version.return_value = "0.1.2"
        
        tool = DeepEvalTestCaseConverterTool()
        
        logs_path = str(temp_base_dir / "review_logs")
        output_path = str(temp_base_dir / "deep_eval_test_case")
        
        result = tool.execute(
            session_id=sample_session_id,
            review_logs_path=logs_path,
            output_path=output_path
        )
        
        assert result.success is True
        assert result.data["session_id"] == sample_session_id
        assert result.data["total_files"] == 2  # gemini-2.5-pro, claude-3-sonnet
        assert "converted_files" in result.data
        assert "metadata_path" in result.data
        
        # 파일이 실제로 생성되었는지 확인
        session_dir = Path(output_path) / sample_session_id
        assert session_dir.exists()
        assert (session_dir / "metadata.json").exists()
        
        # 모델별 파일 확인
        for model_name in ["gemini-2.5-pro", "claude-3-sonnet"]:
            model_dir = session_dir / model_name
            assert model_dir.exists()
            assert (model_dir / "test_cases.json").exists()


@pytest.mark.unit
class TestGetSelvageVersion:
    """get_selvage_version 함수 테스트"""
    
    @patch('subprocess.run')
    def test_get_selvage_version_success(self, mock_run):
        """성공적인 버전 확인 테스트"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "selvage 0.1.2\n"
        mock_run.return_value = mock_result
        
        version = get_selvage_version()
        assert version == "selvage 0.1.2"
        
        mock_run.assert_called_once_with(
            ["/Users/demin_coder/.local/bin/selvage", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
    
    @patch('subprocess.run')
    def test_get_selvage_version_failure(self, mock_run):
        """버전 확인 실패 테스트"""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        version = get_selvage_version()
        assert version == "unknown"
    
    @patch('subprocess.run')
    def test_get_selvage_version_exception(self, mock_run):
        """예외 발생 시 테스트"""
        mock_run.side_effect = Exception("Command failed")
        
        version = get_selvage_version()
        assert version == "unknown"


@pytest.mark.unit
class TestReviewLogInfo:
    """ReviewLogInfo 데이터 클래스 테스트"""
    
    def test_review_log_info_creation(self):
        """ReviewLogInfo 객체 생성 테스트"""
        log_info = ReviewLogInfo(
            repo_name="test-repo",
            commit_id="abc123",
            model_name="gemini-2.5-pro",
            file_path=Path("/test/file.json"),
            file_name="file.json"
        )
        
        assert log_info.repo_name == "test-repo"
        assert log_info.commit_id == "abc123"
        assert log_info.model_name == "gemini-2.5-pro"
        assert log_info.file_path == Path("/test/file.json")
        assert log_info.file_name == "file.json"


@pytest.mark.unit
class TestDeepEvalTestCase:
    """DeepEvalTestCase 데이터 클래스 테스트"""
    
    def test_deepeval_test_case_creation(self):
        """DeepEvalTestCase 객체 생성 테스트"""
        test_case = DeepEvalTestCase(
            input="test input",
            actual_output="test output",
            expected_output="expected output"
        )
        
        assert test_case.input == "test input"
        assert test_case.actual_output == "test output"
        assert test_case.expected_output == "expected output"
    
    def test_deepeval_test_case_optional_expected_output(self):
        """expected_output이 옵션인 경우 테스트"""
        test_case = DeepEvalTestCase(
            input="test input",
            actual_output="test output"
        )
        
        assert test_case.input == "test input"
        assert test_case.actual_output == "test output"
        assert test_case.expected_output is None