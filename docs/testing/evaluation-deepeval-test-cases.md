# DeepEval 평가 시스템 - 테스트 케이스

본 문서는 [`evaluation-deepeval-implementation.md`](../implementation/evaluation-deepeval-implementation.md) 구현을 위한 포괄적인 테스트 케이스입니다.

## 1. 테스트 환경 설정

```python
"""테스트 의존성 및 픽스처"""
import pytest
import json
import tempfile
import os
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# 테스트 대상 클래스들
from selvage_eval.tools.deepeval_test_case_converter_tool import (
    DeepEvalTestCaseConverterTool, ReviewLogInfo, DeepEvalTestCase
)
from selvage_eval.tools.deepeval_executor_tool import DeepEvalExecutorTool
from selvage_eval.tools.tool_result import ToolResult

# 테스트 마커 정의
pytestmark = [
    pytest.mark.unit,  # 기본적으로 모든 테스트는 unit 테스트
]


@pytest.fixture
def temp_base_dir():
    """모든 테스트용 기본 임시 디렉토리"""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)
        
        # 표준 디렉토리 구조 생성
        (base_path / "review_logs").mkdir(exist_ok=True)
        (base_path / "deep_eval_test_case").mkdir(exist_ok=True)
        (base_path / "deepeval_results").mkdir(exist_ok=True)
        
        yield base_path


@pytest.fixture
def mock_tool_paths(temp_base_dir):
    """Tool 클래스의 경로를 임시 디렉토리로 패치"""
    with patch.object(DeepEvalTestCaseConverterTool, 'review_logs_base_path', str(temp_base_dir / "review_logs")), \
         patch.object(DeepEvalTestCaseConverterTool, 'output_base_path', str(temp_base_dir / "deep_eval_test_case")), \
         patch.object(DeepEvalExecutorTool, 'test_case_base_path', str(temp_base_dir / "deep_eval_test_case")), \
         patch.object(DeepEvalExecutorTool, 'results_base_path', str(temp_base_dir / "deepeval_results")):
        yield temp_base_dir


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
        ("test-repo-1", "def456", "claude-sonnet-4"),
        ("test-repo-2", "ghi789", "gemini-2.5-pro")
    ]
    
    for repo_name, commit_id, model_name in repos_data:
        log_dir = base_dir / repo_name / commit_id / model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 리뷰 로그 파일 생성
        log_file = log_dir / f"review_log_{commit_id}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(sample_review_log_data, f, ensure_ascii=False, indent=2)
    
    return base_dir


@pytest.fixture
def sample_review_log_info(temp_base_dir):
    """샘플 ReviewLogInfo 객체"""
    return ReviewLogInfo(
        repo_name="test-repo",
        commit_id="abc123",
        model_name="gemini-2.5-pro",
        file_path=temp_base_dir / "review_log.json",
        file_name="review_log.json"
    )


@pytest.fixture
def sample_deepeval_test_case():
    """샘플 DeepEvalTestCase 객체"""
    return DeepEvalTestCase(
        input='[{"role": "system", "content": "Review code"}]',
        actual_output='{"issues": [], "summary": "No issues found"}',
        expected_output=None
    )


@pytest.fixture
def mock_env_vars():
    """환경 변수 Mock"""
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "GEMINI_API_KEY": "test-gemini-key"
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def performance_thresholds():
    """환경별 성능 임계값"""
    # CI 환경 감지
    is_ci = os.getenv("CI", "false").lower() == "true"
    
    if is_ci:
        return {
            "processing_time": 60.0,  # CI에서는 더 여유있게
            "memory_increase": 200,   # MB
            "concurrent_time": 120.0
        }
    else:
        return {
            "processing_time": 30.0,  # 로컬에서는 빠르게
            "memory_increase": 100,   # MB  
            "concurrent_time": 60.0
        }


@pytest.fixture
def test_cases_directory(temp_base_dir, sample_session_id):
    """테스트 케이스 디렉토리 구조"""
    test_cases_dir = temp_base_dir / "deep_eval_test_case" / sample_session_id
    
    # 메타데이터 파일
    metadata = {
        "selvage_version": "0.1.2",
        "execution_date": datetime.now().isoformat(),
        "tool_name": "deepeval_test_case_converter",
        "created_by": "DeepEvalTestCaseConverterTool",
        "evaluation_framework": "DeepEval"
    }
    
    test_cases_dir.mkdir(parents=True, exist_ok=True)
    with open(test_cases_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # 모델별 테스트 케이스 파일
    models = ["gemini-2.5-pro", "claude-sonnet-4"]
    for model in models:
        model_dir = test_cases_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        
        test_cases = [
            {
                "input": f'[{{"role": "user", "content": "Review this code"}}]',
                "actual_output": f'{{"summary": "Code reviewed by {model}"}}',
                "expected_output": None,
                "repo_name": "test-repo"
            }
        ]
        
        with open(model_dir / "test_cases.json", 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, ensure_ascii=False, indent=2)
    
    return test_cases_dir
```

## 2. 데이터 클래스 단위 테스트

### 2.1 ReviewLogInfo 테스트

```python
class TestReviewLogInfo:
    """ReviewLogInfo 데이터 클래스 테스트"""
    
    def test_creation(self, sample_review_log_info):
        """정상적인 객체 생성 테스트"""
        assert sample_review_log_info.repo_name == "test-repo"
        assert sample_review_log_info.commit_id == "abc123"
        assert sample_review_log_info.model_name == "gemini-2.5-pro"
        assert sample_review_log_info.file_path == temp_base_dir / "review_log.json"
        assert sample_review_log_info.file_name == "review_log.json"
    
    def test_equality(self):
        """객체 동등성 테스트"""
        log1 = ReviewLogInfo(
            repo_name="repo1",
            commit_id="commit1",
            model_name="model1",
            file_path=temp_base_dir / "file1.json",
            file_name="file1.json"
        )
        
        log2 = ReviewLogInfo(
            repo_name="repo1",
            commit_id="commit1",
            model_name="model1",
            file_path=temp_base_dir / "file1.json",
            file_name="file1.json"
        )
        
        assert log1 == log2
    
    def test_different_objects(self):
        """다른 객체 비교 테스트"""
        log1 = ReviewLogInfo(
            repo_name="repo1",
            commit_id="commit1",
            model_name="model1",
            file_path=temp_base_dir / "file1.json",
            file_name="file1.json"
        )
        
        log2 = ReviewLogInfo(
            repo_name="repo2",  # 다른 저장소
            commit_id="commit1",
            model_name="model1",
            file_path=temp_base_dir / "file1.json",
            file_name="file1.json"
        )
        
        assert log1 != log2


### 2.2 DeepEvalTestCase 테스트

```python
class TestDeepEvalTestCase:
    """DeepEvalTestCase 데이터 클래스 테스트"""
    
    def test_creation(self, sample_deepeval_test_case):
        """정상적인 객체 생성 테스트"""
        assert sample_deepeval_test_case.input == '[{"role": "system", "content": "Review code"}]'
        assert sample_deepeval_test_case.actual_output == '{"issues": [], "summary": "No issues found"}'
        assert sample_deepeval_test_case.expected_output is None
    
    def test_with_expected_output(self):
        """expected_output이 있는 경우 테스트"""
        test_case = DeepEvalTestCase(
            input="test input",
            actual_output="actual result",
            expected_output="expected result"
        )
        
        assert test_case.input == "test input"
        assert test_case.actual_output == "actual result"
        assert test_case.expected_output == "expected result"
    
    def test_empty_strings(self):
        """빈 문자열 처리 테스트"""
        test_case = DeepEvalTestCase(
            input="",
            actual_output="",
            expected_output=""
        )
        
        assert test_case.input == ""
        assert test_case.actual_output == ""
        assert test_case.expected_output == ""
```

## 3. DeepEvalTestCaseConverterTool 단위 테스트

### 3.1 초기화 및 기본 기능 테스트

```python
class TestDeepEvalTestCaseConverterTool:
    """DeepEvalTestCaseConverterTool 클래스 테스트"""
    
    def test_initialization(self, mock_tool_paths):
        """정상적인 초기화 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        assert tool.name == "deepeval_test_case_converter"
        assert "리뷰 로그를 DeepEval 테스트 케이스 형식으로 변환" in tool.description
        assert tool.review_logs_base_path == str(mock_tool_paths / "review_logs")
        assert tool.output_base_path == str(mock_tool_paths / "deep_eval_test_case")
        assert tool.parameters_schema is not None
    
    def test_validate_parameters_success(self, mock_tool_paths):
        """파라미터 유효성 검증 성공 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        valid_params = {
            "session_id": "test-session-123"
        }
        
        assert tool.validate_parameters(valid_params) is True
    
    def test_validate_parameters_with_optional(self, mock_tool_paths):
        """선택적 파라미터 포함 유효성 검증 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        valid_params = {
            "session_id": "test-session-123",
            "review_logs_path": "/custom/path",
            "output_path": "/custom/output"
        }
        
        assert tool.validate_parameters(valid_params) is True
    
    def test_validate_parameters_missing_session_id(self, mock_tool_paths):
        """session_id 누락 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        invalid_params = {
            "review_logs_path": "/path/to/logs"
        }
        
        assert tool.validate_parameters(invalid_params) is False
    
    def test_validate_parameters_empty_session_id(self, mock_tool_paths):
        """빈 session_id 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        invalid_params = {
            "session_id": ""
        }
        
        assert tool.validate_parameters(invalid_params) is False
    
    def test_validate_parameters_wrong_type(self, mock_tool_paths):
        """잘못된 타입 파라미터 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        invalid_params = {
            "session_id": 123  # int 타입
        }
        
        assert tool.validate_parameters(invalid_params) is False


class TestReviewLogScanning:
    """리뷰 로그 스캔 기능 테스트"""
    
    def test_scan_review_logs_success(self, review_logs_structure, mock_tool_paths):
        """리뷰 로그 스캔 성공 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        review_logs = tool._scan_review_logs(review_logs_structure)
        
        assert len(review_logs) == 3  # 3개의 로그 파일
        
        # 각 로그의 정보 확인
        log_info = review_logs[0]
        assert isinstance(log_info, ReviewLogInfo)
        assert log_info.repo_name in ["test-repo-1", "test-repo-2"]
        assert log_info.commit_id in ["abc123", "def456", "ghi789"]
        assert log_info.model_name in ["gemini-2.5-pro", "claude-sonnet-4"]
    
    def test_scan_review_logs_empty_directory(self, temp_base_dir, mock_tool_paths):
        """빈 디렉토리 스캔 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        empty_dir = temp_base_dir / "empty_logs"
        empty_dir.mkdir(parents=True, exist_ok=True)
        
        review_logs = tool._scan_review_logs(empty_dir)
        
        assert len(review_logs) == 0
    
    def test_scan_review_logs_nonexistent_directory(self, mock_tool_paths):
        """존재하지 않는 디렉토리 스캔 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        nonexistent_dir = Path("/nonexistent/directory")
        
        review_logs = tool._scan_review_logs(nonexistent_dir)
        
        assert len(review_logs) == 0
    
    def test_group_logs_by_model(self, mock_tool_paths, temp_base_dir):
        """모델별 로그 그룹화 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        review_logs = [
            ReviewLogInfo("repo1", "commit1", "gemini-2.5-pro", temp_base_dir / "1.json", "1.json"),
            ReviewLogInfo("repo1", "commit2", "gemini-2.5-pro", temp_base_dir / "2.json", "2.json"),
            ReviewLogInfo("repo2", "commit3", "claude-sonnet-4", temp_base_dir / "3.json", "3.json")
        ]
        
        grouped = tool._group_logs_by_model(review_logs)
        
        assert len(grouped) == 2
        assert "gemini-2.5-pro" in grouped
        assert "claude-sonnet-4" in grouped
        assert len(grouped["gemini-2.5-pro"]) == 2
        assert len(grouped["claude-sonnet-4"]) == 1


class TestDataExtraction:
    """데이터 추출 기능 테스트"""
    
    def test_extract_prompt_and_response_success(self, temp_base_dir, sample_review_log_data):
        """prompt와 response 추출 성공 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # 테스트 파일 생성
        log_file = temp_base_dir / "test_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(sample_review_log_data, f)
        
        extracted = tool._extract_prompt_and_response(log_file)
        
        assert extracted is not None
        assert "prompt" in extracted
        assert "review_response" in extracted
        assert "original_data" in extracted
        assert len(extracted["prompt"]) == 2
        assert "issues" in extracted["review_response"]
    
    def test_extract_prompt_and_response_missing_fields(self, temp_base_dir):
        """필수 필드 누락 시 추출 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # prompt 필드 누락
        incomplete_data = {
            "review_response": {"summary": "test"}
        }
        
        log_file = temp_base_dir / "incomplete_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(incomplete_data, f)
        
        extracted = tool._extract_prompt_and_response(log_file)
        
        assert extracted is None
    
    def test_extract_prompt_and_response_invalid_json(self, temp_base_dir):
        """잘못된 JSON 파일 처리 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # 잘못된 JSON 파일 생성
        log_file = temp_base_dir / "invalid_log.json"
        with open(log_file, 'w') as f:
            f.write("{ invalid json content")
        
        extracted = tool._extract_prompt_and_response(log_file)
        
        assert extracted is None
    
    def test_convert_to_deepeval_format(self, sample_review_log_data):
        """DeepEval 형식 변환 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        extracted_data = {
            "prompt": sample_review_log_data["prompt"],
            "review_response": sample_review_log_data["review_response"]
        }
        
        test_case = tool._convert_to_deepeval_format(extracted_data)
        
        assert isinstance(test_case, DeepEvalTestCase)
        assert test_case.input is not None
        assert test_case.actual_output is not None
        assert test_case.expected_output is None
        
        # JSON 형식인지 확인
        input_data = json.loads(test_case.input)
        output_data = json.loads(test_case.actual_output)
        
        assert isinstance(input_data, list)
        assert isinstance(output_data, dict)
    
    def test_test_case_to_dict(self, sample_deepeval_test_case):
        """테스트 케이스 딕셔너리 변환 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # repo_name 속성 추가
        sample_deepeval_test_case.repo_name = "test-repo"
        
        result_dict = tool._test_case_to_dict(sample_deepeval_test_case)
        
        assert "input" in result_dict
        assert "actual_output" in result_dict
        assert "expected_output" in result_dict
        assert "repo_name" in result_dict
        assert result_dict["repo_name"] == "test-repo"
```

### 3.2 메타데이터 생성 및 전체 실행 테스트

```python
class TestMetadataAndExecution:
    """메타데이터 생성 및 전체 실행 테스트"""
    
    def test_create_metadata(self):
        """메타데이터 생성 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        metadata = tool._create_metadata()
        
        assert "selvage_version" in metadata
        assert "execution_date" in metadata
        assert "tool_name" in metadata
        assert "created_by" in metadata
        assert "evaluation_framework" in metadata
        
        assert metadata["tool_name"] == "deepeval_test_case_converter"
        assert metadata["created_by"] == "DeepEvalTestCaseConverterTool"
        assert metadata["evaluation_framework"] == "DeepEval"
        
        # 날짜 형식 확인 (ISO 형식)
        execution_date = metadata["execution_date"]
        datetime.fromisoformat(execution_date)  # 예외가 발생하지 않으면 올바른 형식
    
    def test_execute_success(self, review_logs_structure, temp_base_dir, sample_session_id, mock_tool_paths):
        """전체 실행 성공 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        result = tool.execute(
            session_id=sample_session_id,
            review_logs_path=str(review_logs_structure),
            output_path=str(temp_base_dir / "output")
        )
        
        assert result.success is True
        assert result.data is not None
        assert result.error_message is None
        
        # 결과 데이터 확인
        data = result.data
        assert data["session_id"] == sample_session_id
        assert "converted_files" in data
        assert "total_files" in data
        assert "metadata_path" in data
        
        # 변환된 파일 확인
        converted_files = data["converted_files"]
        assert len(converted_files) >= 1  # 최소 1개 모델
        
        for model_name, file_info in converted_files.items():
            assert "file_path" in file_info
            assert "test_case_count" in file_info
            assert file_info["test_case_count"] > 0
            
            # 실제 파일 존재 확인
            file_path = Path(file_info["file_path"])
            assert file_path.exists()
            assert file_path.is_file()
    
    def test_execute_no_review_logs(self, temp_base_dir, sample_session_id):
        """리뷰 로그가 없는 경우 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # 빈 디렉토리
        empty_dir = temp_base_dir / "empty_logs"
        empty_dir.mkdir(parents=True, exist_ok=True)
        
        result = tool.execute(
            session_id=sample_session_id,
            review_logs_path=str(empty_dir),
            output_path=str(temp_base_dir / "output")
        )
        
        assert result.success is False
        assert result.data is None
        assert "리뷰 로그 파일을 찾을 수 없습니다" in result.error_message
    
    def test_execute_output_directory_creation(self, review_logs_structure, temp_base_dir, sample_session_id):
        """출력 디렉토리 자동 생성 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # 존재하지 않는 중첩 디렉토리
        nested_output = temp_base_dir / "nested" / "deep" / "output"
        
        result = tool.execute(
            session_id=sample_session_id,
            review_logs_path=str(review_logs_structure),
            output_path=str(nested_output)
        )
        
        assert result.success is True
        
        # 디렉토리가 자동 생성되었는지 확인
        session_dir = nested_output / sample_session_id
        assert session_dir.exists()
        assert session_dir.is_dir()
        
        # 메타데이터 파일 생성 확인
        metadata_file = session_dir / "metadata.json"
        assert metadata_file.exists()
    
    def test_execute_with_default_paths(self, sample_session_id):
        """기본 경로 사용 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # 기본 경로가 존재하지 않는 경우
        result = tool.execute(session_id=sample_session_id)
        
        # 기본 경로에 리뷰 로그가 없으므로 실패해야 함
        assert result.success is False
        assert "리뷰 로그 파일을 찾을 수 없습니다" in result.error_message
    
    def test_execute_exception_handling(self, sample_session_id):
        """예외 처리 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # _scan_review_logs에서 예외 발생 시뮬레이션
        with patch.object(tool, '_scan_review_logs', side_effect=Exception("Test exception")):
            result = tool.execute(session_id=sample_session_id)
        
        assert result.success is False
        assert result.data is None
        assert "DeepEval 테스트 케이스 변환 실패" in result.error_message
        assert "Test exception" in result.error_message


class TestFileOperations:
    """파일 작업 테스트"""
    
    def test_json_file_creation(self, review_logs_structure, temp_base_dir, sample_session_id):
        """JSON 파일 생성 및 내용 확인 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        result = tool.execute(
            session_id=sample_session_id,
            review_logs_path=str(review_logs_structure),
            output_path=str(temp_base_dir)
        )
        
        assert result.success is True
        
        # 생성된 파일들 확인
        session_dir = temp_base_dir / sample_session_id
        
        # 메타데이터 파일 확인
        metadata_file = session_dir / "metadata.json"
        assert metadata_file.exists()
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        assert "selvage_version" in metadata
        assert "execution_date" in metadata
        
        # 모델별 테스트 케이스 파일 확인
        for model_name, file_info in result.data["converted_files"].items():
            test_cases_file = Path(file_info["file_path"])
            assert test_cases_file.exists()
            
            with open(test_cases_file, 'r', encoding='utf-8') as f:
                test_cases = json.load(f)
            
            assert isinstance(test_cases, list)
            assert len(test_cases) > 0
            
            for test_case in test_cases:
                assert "input" in test_case
                assert "actual_output" in test_case
                assert "expected_output" in test_case
                
                # JSON 형식 확인
                json.loads(test_case["input"])  # 예외가 발생하지 않으면 올바른 JSON
                json.loads(test_case["actual_output"])
    
    def test_unicode_handling(self, temp_base_dir, sample_session_id):
        """유니코드 처리 테스트"""
        tool = DeepEvalTestCaseConverterTool()
        
        # 유니코드 포함 리뷰 로그 생성
        base_dir = temp_base_dir / "unicode_logs"
        log_dir = base_dir / "한글저장소" / "커밋123" / "gemini-2.5-pro"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        unicode_log_data = {
            "prompt": [
                {"role": "system", "content": "한글 시스템 메시지"},
                {"role": "user", "content": "코드를 리뷰해주세요"}
            ],
            "review_response": {
                "issues": [
                    {
                        "type": "버그",
                        "description": "한글로 된 설명",
                        "suggestion": "개선 제안사항"
                    }
                ],
                "summary": "한글 요약"
            }
        }
        
        log_file = log_dir / "unicode_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(unicode_log_data, f, ensure_ascii=False, indent=2)
        
        # 변환 실행
        result = tool.execute(
            session_id=sample_session_id,
            review_logs_path=str(base_dir),
            output_path=str(temp_base_dir / "unicode_output")
        )
        
        assert result.success is True
        
        # 유니코드가 올바르게 처리되었는지 확인
        for model_name, file_info in result.data["converted_files"].items():
            test_cases_file = Path(file_info["file_path"])
            
            with open(test_cases_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "한글" in content  # 유니코드가 보존되었는지 확인
```

## 4. DeepEvalExecutorTool 단위 테스트

### 4.1 초기화 및 기본 기능 테스트

```python
class TestDeepEvalExecutorTool:
    """DeepEvalExecutorTool 클래스 테스트"""
    
    def test_initialization(self, mock_tool_paths):
        """정상적인 초기화 테스트"""
        tool = DeepEvalExecutorTool()
        
        assert tool.name == "deepeval_executor"
        assert "DeepEval을 사용하여 코드 리뷰 품질을 평가" in tool.description
        assert tool.test_case_base_path == str(mock_tool_paths / "deep_eval_test_case")
        assert tool.results_base_path == str(mock_tool_paths / "deepeval_results")
        assert tool.command_executor is not None
        assert tool.parameters_schema is not None
    
    def test_validate_parameters_success(self):
        """파라미터 유효성 검증 성공 테스트"""
        tool = DeepEvalExecutorTool()
        
        valid_params = {
            "session_id": "test-eval-session"
        }
        
        assert tool.validate_parameters(valid_params) is True
    
    def test_validate_parameters_with_optional(self):
        """선택적 파라미터 포함 유효성 검증 테스트"""
        tool = DeepEvalExecutorTool()
        
        valid_params = {
            "session_id": "test-eval-session",
            "test_case_path": "/custom/test/path",
            "parallel_workers": 4,
            "display_filter": "failing"
        }
        
        assert tool.validate_parameters(valid_params) is True
    
    def test_validate_parameters_missing_session_id(self):
        """session_id 누락 테스트"""
        tool = DeepEvalExecutorTool()
        
        invalid_params = {
            "parallel_workers": 2
        }
        
        assert tool.validate_parameters(invalid_params) is False
    
    def test_validate_parameters_invalid_parallel_workers(self):
        """잘못된 parallel_workers 테스트"""
        tool = DeepEvalExecutorTool()
        
        invalid_params = {
            "session_id": "test-session",
            "parallel_workers": 0  # 0 이하는 무효
        }
        
        assert tool.validate_parameters(invalid_params) is False
        
        invalid_params["parallel_workers"] = -1  # 음수 테스트
        assert tool.validate_parameters(invalid_params) is False
        
        invalid_params["parallel_workers"] = "not_int"  # 문자열 테스트
        assert tool.validate_parameters(invalid_params) is False


class TestEnvironmentChecking:
    """환경 변수 확인 테스트"""
    
    def test_check_environment_success(self, mock_env_vars):
        """환경 변수 확인 성공 테스트"""
        tool = DeepEvalExecutorTool()
        
        env_check = tool._check_environment()
        
        assert env_check["valid"] is True
        assert "환경 설정 완료" in env_check["message"]
    
    def test_check_environment_missing_openai_key(self):
        """OPENAI_API_KEY 누락 테스트"""
        tool = DeepEvalExecutorTool()
        
        # GEMINI_API_KEY만 설정
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            env_check = tool._check_environment()
        
        assert env_check["valid"] is False
        assert "OPENAI_API_KEY" in env_check["message"]
    
    def test_check_environment_missing_gemini_key(self):
        """GEMINI_API_KEY 누락 테스트"""
        tool = DeepEvalExecutorTool()
        
        # OPENAI_API_KEY만 설정
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            env_check = tool._check_environment()
        
        assert env_check["valid"] is False
        assert "GEMINI_API_KEY" in env_check["message"]
    
    def test_check_environment_no_keys(self):
        """모든 키 누락 테스트"""
        tool = DeepEvalExecutorTool()
        
        with patch.dict(os.environ, {}, clear=True):
            env_check = tool._check_environment()
        
        assert env_check["valid"] is False
        assert "OPENAI_API_KEY" in env_check["message"]
        assert "GEMINI_API_KEY" in env_check["message"]


class TestTestCaseLoading:
    """테스트 케이스 로딩 테스트"""
    
    def test_load_test_cases_success(self, test_cases_directory):
        """테스트 케이스 로드 성공 테스트"""
        tool = DeepEvalExecutorTool()
        
        # 첫 번째 모델의 테스트 케이스 파일
        model_dirs = list(test_cases_directory.iterdir())
        model_dirs = [d for d in model_dirs if d.is_dir()]
        
        if model_dirs:
            test_cases_file = model_dirs[0] / "test_cases.json"
            test_cases = tool._load_test_cases(test_cases_file)
            
            assert isinstance(test_cases, list)
            assert len(test_cases) > 0
            
            # LLMTestCase 객체인지 확인 (Mock 사용)
            with patch('selvage_eval.tools.deepeval_executor_tool.LLMTestCase') as mock_test_case:
                tool._load_test_cases(test_cases_file)
                assert mock_test_case.called
    
    def test_load_test_cases_file_not_found(self):
        """존재하지 않는 파일 로드 테스트"""
        tool = DeepEvalExecutorTool()
        
        nonexistent_file = Path("/nonexistent/test_cases.json")
        test_cases = tool._load_test_cases(nonexistent_file)
        
        assert test_cases == []
    
    def test_load_test_cases_invalid_json(self, temp_base_dir):
        """잘못된 JSON 파일 로드 테스트"""
        tool = DeepEvalExecutorTool()
        
        # 잘못된 JSON 파일 생성
        invalid_file = temp_base_dir / "invalid_test_cases.json"
        with open(invalid_file, 'w') as f:
            f.write("{ invalid json")
        
        test_cases = tool._load_test_cases(invalid_file)
        
        assert test_cases == []
    
    def test_load_test_cases_empty_array(self, temp_base_dir):
        """빈 배열 JSON 파일 로드 테스트"""
        tool = DeepEvalExecutorTool()
        
        # 빈 배열 JSON 파일 생성
        empty_file = temp_base_dir / "empty_test_cases.json"
        with open(empty_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        
        test_cases = tool._load_test_cases(empty_file)
        
        assert isinstance(test_cases, list)
        assert len(test_cases) == 0


class TestMetricsCreation:
    """메트릭 생성 테스트"""
    
    @patch('selvage_eval.tools.deepeval_executor_tool.GEval')
    @patch('selvage_eval.tools.deepeval_executor_tool.JsonCorrectnessMetric')
    def test_create_metrics(self, mock_json_metric, mock_g_eval):
        """메트릭 생성 테스트"""
        tool = DeepEvalExecutorTool()
        
        metrics = tool._create_metrics()
        
        assert isinstance(metrics, list)
        # GEval이 3번 호출되었는지 확인 (Correctness, Clarity, Actionability)
        assert mock_g_eval.call_count == 3
        # JsonCorrectnessMetric이 1번 호출되었는지 확인
        assert mock_json_metric.call_count == 1
    
    @patch('selvage_eval.tools.deepeval_executor_tool.GEval')
    def test_create_metrics_correctness_config(self, mock_g_eval):
        """Correctness 메트릭 설정 확인 테스트"""
        tool = DeepEvalExecutorTool()
        
        tool._create_metrics()
        
        # 첫 번째 GEval 호출 (Correctness)이 올바른 설정으로 되었는지 확인
        first_call = mock_g_eval.call_args_list[0]
        kwargs = first_call[1]
        
        assert kwargs["name"] == "Correctness"
        assert kwargs["model"] == "gemini-2.0-flash-exp"
        assert kwargs["threshold"] == 0.7
        assert isinstance(kwargs["evaluation_steps"], list)
        assert len(kwargs["evaluation_steps"]) > 0


class TestEvaluationExecution:
    """평가 실행 테스트"""
    
    @patch('selvage_eval.tools.deepeval_executor_tool.evaluate')
    def test_run_evaluation_success(self, mock_evaluate, test_cases_directory):
        """평가 실행 성공 테스트"""
        tool = DeepEvalExecutorTool()
        
        # Mock 설정
        mock_evaluate.return_value = None  # evaluate 함수는 반환값 없음
        
        # 첫 번째 모델의 테스트 케이스 파일
        model_dirs = [d for d in test_cases_directory.iterdir() if d.is_dir()]
        if model_dirs:
            test_cases_file = model_dirs[0] / "test_cases.json"
            output_path = test_cases_directory / "output"
            
            with patch.object(tool, '_load_test_cases', return_value=[Mock()]), \
                 patch.object(tool, '_create_metrics', return_value=[Mock()]):
                
                result = tool._run_evaluation(
                    test_cases_file=test_cases_file,
                    output_path=output_path,
                    parallel_workers=2,
                    display_filter="all"
                )
            
            assert result["success"] is True
            assert "executed" in result["data"]
            mock_evaluate.assert_called_once()
    
    def test_run_evaluation_no_test_cases(self, temp_base_dir):
        """테스트 케이스 없음 오류 테스트"""
        tool = DeepEvalExecutorTool()
        
        # 빈 테스트 케이스 파일
        empty_file = temp_base_dir / "empty.json"
        with open(empty_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        
        output_path = temp_base_dir / "output"
        
        result = tool._run_evaluation(
            test_cases_file=empty_file,
            output_path=output_path,
            parallel_workers=1,
            display_filter="all"
        )
        
        assert result["success"] is False
        assert "테스트 케이스를 로드할 수 없습니다" in result["error"]
    
    @patch('selvage_eval.tools.deepeval_executor_tool.evaluate')
    def test_run_evaluation_exception(self, mock_evaluate, test_cases_directory):
        """평가 실행 중 예외 처리 테스트"""
        tool = DeepEvalExecutorTool()
        
        # Mock에서 예외 발생
        mock_evaluate.side_effect = Exception("DeepEval execution error")
        
        model_dirs = [d for d in test_cases_directory.iterdir() if d.is_dir()]
        if model_dirs:
            test_cases_file = model_dirs[0] / "test_cases.json"
            output_path = test_cases_directory / "output"
            
            with patch.object(tool, '_load_test_cases', return_value=[Mock()]), \
                 patch.object(tool, '_create_metrics', return_value=[Mock()]):
                
                result = tool._run_evaluation(
                    test_cases_file=test_cases_file,
                    output_path=output_path,
                    parallel_workers=1,
                    display_filter="all"
                )
            
            assert result["success"] is False
            assert "평가 실행 중 오류" in result["error"]
            assert "DeepEval execution error" in result["error"]
```

## 5. 전체 실행 프로세스 테스트

```python
class TestFullExecution:
    """전체 실행 프로세스 테스트"""
    
    def test_execute_success(self, test_cases_directory, sample_session_id, mock_env_vars):
        """전체 실행 성공 테스트"""
        tool = DeepEvalExecutorTool()
        
        with patch.object(tool, '_run_evaluation', return_value={"success": True, "data": {"executed": True}}):
            result = tool.execute(
                session_id=sample_session_id,
                test_case_path=str(test_cases_directory),
                parallel_workers=2,
                display_filter="all"
            )
        
        assert result.success is True
        assert result.data is not None
        assert result.error_message is None
        
        # 결과 데이터 확인
        data = result.data
        assert data["session_id"] == sample_session_id
        assert "evaluation_results" in data
        assert "total_evaluations" in data
        assert "metadata_path" in data
        
        # 평가 결과 확인
        evaluation_results = data["evaluation_results"]
        assert isinstance(evaluation_results, dict)
        assert len(evaluation_results) >= 1  # 최소 1개 모델
    
    def test_execute_environment_check_failure(self, sample_session_id):
        """환경 확인 실패 테스트"""
        tool = DeepEvalExecutorTool()
        
        with patch.dict(os.environ, {}, clear=True):
            result = tool.execute(session_id=sample_session_id)
        
        assert result.success is False
        assert result.data is None
        assert "환경 설정 오류" in result.error_message
    
    def test_execute_test_case_directory_not_found(self, sample_session_id, mock_env_vars):
        """테스트 케이스 디렉토리 없음 테스트"""
        tool = DeepEvalExecutorTool()
        
        result = tool.execute(
            session_id=sample_session_id,
            test_case_path="/nonexistent/directory"
        )
        
        assert result.success is False
        assert result.data is None
        assert "테스트 케이스 디렉토리를 찾을 수 없습니다" in result.error_message
    
    def test_execute_with_session_id_auto_discovery(self, test_cases_directory, sample_session_id, mock_env_vars):
        """session_id로 자동 디렉토리 검색 테스트"""
        tool = DeepEvalExecutorTool()
        
        # test_case_base_path를 임시 디렉토리로 설정
        tool.test_case_base_path = str(test_cases_directory.parent)
        
        with patch.object(tool, '_run_evaluation', return_value={"success": True, "data": {"executed": True}}):
            result = tool.execute(session_id=sample_session_id)
        
        assert result.success is True
        assert result.data["session_id"] == sample_session_id
    
    def test_execute_results_directory_creation(self, test_cases_directory, sample_session_id, mock_env_vars, temp_base_dir):
        """결과 디렉토리 자동 생성 테스트"""
        tool = DeepEvalExecutorTool()
        
        # results_base_path를 임시 디렉토리로 설정
        tool.results_base_path = str(temp_base_dir / "deepeval_results")
        
        with patch.object(tool, '_run_evaluation', return_value={"success": True, "data": {"executed": True}}):
            result = tool.execute(
                session_id=sample_session_id,
                test_case_path=str(test_cases_directory)
            )
        
        assert result.success is True
        
        # 결과 디렉토리가 생성되었는지 확인
        results_dir = Path(tool.results_base_path) / sample_session_id
        assert results_dir.exists()
        assert results_dir.is_dir()
        
        # 메타데이터 파일 생성 확인
        metadata_file = results_dir / "metadata.json"
        assert metadata_file.exists()
    
    def test_execute_multiple_models(self, test_cases_directory, sample_session_id, mock_env_vars):
        """다중 모델 평가 테스트"""
        tool = DeepEvalExecutorTool()
        
        with patch.object(tool, '_run_evaluation', return_value={"success": True, "data": {"executed": True}}):
            result = tool.execute(
                session_id=sample_session_id,
                test_case_path=str(test_cases_directory)
            )
        
        assert result.success is True
        
        # 여러 모델이 처리되었는지 확인
        evaluation_results = result.data["evaluation_results"]
        
        # test_cases_directory fixture에서 2개 모델 생성하므로
        assert len(evaluation_results) == 2
        
        # 각 모델의 결과 확인
        for model_name, model_result in evaluation_results.items():
            assert "success" in model_result
            assert model_result["success"] is True
            assert model_result.get("error") is None
    
    def test_execute_metadata_creation(self, test_cases_directory, sample_session_id, mock_env_vars, temp_base_dir):
        """메타데이터 생성 확인 테스트"""
        tool = DeepEvalExecutorTool()
        tool.results_base_path = str(temp_base_dir / "results")
        
        with patch.object(tool, '_run_evaluation', return_value={"success": True, "data": {"executed": True}}):
            result = tool.execute(
                session_id=sample_session_id,
                test_case_path=str(test_cases_directory)
            )
        
        assert result.success is True
        
        # 메타데이터 파일 확인
        metadata_path = Path(result.data["metadata_path"])
        assert metadata_path.exists()
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        assert metadata["session_id"] == sample_session_id
        assert metadata["tool_name"] == "deepeval_executor"
        assert metadata["created_by"] == "DeepEvalExecutorTool"
        assert metadata["deep_eval_test_case_path"] == str(test_cases_directory)
    
    def test_execute_exception_handling(self, sample_session_id, mock_env_vars):
        """예외 처리 테스트"""
        tool = DeepEvalExecutorTool()
        
        # _check_environment에서 예외 발생 시뮬레이션
        with patch.object(tool, '_check_environment', side_effect=Exception("Unexpected error")):
            result = tool.execute(session_id=sample_session_id)
        
        assert result.success is False
        assert result.data is None
        assert "DeepEval 평가 실행 실패" in result.error_message
        assert "Unexpected error" in result.error_message
```

## 6. 통합 테스트

```python
@pytest.mark.integration
class TestIntegration:
    """통합 테스트"""
    
    def test_full_workflow_converter_to_executor(self, review_logs_structure, temp_base_dir, sample_session_id, mock_env_vars):
        """변환기 → 실행기 전체 워크플로우 테스트"""
        # 1. DeepEval 테스트 케이스 변환
        converter = DeepEvalTestCaseConverterTool()
        
        converter_result = converter.execute(
            session_id=sample_session_id,
            review_logs_path=str(review_logs_structure),
            output_path=str(temp_base_dir / "test_cases")
        )
        
        assert converter_result.success is True
        assert len(converter_result.data["converted_files"]) > 0
        
        # 2. DeepEval 평가 실행
        executor = DeepEvalExecutorTool()
        executor.results_base_path = str(temp_base_dir / "results")
        
        test_cases_dir = temp_base_dir / "test_cases" / sample_session_id
        
        with patch.object(executor, '_run_evaluation', return_value={"success": True, "data": {"executed": True}}):
            executor_result = executor.execute(
                session_id=sample_session_id,
                test_case_path=str(test_cases_dir)
            )
        
        assert executor_result.success is True
        assert executor_result.data["session_id"] == sample_session_id
        
        # 3. 결과 파일 확인
        results_dir = Path(executor.results_base_path) / sample_session_id
        assert results_dir.exists()
        
        metadata_file = results_dir / "metadata.json"
        assert metadata_file.exists()
    
    def test_multiple_session_workflow(self, review_logs_structure, temp_base_dir, mock_env_vars):
        """다중 세션 워크플로우 테스트"""
        sessions = ["session-1", "session-2", "session-3"]
        
        for session_id in sessions:
            # 변환 단계
            converter = DeepEvalTestCaseConverterTool()
            converter_result = converter.execute(
                session_id=session_id,
                review_logs_path=str(review_logs_structure),
                output_path=str(temp_base_dir / "test_cases")
            )
            
            assert converter_result.success is True
            assert converter_result.data["session_id"] == session_id
            
            # 실행 단계
            executor = DeepEvalExecutorTool()
            executor.results_base_path = str(temp_base_dir / "results")
            
            test_cases_dir = temp_base_dir / "test_cases" / session_id
            
            with patch.object(executor, '_run_evaluation', return_value={"success": True, "data": {"executed": True}}):
                executor_result = executor.execute(
                    session_id=session_id,
                    test_case_path=str(test_cases_dir)
                )
            
            assert executor_result.success is True
            assert executor_result.data["session_id"] == session_id
        
        # 모든 세션의 결과 디렉토리 확인
        results_base = temp_base_dir / "results"
        for session_id in sessions:
            session_results_dir = results_base / session_id
            assert session_results_dir.exists()
    
    def test_concurrent_execution_simulation(self, review_logs_structure, temp_base_dir, mock_env_vars):
        """동시 실행 시뮬레이션 테스트"""
        import concurrent.futures
        
        def process_session(session_id):
            # 변환
            converter = DeepEvalTestCaseConverterTool()
            converter_result = converter.execute(
                session_id=session_id,
                review_logs_path=str(review_logs_structure),
                output_path=str(temp_base_dir / f"concurrent_{session_id}")
            )
            
            if not converter_result.success:
                return {"session_id": session_id, "success": False, "error": "converter_failed"}
            
            # 실행
            executor = DeepEvalExecutorTool()
            executor.results_base_path = str(temp_base_dir / "concurrent_results")
            
            test_cases_dir = temp_base_dir / "concurrent_test_cases" / session_id
            
            with patch.object(executor, '_run_evaluation', return_value={"success": True, "data": {"executed": True}}):
                executor_result = executor.execute(
                    session_id=session_id,
                    test_case_path=str(test_cases_dir)
                )
            
            return {
                "session_id": session_id,
                "success": executor_result.success,
                "error": executor_result.error_message if not executor_result.success else None
            }
        
        sessions = [f"concurrent-session-{i}" for i in range(3)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_session = {executor.submit(process_session, session): session for session in sessions}
            results = []
            
            for future in concurrent.futures.as_completed(future_to_session):
                result = future.result()
                results.append(result)
        
        # 모든 세션이 성공했는지 확인
        assert len(results) == 3
        for result in results:
            assert result["success"] is True
            assert result["error"] is None
    
    def test_large_dataset_processing(self, temp_base_dir, sample_session_id, mock_env_vars):
        """대용량 데이터셋 처리 테스트"""
        # 대량의 리뷰 로그 생성
        large_logs_dir = temp_base_dir / "large_logs"
        
        # 10개 저장소, 각각 5개 커밋, 각각 2개 모델 = 100개 리뷰 로그
        repos_count = 10
        commits_per_repo = 5
        models = ["gemini-2.5-pro", "claude-sonnet-4"]
        
        sample_log_data = {
            "prompt": [{"role": "user", "content": "Review this code"}],
            "review_response": {"issues": [], "summary": "No issues found"}
        }
        
        for repo_idx in range(repos_count):
            for commit_idx in range(commits_per_repo):
                for model in models:
                    log_dir = large_logs_dir / f"repo-{repo_idx}" / f"commit-{commit_idx}" / model
                    log_dir.mkdir(parents=True, exist_ok=True)
                    
                    log_file = log_dir / f"review_log_{commit_idx}.json"
                    with open(log_file, 'w', encoding='utf-8') as f:
                        json.dump(sample_log_data, f)
        
        # 변환 실행
        converter = DeepEvalTestCaseConverterTool()
        converter_result = converter.execute(
            session_id=sample_session_id,
            review_logs_path=str(large_logs_dir),
            output_path=str(temp_base_dir / "large_test_cases")
        )
        
        assert converter_result.success is True
        
        # 많은 파일이 변환되었는지 확인
        converted_files = converter_result.data["converted_files"]
        assert len(converted_files) == len(models)  # 2개 모델
        
        for model_name, file_info in converted_files.items():
            # 각 모델별로 50개 테스트 케이스 (10 repos * 5 commits)
            assert file_info["test_case_count"] == repos_count * commits_per_repo
        
        # 실행 단계
        executor = DeepEvalExecutorTool()
        executor.results_base_path = str(temp_base_dir / "large_results")
        
        test_cases_dir = temp_base_dir / "large_test_cases" / sample_session_id
        
        with patch.object(executor, '_run_evaluation', return_value={"success": True, "data": {"executed": True}}):
            executor_result = executor.execute(
                session_id=sample_session_id,
                test_case_path=str(test_cases_dir),
                parallel_workers=4  # 병렬 처리
            )
        
        assert executor_result.success is True
        assert executor_result.data["total_evaluations"] == len(models)


class TestDataConsistency:
    """데이터 일관성 테스트"""
    
    def test_test_case_format_consistency(self, review_logs_structure, temp_base_dir, sample_session_id):
        """테스트 케이스 형식 일관성 테스트"""
        converter = DeepEvalTestCaseConverterTool()
        
        result = converter.execute(
            session_id=sample_session_id,
            review_logs_path=str(review_logs_structure),
            output_path=str(temp_base_dir)
        )
        
        assert result.success is True
        
        # 모든 변환된 파일의 형식 일관성 확인
        for model_name, file_info in result.data["converted_files"].items():
            test_cases_file = Path(file_info["file_path"])
            
            with open(test_cases_file, 'r', encoding='utf-8') as f:
                test_cases = json.load(f)
            
            for i, test_case in enumerate(test_cases):
                # 필수 필드 존재 확인
                assert "input" in test_case, f"Model {model_name}, case {i}: missing 'input'"
                assert "actual_output" in test_case, f"Model {model_name}, case {i}: missing 'actual_output'"
                assert "expected_output" in test_case, f"Model {model_name}, case {i}: missing 'expected_output'"
                
                # JSON 형식 유효성 확인
                try:
                    json.loads(test_case["input"])
                except json.JSONDecodeError:
                    pytest.fail(f"Model {model_name}, case {i}: 'input' is not valid JSON")
                
                try:
                    json.loads(test_case["actual_output"])
                except json.JSONDecodeError:
                    pytest.fail(f"Model {model_name}, case {i}: 'actual_output' is not valid JSON")
    
    def test_metadata_consistency(self, review_logs_structure, temp_base_dir, sample_session_id, mock_env_vars):
        """메타데이터 일관성 테스트"""
        # 변환기 메타데이터
        converter = DeepEvalTestCaseConverterTool()
        converter_result = converter.execute(
            session_id=sample_session_id,
            review_logs_path=str(review_logs_structure),
            output_path=str(temp_base_dir / "test_cases")
        )
        
        assert converter_result.success is True
        
        converter_metadata_path = Path(converter_result.data["metadata_path"])
        with open(converter_metadata_path, 'r', encoding='utf-8') as f:
            converter_metadata = json.load(f)
        
        # 실행기 메타데이터
        executor = DeepEvalExecutorTool()
        executor.results_base_path = str(temp_base_dir / "results")
        
        test_cases_dir = temp_base_dir / "test_cases" / sample_session_id
        
        with patch.object(executor, '_run_evaluation', return_value={"success": True, "data": {"executed": True}}):
            executor_result = executor.execute(
                session_id=sample_session_id,
                test_case_path=str(test_cases_dir)
            )
        
        assert executor_result.success is True
        
        executor_metadata_path = Path(executor_result.data["metadata_path"])
        with open(executor_metadata_path, 'r', encoding='utf-8') as f:
            executor_metadata = json.load(f)
        
        # 일관성 확인
        assert converter_metadata["selvage_version"] == executor_metadata["selvage_version"]
        assert executor_metadata["session_id"] == sample_session_id
        assert executor_metadata["deep_eval_test_case_path"] == str(test_cases_dir)
```

## 7. 에러 처리 및 복구 테스트

```python
class TestErrorHandling:
    """에러 처리 테스트"""
    
    def test_converter_corrupted_log_files(self, temp_base_dir, sample_session_id):
        """손상된 로그 파일 처리 테스트"""
        logs_dir = temp_base_dir / "corrupted_logs"
        
        # 정상 파일과 손상된 파일 혼재
        repo_dir = logs_dir / "test-repo" / "commit123" / "gemini-2.5-pro"
        repo_dir.mkdir(parents=True, exist_ok=True)
        
        # 정상 파일
        normal_log = {
            "prompt": [{"role": "user", "content": "Review"}],
            "review_response": {"summary": "Good"}
        }
        with open(repo_dir / "normal.json", 'w', encoding='utf-8') as f:
            json.dump(normal_log, f)
        
        # 손상된 JSON 파일
        with open(repo_dir / "corrupted.json", 'w') as f:
            f.write("{ corrupted json")
        
        # 필드 누락 파일
        incomplete_log = {"prompt": [{"role": "user", "content": "Review"}]}  # review_response 누락
        with open(repo_dir / "incomplete.json", 'w', encoding='utf-8') as f:
            json.dump(incomplete_log, f)
        
        # 변환 실행
        converter = DeepEvalTestCaseConverterTool()
        result = converter.execute(
            session_id=sample_session_id,
            review_logs_path=str(logs_dir),
            output_path=str(temp_base_dir / "output")
        )
        
        # 정상 파일만 처리되어 성공해야 함
        assert result.success is True
        
        # 정상 파일만 변환되었는지 확인
        for model_name, file_info in result.data["converted_files"].items():
            assert file_info["test_case_count"] == 1  # 정상 파일 1개만
    
    def test_executor_missing_deepeval_dependencies(self, test_cases_directory, sample_session_id, mock_env_vars):
        """DeepEval 의존성 누락 테스트"""
        executor = DeepEvalExecutorTool()
        
        # DeepEval import 실패 시뮬레이션
        with patch.object(executor, '_run_evaluation', side_effect=ImportError("No module named 'deepeval'")):
            result = executor.execute(
                session_id=sample_session_id,
                test_case_path=str(test_cases_directory)
            )
        
        assert result.success is False
        assert "DeepEval 평가 실행 실패" in result.error_message
    
    def test_disk_space_simulation(self, review_logs_structure, temp_base_dir, sample_session_id):
        """디스크 공간 부족 시뮬레이션 테스트"""
        converter = DeepEvalTestCaseConverterTool()
        
        # 파일 쓰기 실패 시뮬레이션
        original_open = open
        
        def mock_open(*args, **kwargs):
            if 'w' in args[1] if len(args) > 1 else kwargs.get('mode', ''):
                if 'test_cases.json' in args[0]:
                    raise OSError("No space left on device")
            return original_open(*args, **kwargs)
        
        with patch('builtins.open', side_effect=mock_open):
            result = converter.execute(
                session_id=sample_session_id,
                review_logs_path=str(review_logs_structure),
                output_path=str(temp_base_dir / "output")
            )
        
        assert result.success is False
        assert "DeepEval 테스트 케이스 변환 실패" in result.error_message
    
    def test_permission_denied_scenarios(self, review_logs_structure, sample_session_id, temp_base_dir, mock_tool_paths):
        """권한 부족 시나리오 테스트"""
        converter = DeepEvalTestCaseConverterTool()
        
        # 권한 거부 시뮬레이션 - 실제 파일 생성 시 권한 에러 발생시키기
        restricted_dir = temp_base_dir / "restricted"
        restricted_dir.mkdir(exist_ok=True)
        
        # mkdir에서 권한 에러 시뮬레이션
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            result = converter.execute(
                session_id=sample_session_id,
                review_logs_path=str(review_logs_structure),
                output_path=str(restricted_dir / "output")
            )
        
        # 권한 에러로 인해 실패해야 함
        assert result.success is False
        assert "DeepEval 테스트 케이스 변환 실패" in result.error_message
    
    def test_network_timeout_simulation(self, test_cases_directory, sample_session_id, mock_env_vars):
        """네트워크 타임아웃 시뮬레이션 테스트"""
        executor = DeepEvalExecutorTool()
        
        # DeepEval API 호출 타임아웃 시뮬레이션
        import socket
        
        with patch.object(executor, '_run_evaluation', side_effect=socket.timeout("API timeout")):
            result = executor.execute(
                session_id=sample_session_id,
                test_case_path=str(test_cases_directory)
            )
        
        assert result.success is False
        assert "DeepEval 평가 실행 실패" in result.error_message


class TestRecoveryMechanisms:
    """복구 메커니즘 테스트"""
    
    def test_partial_failure_recovery(self, temp_base_dir, sample_session_id):
        """부분 실패 복구 테스트"""
        # 일부 모델은 성공, 일부는 실패하는 시나리오
        logs_dir = temp_base_dir / "partial_logs"
        
        # 성공할 모델 데이터
        success_dir = logs_dir / "repo1" / "commit1" / "gemini-2.5-pro"
        success_dir.mkdir(parents=True, exist_ok=True)
        
        success_log = {
            "prompt": [{"role": "user", "content": "Review"}],
            "review_response": {"summary": "Success"}
        }
        with open(success_dir / "success.json", 'w', encoding='utf-8') as f:
            json.dump(success_log, f)
        
        # 실패할 모델 데이터 (빈 디렉토리)
        fail_dir = logs_dir / "repo1" / "commit1" / "claude-sonnet-4"
        fail_dir.mkdir(parents=True, exist_ok=True)
        # 로그 파일 없음
        
        converter = DeepEvalTestCaseConverterTool()
        result = converter.execute(
            session_id=sample_session_id,
            review_logs_path=str(logs_dir),
            output_path=str(temp_base_dir / "output")
        )
        
        # 부분 성공이라도 전체적으로는 성공으로 간주
        assert result.success is True
        
        # 성공한 모델만 결과에 포함
        converted_files = result.data["converted_files"]
        assert "gemini-2.5-pro" in converted_files
        assert "claude-sonnet-4" not in converted_files or converted_files["claude-sonnet-4"]["test_case_count"] == 0
    
    def test_retry_mechanism_simulation(self, test_cases_directory, sample_session_id, mock_env_vars):
        """재시도 메커니즘 시뮬레이션 테스트"""
        executor = DeepEvalExecutorTool()
        
        # 첫 번째 시도는 실패, 두 번째는 성공하는 시뮬레이션
        call_count = 0
        
        def mock_run_evaluation(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": False, "error": "Temporary failure"}
            else:
                return {"success": True, "data": {"executed": True}}
        
        with patch.object(executor, '_run_evaluation', side_effect=mock_run_evaluation):
            # 첫 번째 실행 (실패)
            result1 = executor.execute(
                session_id=sample_session_id,
                test_case_path=str(test_cases_directory)
            )
            
            # 두 번째 실행 (성공)
            result2 = executor.execute(
                session_id=sample_session_id,
                test_case_path=str(test_cases_directory)
            )
        
        # 첫 번째는 실패, 두 번째는 성공
        assert result1.success is True  # 전체 프로세스는 성공 (개별 모델 실패는 있을 수 있음)
        assert result2.success is True
```

## 8. 성능 및 에지 케이스 테스트

```python
class TestPerformance:
    """성능 테스트"""
    
    def test_large_json_file_handling(self, temp_base_dir, sample_session_id, mock_tool_paths, performance_thresholds):
        """대용량 JSON 파일 처리 테스트"""
        logs_dir = temp_base_dir / "large_json_logs"
        repo_dir = logs_dir / "large-repo" / "large-commit" / "gemini-2.5-pro"
        repo_dir.mkdir(parents=True, exist_ok=True)
        
        # 대용량 프롬프트 생성 (약 1MB)
        large_prompt = [{"role": "user", "content": "x" * 100000}] * 10
        large_response = {
            "issues": [
                {
                    "type": "performance",
                    "description": "y" * 10000,
                    "suggestion": "z" * 10000
                }
            ] * 100,
            "summary": "Large summary " * 1000
        }
        
        large_log = {
            "prompt": large_prompt,
            "review_response": large_response
        }
        
        log_file = repo_dir / "large_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(large_log, f)
        
        # 처리 시간 측정
        import time
        start_time = time.time()
        
        converter = DeepEvalTestCaseConverterTool()
        result = converter.execute(
            session_id=sample_session_id,
            review_logs_path=str(logs_dir),
            output_path=str(temp_base_dir / "large_output")
        )
        
        processing_time = time.time() - start_time
        
        assert result.success is True
        assert processing_time < performance_thresholds["processing_time"]
        
        # 결과 파일 크기 확인
        for model_name, file_info in result.data["converted_files"].items():
            result_file = Path(file_info["file_path"])
            assert result_file.stat().st_size > 0
    
    def test_memory_usage_with_many_files(self, temp_base_dir, sample_session_id, mock_tool_paths, performance_thresholds):
        """다수 파일 처리 시 메모리 사용량 테스트"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 100개 파일 생성
        logs_dir = temp_base_dir / "many_files_logs"
        
        for i in range(100):
            repo_dir = logs_dir / f"repo-{i}" / f"commit-{i}" / "gemini-2.5-pro"
            repo_dir.mkdir(parents=True, exist_ok=True)
            
            log_data = {
                "prompt": [{"role": "user", "content": f"Review file {i}"}],
                "review_response": {"summary": f"Review {i} completed"}
            }
            
            with open(repo_dir / f"log_{i}.json", 'w', encoding='utf-8') as f:
                json.dump(log_data, f)
        
        converter = DeepEvalTestCaseConverterTool()
        result = converter.execute(
            session_id=sample_session_id,
            review_logs_path=str(logs_dir),
            output_path=str(temp_base_dir / "many_files_output")
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert result.success is True
        assert memory_increase < performance_thresholds["memory_increase"]
        assert result.data["converted_files"]["gemini-2.5-pro"]["test_case_count"] == 100
    
    def test_concurrent_session_performance(self, review_logs_structure, temp_base_dir, mock_env_vars):
        """동시 세션 성능 테스트"""
        import threading
        import time
        
        def process_session(session_id):
            start_time = time.time()
            
            converter = DeepEvalTestCaseConverterTool()
            result = converter.execute(
                session_id=session_id,
                review_logs_path=str(review_logs_structure),
                output_path=str(temp_base_dir / f"concurrent_{session_id}")
            )
            
            processing_time = time.time() - start_time
            results_list.append({
                "session_id": session_id,
                "success": result.success,
                "processing_time": processing_time
            })
        
        # 5개 동시 세션
        sessions = [f"perf-session-{i}" for i in range(5)]
        results_list = []
        threads = []
        
        overall_start = time.time()
        
        for session_id in sessions:
            thread = threading.Thread(target=process_session, args=(session_id, results_list))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
        
        overall_time = time.time() - overall_start
        
        # 모든 세션 성공 확인
        assert len(results_list) == 5
        for result in results_list:
            assert result["success"] is True
            assert result["processing_time"] < 20.0  # 개별 세션 20초 이내
        
        # 전체 처리 시간이 순차 처리보다 빠른지 확인 (동시 처리 효과)
        assert overall_time < 60.0  # 전체 60초 이내


class TestEdgeCases:
    """에지 케이스 테스트"""
    
    def test_empty_prompt_and_response(self, temp_base_dir, sample_session_id):
        """빈 프롬프트 및 응답 처리 테스트"""
        logs_dir = temp_base_dir / "empty_content_logs"
        repo_dir = logs_dir / "empty-repo" / "empty-commit" / "gemini-2.5-pro"
        repo_dir.mkdir(parents=True, exist_ok=True)
        
        # 빈 프롬프트와 응답
        empty_log = {
            "prompt": [],
            "review_response": {}
        }
        
        with open(repo_dir / "empty_log.json", 'w', encoding='utf-8') as f:
            json.dump(empty_log, f)
        
        converter = DeepEvalTestCaseConverterTool()
        result = converter.execute(
            session_id=sample_session_id,
            review_logs_path=str(logs_dir),
            output_path=str(temp_base_dir / "empty_output")
        )
        
        # 빈 데이터는 스킵되어야 함
        assert result.success is False  # 유효한 로그가 없으므로 실패
        assert "리뷰 로그 파일을 찾을 수 없습니다" in result.error_message
    
    def test_special_characters_in_paths(self, temp_base_dir, sample_session_id):
        """경로에 특수 문자 포함 테스트"""
        # 특수 문자가 포함된 디렉토리 이름
        logs_dir = temp_base_dir / "special chars logs & symbols"
        repo_dir = logs_dir / "repo with spaces" / "commit-with-dashes" / "gemini-2.5-pro"
        repo_dir.mkdir(parents=True, exist_ok=True)
        
        log_data = {
            "prompt": [{"role": "user", "content": "Review with special chars: @#$%^&*()"}],
            "review_response": {"summary": "Special chars handled"}
        }
        
        with open(repo_dir / "special_log.json", 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False)
        
        converter = DeepEvalTestCaseConverterTool()
        result = converter.execute(
            session_id=sample_session_id,
            review_logs_path=str(logs_dir),
            output_path=str(temp_base_dir / "special output")
        )
        
        assert result.success is True
        
        # 이모지가 보존되었는지 확인
        for model_name, file_info in result.data["converted_files"].items():
            test_cases_file = Path(file_info["file_path"])
            
            with open(test_cases_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "🤖" in content
                assert "📝" in content
                assert "😊" in content
    
    def test_extremely_long_session_id(self, review_logs_structure, temp_base_dir):
        """매우 긴 세션 ID 처리 테스트"""
        # 255자 세션 ID (파일 시스템 제한 테스트)
        long_session_id = "a" * 200 + "-session-2024"
        
        converter = DeepEvalTestCaseConverterTool()
        result = converter.execute(
            session_id=long_session_id,
            review_logs_path=str(review_logs_structure),
            output_path=str(temp_base_dir / "long_session_output")
        )
        
        # 파일 시스템이 지원하면 성공, 그렇지 않으면 실패
        if result.success:
            assert result.data["session_id"] == long_session_id
        else:
            assert "DeepEval 테스트 케이스 변환 실패" in result.error_message
    
    def test_unicode_emoji_in_content(self, temp_base_dir, sample_session_id):
        """유니코드 이모지 포함 콘텐츠 테스트"""
        logs_dir = temp_base_dir / "emoji_logs"
        repo_dir = logs_dir / "emoji-repo" / "emoji-commit" / "gemini-2.5-pro"
        repo_dir.mkdir(parents=True, exist_ok=True)
        
        emoji_log = {
            "prompt": [
                {"role": "system", "content": "You are a helpful assistant 🤖"},
                {"role": "user", "content": "Please review this code 📝✨"}
            ],
            "review_response": {
                "issues": [
                    {
                        "type": "style",
                        "description": "Consider adding more emojis for better readability 😊",
                        "suggestion": "Use 🔧 for fixes and ⚡ for performance improvements"
                    }
                ],
                "summary": "Code looks good! 👍✅"
            }
        }
        
        with open(repo_dir / "emoji_log.json", 'w', encoding='utf-8') as f:
            json.dump(emoji_log, f, ensure_ascii=False)
        
        converter = DeepEvalTestCaseConverterTool()
        result = converter.execute(
            session_id=sample_session_id,
            review_logs_path=str(logs_dir),
            output_path=str(temp_base_dir / "emoji_output")
        )
        
        assert result.success is True
        
        # 이모지가 보존되었는지 확인
        for model_name, file_info in result.data["converted_files"].items():
            test_cases_file = Path(file_info["file_path"])
            
            with open(test_cases_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "🤖" in content
                assert "📝" in content
                assert "😊" in content
    
    def test_invalid_session_id_characters(self, review_logs_structure, temp_base_dir):
        """잘못된 세션 ID 문자 테스트"""
        # 파일 시스템에서 허용되지 않는 문자들
        invalid_session_ids = [
            "session/with/slashes",
            "session:with:colons",
            "session*with*asterisks",
            "session?with?questions",
            "session<with>brackets"
        ]
        
        converter = DeepEvalTestCaseConverterTool()
        
        for invalid_id in invalid_session_ids:
            result = converter.execute(
                session_id=invalid_id,
                review_logs_path=str(review_logs_structure),
                output_path=str(temp_base_dir / "invalid_output")
            )
            
            # 플랫폼에 따라 처리가 다를 수 있음
            # 일부는 성공할 수도, 일부는 실패할 수도 있음
            if not result.success:
                assert "DeepEval 테스트 케이스 변환 실패" in result.error_message


if __name__ == "__main__":
    # 특정 테스트만 실행하는 예제
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "not performance"  # 성능 테스트 제외
    ])
```

## 9. 테스트 실행 가이드

### 9.1 테스트 실행 명령어

```bash
# 전체 테스트 실행
pytest docs/testing/evaluation-deepeval-test-cases.md -v

# 특정 테스트 클래스 실행
pytest docs/testing/evaluation-deepeval-test-cases.md::TestDeepEvalTestCaseConverterTool -v

# 특정 기능별 테스트 실행
pytest docs/testing/evaluation-deepeval-test-cases.md -k "converter" -v
pytest docs/testing/evaluation-deepeval-test-cases.md -k "executor" -v
pytest docs/testing/evaluation-deepeval-test-cases.md -k "integration" -v

# 커버리지 포함 실행
pytest docs/testing/evaluation-deepeval-test-cases.md \
  --cov=selvage_eval.tools.deepeval_test_case_converter_tool \
  --cov=selvage_eval.tools.deepeval_executor_tool \
  --cov-report=html \
  --cov-report=term

# 성능 테스트만 실행
pytest docs/testing/evaluation-deepeval-test-cases.md -k "performance" -v

# 에러 처리 테스트만 실행
pytest docs/testing/evaluation-deepeval-test-cases.md -k "error" -v

# 병렬 실행 (pytest-xdist 사용)
pytest docs/testing/evaluation-deepeval-test-cases.md -n auto -v

# 상세한 출력과 함께 실행
pytest docs/testing/evaluation-deepeval-test-cases.md -v -s --tb=long
```

### 9.2 테스트 환경 설정

#### 필수 의존성

```bash
# 기본 테스트 도구
pip install pytest pytest-mock pytest-asyncio

# 커버리지 도구
pip install pytest-cov

# 병렬 실행
pip install pytest-xdist

# 성능 모니터링
pip install psutil

# DeepEval (실제 통합 테스트용)
pip install deepeval
```

#### 환경 변수 설정

```bash
# 테스트용 API 키 (실제 키 또는 테스트용 더미 키)
export OPENAI_API_KEY="your-test-openai-key"
export GEMINI_API_KEY="your-test-gemini-key"

# 선택적: 테스트 로그 레벨
export PYTEST_LOG_LEVEL="DEBUG"
```

### 9.3 테스트 커버리지 목표

- **DeepEvalTestCaseConverterTool**: 95% 이상
- **DeepEvalExecutorTool**: 95% 이상
- **데이터 클래스들**: 100%
- **에러 처리 로직**: 90% 이상
- **통합 테스트**: 전체 워크플로우 커버리지

### 9.4 테스트 마커 설정

pytest.ini 파일에 다음과 같이 마커를 정의합니다:

```ini
[tool:pytest]
markers =
    unit: 단위 테스트 (빠른 실행, Mock 사용)
    integration: 통합 테스트 (실제 API 호출 포함)
    performance: 성능 테스트 (시간이 오래 걸림)
    slow: 느린 테스트 (대용량 데이터 처리 등)
```

### 9.5 CI/CD 통합

```yaml
# .github/workflows/test-deepeval.yml
name: Test DeepEval Components

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-mock pytest-xdist psutil
    
    - name: Run unit tests (without API integration)
      env:
        CI: "true"
        OPENAI_API_KEY: "test-dummy-key"
        GEMINI_API_KEY: "test-dummy-key"
      run: |
        pytest docs/testing/evaluation-deepeval-test-cases.md \
          --cov=selvage_eval.tools.deepeval_test_case_converter_tool \
          --cov=selvage_eval.tools.deepeval_executor_tool \
          --cov-report=xml \
          --cov-fail-under=85 \
          -v \
          -m "not integration" \
          --tb=short
    
    - name: Run integration tests (with real API keys - manual trigger only)
      if: github.event_name == 'workflow_dispatch'
      env:
        CI: "true"
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      run: |
        pytest docs/testing/evaluation-deepeval-test-cases.md \
          -m "integration" \
          -v \
          --tb=short
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: deepeval
        name: deepeval-coverage
```

### 9.5 테스트 데이터 관리

#### 테스트 데이터 구조

```
tests/fixtures/
├── sample_review_logs/
│   └── repo-name/
│       └── commit-id/
│           └── model-name/
│               └── review_log.json
├── expected_test_cases/
│   └── session-id/
│       └── model-name/
│           └── test_cases.json
└── expected_results/
    └── session-id/
        └── model-name/
            └── evaluation_results.json
```

#### 테스트 데이터 생성 스크립트

```python
# scripts/generate_test_data.py
"""테스트 데이터 생성 스크립트"""

import json
from pathlib import Path
from datetime import datetime

def generate_sample_review_logs(output_dir: Path):
    """샘플 리뷰 로그 생성"""
    repos = ["test-repo-1", "test-repo-2"]
    commits = ["abc123", "def456", "ghi789"]
    models = ["gemini-2.5-pro", "claude-sonnet-4"]
    
    for repo in repos:
        for commit in commits:
            for model in models:
                log_dir = output_dir / repo / commit / model
                log_dir.mkdir(parents=True, exist_ok=True)
                
                log_data = {
                    "prompt": [
                        {"role": "system", "content": "You are a code reviewer."},
                        {"role": "user", "content": f"Review commit {commit}"}
                    ],
                    "review_response": {
                        "issues": [
                            {
                                "type": "bug",
                                "description": f"Issue in {repo}",
                                "suggestion": "Fix suggestion"
                            }
                        ],
                        "summary": f"Review for {repo}/{commit}",
                        "score": 7.5
                    },
                    "metadata": {
                        "model": model,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                with open(log_dir / f"review_log_{commit}.json", 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    test_data_dir = Path("tests/fixtures/sample_review_logs")
    generate_sample_review_logs(test_data_dir)
    print(f"테스트 데이터 생성 완료: {test_data_dir}")
```

### 9.6 테스트 실행 체크리스트

#### 개발 단계
- [ ] 단위 테스트 통과
- [ ] 커버리지 90% 이상
- [ ] 에러 처리 테스트 통과
- [ ] 메모리 누수 없음

#### 통합 테스트 단계
- [ ] 전체 워크플로우 테스트 통과
- [ ] 다중 세션 처리 정상
- [ ] 대용량 데이터 처리 성능 기준 만족
- [ ] 동시 실행 안정성 확인

#### 배포 전 단계
- [ ] 모든 테스트 통과
- [ ] 성능 테스트 기준 만족
- [ ] 에지 케이스 처리 확인
- [ ] 실제 환경 통합 테스트 완료

이 테스트 케이스 문서는 DeepEval 평가 시스템의 모든 구성 요소를 포괄적으로 검증하며, 개발부터 배포까지 전체 생명주기를 지원합니다. 