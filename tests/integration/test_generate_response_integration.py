"""generate_response 메서드 통합 테스트

실제 message, plan, tool_results를 구성하여 최종 응답 생성을 검증합니다.
실제 작업 흐름과 유사한 시나리오를 사용합니다.
"""

import pytest
import os
import json
from unittest.mock import Mock

from selvage_eval.agent.core_agent import SelvageEvaluationAgent
from selvage_eval.tools.execution_plan import ExecutionPlan
from selvage_eval.tools.tool_call import ToolCall
from selvage_eval.tools.tool_result import ToolResult


@pytest.mark.integration
@pytest.mark.slow
class TestGenerateResponseIntegration:
    """generate_response 메서드 실제 LLM 호출 통합 테스트"""

    @pytest.fixture
    def real_config(self, temp_dir):
        """실제 설정 객체 (API 키 필요)"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("GEMINI_API_KEY가 설정되지 않아 통합 테스트를 건너뜁니다")
        
        config = Mock()
        config.agent_model = "gemini-2.5-pro"
        config.review_models = []
        config.target_repositories = []
        config.commits_per_repo = 10
        
        # workflow mock
        workflow_mock = Mock()
        workflow_mock.model_dump = Mock(return_value={"type": "test_workflow"})
        config.workflow = workflow_mock
        
        # deepeval mock
        deepeval_mock = Mock()
        deepeval_mock.metrics = []
        config.deepeval = deepeval_mock
        
        evaluation_mock = Mock()
        evaluation_mock.output_dir = str(temp_dir)
        config.evaluation = evaluation_mock
        
        config.get_output_path = lambda *args: str(temp_dir / "_".join(args))
        return config

    @pytest.fixture
    def agent(self, real_config, temp_dir):
        """실제 응답 생성이 가능한 에이전트 인스턴스"""
        agent = SelvageEvaluationAgent(real_config, work_dir=str(temp_dir))
        return agent

    @pytest.fixture
    def sample_project_structure(self, temp_dir):
        """실제 프로젝트 구조를 모방한 테스트 환경"""
        # README.md 생성
        (temp_dir / "README.md").write_text("""# 테스트 프로젝트

이것은 Selvage 평가 에이전트 테스트를 위한 프로젝트입니다.

## 구조
- src/: 소스 코드
- tests/: 테스트 파일
- config.json: 설정 파일

## 기능
- 파일 읽기/쓰기
- 디렉토리 탐색
- 명령어 실행
""", encoding="utf-8")
        
        # config.json 생성
        config_data = {
            "project_name": "test-project",
            "version": "1.0.0",
            "author": "테스트 작성자",
            "dependencies": ["pytest", "mock"],
            "description": "통합 테스트용 프로젝트"
        }
        (temp_dir / "config.json").write_text(
            json.dumps(config_data, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        
        # 소스 디렉토리 구조
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def main():
    print("안녕하세요, 테스트 프로젝트입니다!")
    return "success"

if __name__ == "__main__":
    main()
""", encoding="utf-8")
        
        # 테스트 디렉토리
        tests_dir = temp_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_main.py").write_text("""import pytest
from src.main import main

def test_main():
    result = main()
    assert result == "success"
""", encoding="utf-8")
        
        return temp_dir

    def test_generate_response_file_analysis_scenario(self, agent: SelvageEvaluationAgent, sample_project_structure):
        """파일 분석 시나리오에 대한 응답 생성 테스트"""
        # Given: 프로젝트 분석 요청과 도구 실행 결과
        user_query = "프로젝트 구조를 분석하고 주요 파일들의 내용을 요약해주세요"
        temp_dir = sample_project_structure
        
        # 실행 계획 구성 (실제 plan_execution 결과를 모방)
        execution_plan = ExecutionPlan(
            intent_summary="프로젝트 구조 분석 및 주요 파일 요약",
            confidence=0.9,
            parameters={},
            tool_calls=[
                ToolCall(
                    tool="list_directory",
                    params={"directory_path": str(temp_dir)},
                    rationale="프로젝트 루트 디렉토리 구조 파악"
                ),
                ToolCall(
                    tool="read_file",
                    params={"file_path": str(temp_dir / "README.md")},
                    rationale="프로젝트 개요 파악"
                ),
                ToolCall(
                    tool="read_file",
                    params={"file_path": str(temp_dir / "config.json")},
                    rationale="프로젝트 설정 정보 확인"
                )
            ],
            safety_check="읽기 전용 작업으로 안전함",
            expected_outcome="프로젝트 구조 및 주요 파일 내용 요약"
        )
        
        # 도구 실행 결과 구성 (실제 도구 실행 결과를 모방)
        tool_results = [
            {
                "tool": "list_directory",
                "result": ToolResult(
                    success=True,
                    data={
                        "items": [
                            {"name": "README.md", "type": "file", "size": 245},
                            {"name": "config.json", "type": "file", "size": 158},
                            {"name": "src", "type": "directory"},
                            {"name": "tests", "type": "directory"}
                        ]
                    },
                    error_message=None
                ),
                "rationale": "프로젝트 루트 디렉토리 구조 파악"
            },
            {
                "tool": "read_file",
                "result": ToolResult(
                    success=True,
                    data={"content": (temp_dir / "README.md").read_text(encoding="utf-8")},
                    error_message=None
                ),
                "rationale": "프로젝트 개요 파악"
            },
            {
                "tool": "read_file",
                "result": ToolResult(
                    success=True,
                    data={"content": (temp_dir / "config.json").read_text(encoding="utf-8")},
                    error_message=None
                ),
                "rationale": "프로젝트 설정 정보 확인"
            }
        ]
        
        # When: 실제 LLM 호출로 응답 생성
        response = agent.generate_response(user_query, execution_plan, tool_results)
        
        # Then: 적절한 응답이 생성되는지 검증
        assert isinstance(response, str)
        assert len(response) > 0
        
        # 응답에 주요 정보가 포함되어 있는지 확인
        assert "프로젝트" in response
        assert "구조" in response or "파일" in response
        
        print(f"Generated response length: {len(response)}")
        print(f"Response preview: {response[:200]}...")
        
        # 응답이 한국어로 작성되었는지 확인
        korean_chars = sum(1 for char in response if '\uac00' <= char <= '\ud7af')
        assert korean_chars > 10, "응답에 충분한 한국어 내용이 포함되어야 합니다"

    def test_generate_response_file_creation_scenario(self, agent, temp_dir):
        """파일 생성 시나리오에 대한 응답 생성 테스트"""
        # Given: 파일 생성 요청과 성공적인 실행 결과
        user_query = "새로운 설정 파일을 생성해주세요"
        
        execution_plan = ExecutionPlan(
            intent_summary="새 설정 파일 생성",
            confidence=0.8,
            parameters={},
            tool_calls=[
                ToolCall(
                    tool="write_file",
                    params={
                        "file_path": str(temp_dir / "new_config.yaml"),
                        "content": "app:\n  name: test-app\n  version: 1.0.0\n  debug: true"
                    },
                    rationale="새로운 YAML 설정 파일 생성"
                )
            ],
            safety_check="설정 파일 생성은 안전함",
            expected_outcome="새 설정 파일 생성 완료"
        )
        
        tool_results = [
            {
                "tool": "write_file",
                "result": ToolResult(
                    success=True,
                    data={"file_created": True, "path": str(temp_dir / "new_config.yaml")},
                    error_message=None
                ),
                "rationale": "새로운 YAML 설정 파일 생성"
            }
        ]
        
        # When: 응답 생성
        response = agent.generate_response(user_query, execution_plan, tool_results)
        
        # Then: 파일 생성 성공에 대한 응답 검증
        assert isinstance(response, str)
        assert len(response) > 0
        assert "설정" in response or "파일" in response
        assert "생성" in response or "완료" in response
        
        print(f"File creation response: {response}")

    def test_generate_response_error_handling_scenario(self, agent, temp_dir):
        """오류 처리 시나리오에 대한 응답 생성 테스트"""
        # Given: 실패한 도구 실행 결과
        user_query = "존재하지 않는 파일을 읽어주세요"
        
        execution_plan = ExecutionPlan(
            intent_summary="파일 읽기 시도",
            confidence=0.7,
            parameters={},
            tool_calls=[
                ToolCall(
                    tool="read_file",
                    params={"file_path": str(temp_dir / "nonexistent.txt")},
                    rationale="요청된 파일 읽기"
                )
            ],
            safety_check="읽기 작업으로 안전함",
            expected_outcome="파일 내용 반환"
        )
        
        tool_results = [
            {
                "tool": "read_file",
                "result": ToolResult(
                    success=False,
                    data=None,
                    error_message="파일을 찾을 수 없습니다: nonexistent.txt"
                ),
                "rationale": "요청된 파일 읽기"
            }
        ]
        
        # When: 오류 상황에 대한 응답 생성
        response = agent.generate_response(user_query, execution_plan, tool_results)
        
        # Then: 적절한 오류 설명이 포함된 응답 검증
        assert isinstance(response, str)
        assert len(response) > 0
        assert "파일" in response
        assert "없" in response or "오류" in response or "실패" in response
        
        print(f"Error handling response: {response}")

    def test_generate_response_with_conversation_context(self, agent, sample_project_structure):
        """대화 컨텍스트가 있는 상황에서의 응답 생성 테스트"""
        # Given: 이전 대화 컨텍스트가 있는 상황
        temp_dir = sample_project_structure
        
        # 이전 대화 설정
        previous_context = [
            {
                "user_message": "README.md 파일이 있나요?",
                "assistant_response": "네, README.md 파일이 존재합니다. 프로젝트 개요가 담겨 있습니다.",
                "tool_results": [{"tool": "file_exists", "result": {"exists": True}}],
                "timestamp": "2024-01-01T12:00:00Z"
            }
        ]
        agent.session_state.conversation_history = previous_context
        
        # 후속 질문
        user_query = "그 파일의 내용도 자세히 설명해주세요"
        
        execution_plan = ExecutionPlan(
            intent_summary="README.md 파일 내용 상세 설명",
            confidence=0.9,
            parameters={},
            tool_calls=[
                ToolCall(
                    tool="read_file",
                    params={"file_path": str(temp_dir / "README.md")},
                    rationale="README.md 내용 읽기"
                )
            ],
            safety_check="읽기 작업으로 안전함",
            expected_outcome="README.md 내용 상세 설명"
        )
        
        tool_results = [
            {
                "tool": "read_file",
                "result": ToolResult(
                    success=True,
                    data={"content": (temp_dir / "README.md").read_text(encoding="utf-8")},
                    error_message=None
                ),
                "rationale": "README.md 내용 읽기"
            }
        ]
        
        # When: 컨텍스트를 고려한 응답 생성
        response = agent.generate_response(user_query, execution_plan, tool_results)
        
        # Then: 컨텍스트를 반영한 적절한 응답 검증
        assert isinstance(response, str)
        assert len(response) > 0
        assert "README" in response or "파일" in response
        
        print(f"Context-aware response: {response}")

    def test_generate_response_complex_multi_step_scenario(self, agent, sample_project_structure):
        """복잡한 다단계 작업 시나리오에 대한 응답 생성 테스트"""
        # Given: 여러 도구를 사용한 복합 작업 결과
        temp_dir = sample_project_structure
        user_query = "프로젝트를 전체적으로 분석하고 보고서를 작성해주세요"
        
        execution_plan = ExecutionPlan(
            intent_summary="프로젝트 전체 분석 및 보고서 작성",
            confidence=0.95,
            parameters={},
            tool_calls=[
                ToolCall(tool="list_directory", params={"directory_path": str(temp_dir)}, rationale="루트 구조 파악"),
                ToolCall(tool="list_directory", params={"directory_path": str(temp_dir / "src")}, rationale="소스 구조 파악"),
                ToolCall(tool="read_file", params={"file_path": str(temp_dir / "README.md")}, rationale="프로젝트 문서 읽기"),
                ToolCall(tool="read_file", params={"file_path": str(temp_dir / "config.json")}, rationale="설정 정보 읽기"),
                ToolCall(tool="read_file", params={"file_path": str(temp_dir / "src" / "main.py")}, rationale="메인 코드 분석")
            ],
            safety_check="읽기 전용 분석 작업으로 안전함",
            expected_outcome="종합적인 프로젝트 분석 보고서"
        )
        
        # 실제 도구 실행을 통한 결과 생성
        tool_results = []
        for tool_call in execution_plan.tool_calls:
            result = agent.execute_tool(tool_call.tool, tool_call.params)
            tool_results.append({
                "tool": tool_call.tool,
                "result": result,
                "rationale": tool_call.rationale
            })
        
        # When: 복합 작업 결과에 대한 응답 생성
        response = agent.generate_response(user_query, execution_plan, tool_results)
        
        # Then: 종합적인 분석 결과가 포함된 응답 검증
        assert isinstance(response, str)
        assert len(response) > 100  # 상당한 길이의 응답이어야 함
        
        # 주요 분석 요소들이 포함되어 있는지 확인
        assert "프로젝트" in response
        assert "분석" in response or "구조" in response
        
        print(f"Complex analysis response length: {len(response)}")
        print(f"Response preview: {response[:300]}...")

    def test_generate_response_korean_content_handling(self, agent, temp_dir):
        """한글 내용 처리에 대한 응답 생성 테스트"""
        # Given: 한글 내용이 포함된 파일 작업
        korean_file_content = """
# 한국어 프로젝트 문서

## 개요
이 프로젝트는 한국어 자연어 처리를 위한 도구입니다.

## 주요 기능
1. 텍스트 분석
2. 감정 분석  
3. 키워드 추출

## 사용법
```python
from nlp_tool import KoreanAnalyzer

analyzer = KoreanAnalyzer()
result = analyzer.analyze("안녕하세요")
```

## 연락처
개발자: 김개발자
이메일: dev@example.com
        """
        
        korean_file_path = str(temp_dir / "한국어_문서.md")
        (temp_dir / "한국어_문서.md").write_text(korean_file_content, encoding="utf-8")
        
        user_query = "한국어 문서를 분석해주세요"
        
        execution_plan = ExecutionPlan(
            intent_summary="한국어 문서 내용 분석",
            confidence=0.9,
            parameters={},
            tool_calls=[
                ToolCall(
                    tool="read_file",
                    params={"file_path": korean_file_path},
                    rationale="한국어 문서 내용 읽기"
                )
            ],
            safety_check="문서 읽기 작업으로 안전함",
            expected_outcome="한국어 문서 내용 분석 결과"
        )
        
        tool_results = [
            {
                "tool": "read_file",
                "result": ToolResult(
                    success=True,
                    data={"content": korean_file_content},
                    error_message=None
                ),
                "rationale": "한국어 문서 내용 읽기"
            }
        ]
        
        # When: 한글 내용에 대한 응답 생성
        response = agent.generate_response(user_query, execution_plan, tool_results)
        
        # Then: 한글 내용이 적절히 처리된 응답 검증
        assert isinstance(response, str)
        assert len(response) > 0
        
        # 한국어 처리 검증
        korean_chars = sum(1 for char in response if '\uac00' <= char <= '\ud7af')
        assert korean_chars > 20, "충분한 한국어 내용이 응답에 포함되어야 합니다"
        
        # 문서의 주요 내용이 언급되었는지 확인
        assert "자연어" in response or "분석" in response or "프로젝트" in response
        
        print(f"Korean content response: {response[:200]}...")