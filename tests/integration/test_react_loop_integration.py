"""ReAct 루프 통합 테스트

실제 LLM 호출을 통해 plan_execution_loop의 전체 ReAct 패턴을 검증합니다.
Think-Act-Observe 사이클이 올바르게 작동하는지 확인합니다.
"""

import pytest
import os
import json
import stat
from unittest.mock import Mock

from selvage_eval.agent.core_agent import SelvageEvaluationAgent


@pytest.mark.integration
@pytest.mark.slow
class TestReActLoopIntegration:
    """ReAct 루프 패턴 실제 LLM 호출 통합 테스트"""

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
        """실제 ReAct 루프가 가능한 에이전트 인스턴스"""
        agent = SelvageEvaluationAgent(real_config, work_dir=str(temp_dir))
        return agent

    @pytest.fixture
    def sample_project_structure(self, temp_dir):
        """실제 프로젝트 구조를 모방한 테스트 환경"""
        # README.md 생성
        readme_file = temp_dir / "README.md"
        readme_file.write_text("""# 테스트 프로젝트

이것은 Selvage 평가 에이전트 ReAct 루프 테스트를 위한 프로젝트입니다.

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
            "project_name": "react-test-project",
            "version": "2.0.0",
            "author": "ReAct 테스트 작성자",
            "dependencies": ["pytest", "mock", "requests"],
            "description": "ReAct 루프 통합 테스트용 프로젝트",
            "settings": {
                "debug": True,
                "max_retries": 3,
                "timeout": 30
            }
        }
        config_file = temp_dir / "config.json"
        config_file.write_text(
            json.dumps(config_data, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        
        # 소스 디렉토리 구조
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        main_file = src_dir / "main.py"
        main_file.write_text("""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def main():
    print("안녕하세요, ReAct 테스트 프로젝트입니다!")
    return "success"

def process_data(data):
    \"\"\"데이터를 처리하는 함수\"\"\"
    if not data:
        return None
    return data.upper()

if __name__ == "__main__":
    main()
""", encoding="utf-8")
        
        # 테스트 디렉토리
        tests_dir = temp_dir / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_main.py"
        test_file.write_text("""import pytest
from src.main import main, process_data

def test_main():
    result = main()
    assert result == "success"

def test_process_data():
    assert process_data("hello") == "HELLO"
    assert process_data("") is None
    assert process_data(None) is None
""", encoding="utf-8")
        
        # 권한 설정
        try:
            os.chmod(str(temp_dir), stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            for file_path in [readme_file, config_file, main_file, test_file]:
                os.chmod(str(file_path), stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            for dir_path in [src_dir, tests_dir]:
                os.chmod(str(dir_path), stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        except OSError as e:
            print(f"Warning: Could not set permissions on test files: {e}")
        
        return temp_dir

    def test_react_loop_simple_query(self, agent: SelvageEvaluationAgent, sample_project_structure):
        """단순 쿼리에 대한 ReAct 루프 테스트 - plan_execution_loop 메서드 호출 확인"""
        # Given: 샘플 프로젝트 구조와 단순한 파일 존재 확인 요청
        agent.work_dir = str(sample_project_structure)
        user_query = "README.md 파일이 있나요?"
        
        # plan_execution_loop 메서드가 존재하는지 확인
        assert hasattr(agent, 'plan_execution_loop'), "plan_execution_loop 메서드가 구현되어야 합니다"
        
        # When: ReAct 루프를 통한 메시지 처리 (직접 plan_execution_loop 호출)
        response = agent.plan_execution_loop(user_query)
        
        # Then: 적절한 응답 확인
        assert isinstance(response, str)
        assert len(response) > 0
        
        # 파일 존재 여부를 명확히 답변하는지 확인
        response_lower = response.lower()
        assert "readme" in response_lower
        assert ("있습니다" in response or "존재" in response or "파일" in response)
        
        print(f"\n[단순 쿼리 ReAct 루프 테스트]")
        print(f"질문: {user_query}")
        print(f"응답: {response}")

    def test_react_loop_complex_query(self, agent: SelvageEvaluationAgent, sample_project_structure):
        """복잡한 쿼리에 대한 ReAct 루프 테스트 - 여러 단계 도구 호출 필요"""
        # Given: 샘플 프로젝트 구조와 복잡한 설정 파일 분석 요청
        agent.work_dir = str(sample_project_structure)
        user_query = "config.json에는 어떤 설정이 있나요? 주요 설정들을 설명해주세요."
        
        # When: ReAct 루프를 통한 메시지 처리
        response = agent.handle_user_message(user_query)
        
        # Then: 적절한 응답 확인
        assert isinstance(response, str)
        assert len(response) > 100  # 충분히 상세한 응답
        
        # JSON 내용이 분석되어 응답에 포함되는지 확인
        response_lower = response.lower()
        expected_content = ["project_name", "version", "dependencies", "설정", "react-test-project"]
        content_found = [content for content in expected_content if content in response_lower]
        assert len(content_found) >= 2, f"Expected config content in response. Found: {content_found}"
        
        # 대화 히스토리 저장 확인
        assert len(agent.session_state.conversation_history) == 1
        history_entry = agent.session_state.conversation_history[0]
        assert history_entry["user_message"] == user_query
        assert history_entry["assistant_response"] == response
        
        # 여러 도구가 실행되었는지 확인 (파일 존재 확인 + 파일 읽기 등)
        tool_results = history_entry.get('tool_results', [])
        assert len(tool_results) >= 1, "복잡한 쿼리는 최소 1개 이상의 도구 실행이 필요합니다"
        
        print(f"\n[복잡한 쿼리 ReAct 루프 테스트]")
        print(f"질문: {user_query}")
        print(f"응답 길이: {len(response)}자")
        print(f"도구 실행 수: {len(tool_results)}")
        print(f"응답 샘플: {response[:200]}...")

    def test_react_loop_multi_step_analysis(self, agent: SelvageEvaluationAgent, sample_project_structure):
        """다단계 분석이 필요한 ReAct 루프 테스트"""
        # Given: 샘플 프로젝트 구조와 전체 프로젝트 분석 요청
        agent.work_dir = str(sample_project_structure)
        user_query = "이 프로젝트의 전체 구조를 분석하고 주요 파일들의 내용을 요약해주세요."
        
        # When: ReAct 루프를 통한 메시지 처리
        response = agent.handle_user_message(user_query)
        
        # Then: 적절한 응답 확인
        assert isinstance(response, str)
        assert len(response) > 200  # 상세한 분석 응답
        
        # 프로젝트 구조 요소들이 응답에 포함되는지 확인
        response_lower = response.lower()
        expected_elements = ["src", "tests", "readme", "config", "프로젝트", "구조"]
        elements_found = [elem for elem in expected_elements if elem in response_lower]
        assert len(elements_found) >= 4, f"Expected project structure elements. Found: {elements_found}"
        
        # 대화 히스토리 저장 확인
        assert len(agent.session_state.conversation_history) == 1
        history_entry = agent.session_state.conversation_history[0]
        
        # 다단계 분석을 위해 여러 도구가 실행되었는지 확인
        tool_results = history_entry.get('tool_results', [])
        assert len(tool_results) >= 2, "다단계 분석은 최소 2개 이상의 도구 실행이 필요합니다"
        
        print(f"\n[다단계 분석 ReAct 루프 테스트]")
        print(f"질문: {user_query}")
        print(f"응답 길이: {len(response)}자")
        print(f"도구 실행 수: {len(tool_results)}")
        
        # 실행된 도구들 확인
        executed_tools = [result.get('tool', 'Unknown') for result in tool_results]
        print(f"실행된 도구들: {executed_tools}")

    def test_react_loop_error_handling(self, agent: SelvageEvaluationAgent, sample_project_structure):
        """존재하지 않는 파일 요청 시 ReAct 루프 오류 처리 테스트"""
        # Given: 샘플 프로젝트 구조와 존재하지 않는 파일 요청
        agent.work_dir = str(sample_project_structure)
        user_query = "존재하지_않는_파일.txt의 내용을 읽어주세요."
        
        # When: ReAct 루프를 통한 메시지 처리
        response = agent.handle_user_message(user_query)
        
        # Then: 적절한 오류 처리 응답 확인
        assert isinstance(response, str)
        assert len(response) > 0
        
        # 오류 상황을 적절히 처리했는지 확인
        response_lower = response.lower()
        error_indicators = ["존재하지 않", "찾을 수 없", "파일이 없", "오류", "실패"]
        error_found = any(indicator in response_lower for indicator in error_indicators)
        assert error_found, f"Expected error indication in response: {response}"
        
        # 오류 상황도 히스토리에 기록되었는지 확인
        assert len(agent.session_state.conversation_history) == 1
        history_entry = agent.session_state.conversation_history[0]
        assert history_entry["user_message"] == user_query
        assert history_entry["assistant_response"] == response
        
        print(f"\n[오류 처리 ReAct 루프 테스트]")
        print(f"질문: {user_query}")
        print(f"응답: {response}")

    def test_react_loop_max_iterations_handling(self, agent: SelvageEvaluationAgent, sample_project_structure):
        """최대 반복 횟수 도달 시 ReAct 루프 처리 테스트"""
        # Given: 의도적으로 복잡하고 해결하기 어려운 요청
        agent.work_dir = str(sample_project_structure)
        user_query = "존재하지_않는_파일.txt와 또_다른_없는_파일.py를 비교 분석하고 차이점을 상세히 설명해주세요."
        
        # When: ReAct 루프를 통한 메시지 처리
        response = agent.handle_user_message(user_query)
        
        # Then: 적절한 응답 또는 최대 반복 도달 처리 확인
        assert isinstance(response, str)
        assert len(response) > 0
        
        # 응답이 합리적인지 확인 (오류 처리 또는 최대 반복 도달 메시지)
        response_lower = response.lower()
        handling_indicators = [
            "최대", "반복", "처리할 수 없", "해결할 수 없", 
            "존재하지 않", "찾을 수 없", "오류", "실패"
        ]
        handling_found = any(indicator in response_lower for indicator in handling_indicators)
        assert handling_found, f"Expected proper handling indication in response: {response}"
        
        print(f"\n[최대 반복 처리 ReAct 루프 테스트]")
        print(f"질문: {user_query}")
        print(f"응답: {response}")
        
        # 히스토리 기록 확인
        assert len(agent.session_state.conversation_history) == 1

    def test_react_loop_with_conversation_context(self, agent: SelvageEvaluationAgent, sample_project_structure):
        """대화 컨텍스트를 활용한 ReAct 루프 테스트"""
        # Given: 샘플 프로젝트 구조
        agent.work_dir = str(sample_project_structure)
        
        # 첫 번째 대화: 디렉토리 목록 조회
        first_query = "프로젝트에 어떤 파일들이 있나요?"
        first_response = agent.handle_user_message(first_query)
        
        assert isinstance(first_response, str)
        assert len(first_response) > 0
        assert len(agent.session_state.conversation_history) == 1
        
        # 두 번째 대화: 이전 맥락을 활용한 후속 질문
        second_query = "그 중에서 설정 파일의 내용을 자세히 보여주세요."
        second_response = agent.handle_user_message(second_query)
        
        assert isinstance(second_response, str)
        assert len(second_response) > 0
        assert len(agent.session_state.conversation_history) == 2
        
        # 설정 파일 내용이 포함되어 있는지 확인
        response_lower = second_response.lower()
        config_indicators = ["config", "설정", "project_name", "version"]
        config_found = any(indicator in response_lower for indicator in config_indicators)
        assert config_found, f"Expected config content in contextual response: {second_response}"
        
        print(f"\n[컨텍스트 활용 ReAct 루프 테스트]")
        print(f"첫 번째 질문: {first_query}")
        print(f"두 번째 질문: {second_query}")
        print(f"두 번째 응답 길이: {len(second_response)}자")
        print(f"총 대화 히스토리: {len(agent.session_state.conversation_history)}개")