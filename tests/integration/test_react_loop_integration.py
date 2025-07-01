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
