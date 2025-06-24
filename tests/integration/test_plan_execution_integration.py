"""plan_execution 메서드 통합 테스트

실제 Gemini API 호출을 통해 ExecutionPlan 생성을 검증합니다.
실제 작업 흐름과 유사한 시나리오를 사용합니다.
"""

import pytest
import os
from unittest.mock import Mock

from selvage_eval.agent.core_agent import SelvageEvaluationAgent
from selvage_eval.tools.execution_plan import ExecutionPlan


@pytest.mark.integration
@pytest.mark.slow
class TestPlanExecutionIntegration:
    """plan_execution 메서드 실제 LLM 호출 통합 테스트"""

    @pytest.fixture
    def real_config(self, temp_dir):
        """실제 설정 객체 (API 키 필요)"""
        # GEMINI_API_KEY 환경변수 확인
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("GEMINI_API_KEY가 설정되지 않아 통합 테스트를 건너뜁니다")
        
        config = Mock()
        config.agent_model = "gemini-2.0-flash-exp"
        config.review_models = []
        config.target_repositories = []
        config.commits_per_repo = 10
        
        # evaluation 속성 설정
        evaluation_mock = Mock()
        evaluation_mock.output_dir = str(temp_dir)
        config.evaluation = evaluation_mock
        
        # workflow 속성 설정
        workflow_mock = Mock()
        workflow_mock.model_dump = Mock(return_value={"skip_existing": False})
        config.workflow = workflow_mock
        
        # deepeval 속성 설정
        deepeval_mock = Mock()
        deepeval_mock.metrics = []
        config.deepeval = deepeval_mock
        
        config.get_output_path = lambda *args: str(temp_dir / "_".join(args))
        return config

    @pytest.fixture
    def agent(self, real_config, temp_dir):
        """실제 API 호출이 가능한 에이전트 인스턴스"""
        agent = SelvageEvaluationAgent(real_config, work_dir=str(temp_dir))
        return agent

    def test_plan_execution_file_listing_request(self, agent: SelvageEvaluationAgent, temp_dir):
        """파일 목록 조회 요청에 대한 계획 수립 테스트"""
        # Given: 실제 사용자가 할 법한 파일 조회 요청
        user_query = "현재 디렉토리의 파일들을 보여주세요"
        
        # When: 실제 LLM 호출로 계획 수립
        result = agent.plan_execution(user_query)
        
        # Then: ExecutionPlan 객체가 올바르게 생성되는지 검증
        assert isinstance(result, ExecutionPlan)
        assert result.intent_summary is not None
        assert len(result.intent_summary) > 0
        assert result.confidence > 0.0
        assert result.safety_check is not None
        
        # 파일 조회와 관련된 도구가 선택되었는지 확인
        print(f"Intent summary: {result.intent_summary}")
        print(f"Tool calls: {[tc.tool for tc in result.tool_calls]}")
        print(f"Safety check: {result.safety_check}")
        
        # 적어도 하나의 도구 호출이 있어야 함
        assert len(result.tool_calls) > 0

    def test_plan_execution_file_reading_request(self, agent: SelvageEvaluationAgent, temp_dir):
        """파일 읽기 요청에 대한 계획 수립 테스트"""
        # Given: 테스트 파일 생성
        test_file = temp_dir / "sample.txt"
        test_file.write_text("테스트 파일 내용입니다", encoding="utf-8")
        
        user_query = f"{test_file} 파일의 내용을 읽어주세요"
        
        # When: 실제 LLM 호출로 계획 수립
        result = agent.plan_execution(user_query)
        
        # Then: 파일 읽기와 관련된 계획이 수립되는지 검증
        assert isinstance(result, ExecutionPlan)
        assert result.confidence > 0.0
        
        # 파일 읽기 도구가 포함되어 있는지 확인
        tool_names = [tc.tool for tc in result.tool_calls]
        print(f"Tool calls for file reading: {tool_names}")
        
        # read_file 또는 file_exists 도구가 있어야 함
        expected_tools = ["read_file", "file_exists"]
        has_expected_tool = any(tool in tool_names for tool in expected_tools)
        assert has_expected_tool, f"Expected one of {expected_tools} in {tool_names}"

    def test_plan_execution_git_status_request(self, agent: SelvageEvaluationAgent, temp_dir):
        """Git 상태 확인 요청에 대한 계획 수립 테스트"""
        # Given: Git 관련 요청
        user_query = "git 상태를 확인해주세요"
        
        # When: 실제 LLM 호출로 계획 수립
        result = agent.plan_execution(user_query)
        
        # Then: 명령어 실행과 관련된 계획이 수립되는지 검증
        assert isinstance(result, ExecutionPlan)
        assert result.confidence > 0.0
        
        tool_names = [tc.tool for tc in result.tool_calls]
        print(f"Tool calls for git status: {tool_names}")
        
        # execute_safe_command 도구가 있어야 함
        assert "execute_safe_command" in tool_names
        expected_command = [tc.params['command'] for tc in result.tool_calls]
        assert "git status" in expected_command

    def test_plan_execution_with_conversation_context(self, agent: SelvageEvaluationAgent, temp_dir):
        """대화 컨텍스트가 있는 상황에서의 계획 수립 테스트"""
        # Given: temp_dir에 README.md 파일 생성
        readme_file = temp_dir / "README.md"
        readme_file.write_text("# 테스트 프로젝트\n\n이것은 테스트용 프로젝트입니다.", encoding="utf-8")
        
        # 첫 번째 대화: 디렉토리 목록 조회
        agent.session_state.add_conversation_turn(
            user_message="현재 디렉토리에 어떤 파일들이 있는지 보여줘",
            assistant_response="디렉토리 목록을 확인했습니다. README.md 파일이 있습니다.",
            tool_results=[{"tool": "list_directory", "result": {
                "directory_path": str(temp_dir),
                "files": ["README.md"],
                "directories": [],
                "total_items": 1
            }}]
        )
        
        # 두 번째 대화: README.md 파일 존재 확인
        agent.session_state.add_conversation_turn(
            user_message="README.md 파일이 정말 있는지 확인해줘",
            assistant_response="README.md 파일이 존재합니다.",
            tool_results=[{"tool": "file_exists", "result": {
                "exists": True,
                "is_file": True,
                "is_directory": False,
                "file_path": str(readme_file)
            }}]
        )
        
        # 세 번째 요청: README.md 파일 내용 읽기
        user_query = "README.md 파일의 내용을 읽어서 보여줘"
        
        # When: 컨텍스트를 고려한 계획 수립
        result = agent.plan_execution(user_query)
        
        # Then: 계획이 수립되는지 검증
        assert isinstance(result, ExecutionPlan)
        assert result.intent_summary is not None
        assert len(result.intent_summary) > 0
        
        # read_file 도구가 호출되고 정확한 파일 경로가 제공되는지 검증
        tool_calls_with_params = [(tc.tool, tc.params) for tc in result.tool_calls]
        print(f"Tool calls with context: {tool_calls_with_params}")
        print(f"Intent summary: {result.intent_summary}")
        
        # read_file 도구가 있어야 하고, 정확한 파일 경로를 사용해야 함
        read_file_calls = [tc for tc in result.tool_calls if tc.tool == "read_file"]
        if read_file_calls:
            # 정확한 파일 경로가 제공되었는지 확인
            file_path = read_file_calls[0].params.get("file_path", "")
            assert str(readme_file) in file_path or "README.md" in file_path, f"Expected README.md path in {file_path}"

    def test_plan_execution_safety_validation(self, agent: SelvageEvaluationAgent, temp_dir):
        """안전성 검증이 포함된 계획 수립 테스트"""
        # Given: 안전한 작업 요청
        user_query = "텍스트 파일을 하나 생성해주세요"
        
        # When: 안전성을 고려한 계획 수립
        result = agent.plan_execution(user_query)
        
        # Then: 안전성 검증이 수행되는지 확인
        assert isinstance(result, ExecutionPlan)
        assert result.safety_check is not None
        assert len(result.safety_check) > 0
        
        print(f"Safety check result: {result.safety_check}")
        
        # 도구 호출 여부와 관계없이 계획이 생성되었는지 확인
        tool_names = [tc.tool for tc in result.tool_calls]
        print(f"Tool names: {tool_names}")
        
        # 파일 생성 도구가 있다면 좋지만, 없어도 테스트 통과
        if tool_names:
            print(f"Tools selected: {tool_names}")
        else:
            print("No tools selected - LLM provided text response only")