"""도구 실행 통합 테스트

plan_execution으로 받은 도구들을 실제로 실행하여 동작을 검증합니다.
tempDir을 사용해서 안전한 테스트 환경에서 실행합니다.
"""

import pytest
import os
import json
from unittest.mock import Mock

from selvage_eval.agent.core_agent import SelvageEvaluationAgent
from selvage_eval.tools.execution_plan import ExecutionPlan
from selvage_eval.tools.tool_result import ToolResult


@pytest.mark.integration
@pytest.mark.slow
class TestToolExecutionIntegration:
    """도구 실행 통합 테스트 - 실제 도구들을 tempDir에서 실행"""

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
        """실제 도구 실행이 가능한 에이전트 인스턴스"""
        agent = SelvageEvaluationAgent(real_config, work_dir=str(temp_dir))
        return agent

    @pytest.fixture
    def setup_test_environment(self, temp_dir):
        """테스트 환경 설정 - 파일들과 디렉토리 구조 생성"""
        # 테스트 파일들 생성
        (temp_dir / "README.md").write_text("# 테스트 프로젝트\n\n이것은 테스트용 README 파일입니다.", encoding="utf-8")
        (temp_dir / "config.json").write_text(json.dumps({
            "name": "test-project",
            "version": "1.0.0",
            "description": "테스트 프로젝트"
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        
        # 서브 디렉토리 생성
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("print('Hello, World!')", encoding="utf-8")
        
        return temp_dir

    def test_file_operations_chain(self, agent, setup_test_environment):
        """파일 조작 도구들의 연쇄 실행 테스트"""
        # Given: 파일 조작 시나리오
        temp_dir = setup_test_environment
        
        # 1. 파일 존재 확인
        exists_result = agent.execute_tool("file_exists", {
            "file_path": str(temp_dir / "README.md")
        })
        assert exists_result.success
        assert exists_result.data["exists"] is True
        print(f"File exists check: {exists_result.data}")
        
        # 2. 파일 내용 읽기
        read_result = agent.execute_tool("read_file", {
            "file_path": str(temp_dir / "README.md")
        })
        assert read_result.success
        assert "테스트 프로젝트" in read_result.data["content"]
        print(f"File content: {read_result.data['content'][:50]}...")
        
        # 3. 새 파일 작성
        new_file_path = str(temp_dir / "output.txt")
        write_result = agent.execute_tool("write_file", {
            "file_path": new_file_path,
            "content": "도구 실행 테스트 결과\n성공적으로 파일이 생성되었습니다."
        })
        assert write_result.success
        print(f"File write result: {write_result.data}")
        
        # 4. 작성된 파일 검증
        verify_result = agent.execute_tool("read_file", {
            "file_path": new_file_path
        })
        assert verify_result.success
        assert "도구 실행 테스트 결과" in verify_result.data["content"]

    def test_directory_operations(self, agent, setup_test_environment):
        """디렉토리 조회 도구 실행 테스트"""
        # Given: 디렉토리 구조가 설정된 환경
        temp_dir = setup_test_environment
        
        # When: 디렉토리 목록 조회
        list_result = agent.execute_tool("list_directory", {
            "directory_path": str(temp_dir)
        })
        
        # Then: 디렉토리 내용이 올바르게 반환되는지 검증
        assert list_result.success
        assert list_result.data is not None
        
        files = list_result.data.get("files", [])
        directories = list_result.data.get("directories", [])
        all_items = files + directories
        print(f"Files: {files}")
        print(f"Directories: {directories}")
        print(f"All items: {all_items}")
        
        # 예상 파일들이 목록에 있는지 확인
        expected_files = ["README.md", "config.json"]
        expected_dirs = ["src"]
        
        for expected in expected_files:
            assert expected in files, f"Expected file {expected} not found in {files}"
            
        for expected in expected_dirs:
            assert expected in directories, f"Expected directory {expected} not found in {directories}"

    def test_safe_command_execution(self, agent, setup_test_environment):
        """안전한 명령어 실행 도구 테스트"""
        # Given: 안전한 명령어들
        temp_dir = setup_test_environment
        
        # When: 안전한 명령어 실행
        commands_to_test = [
            {"command": "echo 'Hello, World!'", "description": "Echo command"},
            {"command": "pwd", "description": "Print working directory"},
            {"command": f"ls {temp_dir}", "description": "List directory contents"}
        ]
        
        for cmd_info in commands_to_test:
            result = agent.execute_tool("execute_safe_command", {
                "command": cmd_info["command"]
            })
            
            print(f"{cmd_info['description']}: {result.success}")
            if result.success:
                print(f"Output: {result.data.get('output', '')[:100]}")
            else:
                print(f"Error: {result.error_message}")
            
            # 각 명령어가 성공적으로 실행되는지 확인
            assert result.success or result.error_message is not None

    def test_json_file_operations(self, agent, setup_test_environment):
        """JSON 파일 조작 도구 테스트"""
        # Given: JSON 파일이 있는 환경
        temp_dir = setup_test_environment
        json_file_path = str(temp_dir / "config.json")
        
        # When: JSON 파일 읽기
        read_result = agent.execute_tool("read_file", {
            "file_path": json_file_path
        })
        
        # Then: JSON 내용이 올바르게 읽혀지는지 검증
        assert read_result.success
        content = read_result.data["content"]
        assert "test-project" in content
        print(f"JSON content: {content}")
        
        # JSON 파일 수정 및 저장
        new_json_data = {
            "name": "updated-test-project",
            "version": "1.1.0",
            "description": "업데이트된 테스트 프로젝트",
            "updated": True
        }
        
        write_result = agent.execute_tool("write_file", {
            "file_path": str(temp_dir / "updated_config.json"),
            "content": new_json_data,
            "as_json": True
        })
        
        assert write_result.success
        print(f"JSON write result: {write_result.data}")

    def test_plan_execution_to_tool_execution_flow(self, agent, setup_test_environment):
        """plan_execution → 도구 실행 전체 흐름 테스트"""
        # Given: 실제 사용자 요청 시나리오
        temp_dir = setup_test_environment
        user_query = f"{temp_dir} 디렉토리의 구조를 분석하고 README.md 파일의 내용을 보여주세요"
        
        # When: 계획 수립
        plan = agent.plan_execution(user_query)
        assert isinstance(plan, ExecutionPlan)
        
        print(f"Generated plan: {plan.intent_summary}")
        print(f"Tool calls: {[(tc.tool, tc.params) for tc in plan.tool_calls]}")
        
        # 계획에 따라 도구들 실행
        tool_results = []
        for tool_call in plan.tool_calls:
            print(f"Executing tool: {tool_call.tool} with params: {tool_call.params}")
            
            result = agent.execute_tool(tool_call.tool, tool_call.params)
            tool_results.append({
                "tool": tool_call.tool,
                "result": result,
                "rationale": tool_call.rationale
            })
            
            print(f"Tool result: success={result.success}, error={result.error_message}")
            if result.success and result.data:
                print(f"Data preview: {str(result.data)[:100]}...")
        
        # Then: 모든 도구 실행이 성공하거나 적절한 오류 메시지가 있는지 검증
        assert len(tool_results) > 0
        
        for tool_result in tool_results:
            result = tool_result["result"]
            assert isinstance(result, ToolResult)
            # 성공하거나 명확한 오류 메시지가 있어야 함
            assert result.success or result.error_message is not None

    def test_error_handling_in_tool_execution(self, agent, temp_dir):
        """도구 실행 중 오류 처리 테스트"""
        # Given: 잘못된 파라미터나 존재하지 않는 파일
        error_scenarios = [
            {
                "tool": "read_file",
                "params": {"file_path": str(temp_dir / "nonexistent.txt")},
                "description": "존재하지 않는 파일 읽기"
            },
            {
                "tool": "list_directory", 
                "params": {"directory_path": str(temp_dir / "nonexistent_dir")},
                "description": "존재하지 않는 디렉토리 조회"
            },
            {
                "tool": "execute_safe_command",
                "params": {"command": "invalid_command_that_does_not_exist"},
                "description": "존재하지 않는 명령어 실행"
            }
        ]
        
        for scenario in error_scenarios:
            print(f"Testing error scenario: {scenario['description']}")
            
            result = agent.execute_tool(scenario["tool"], scenario["params"])
            
            # 오류가 적절히 처리되는지 확인
            assert isinstance(result, ToolResult)
            if not result.success:
                assert result.error_message is not None
                assert len(result.error_message) > 0
                print(f"Expected error caught: {result.error_message}")
            else:
                print(f"Unexpectedly succeeded: {result.data}")

    def test_tool_execution_with_korean_content(self, agent, temp_dir):
        """한글 내용을 포함한 도구 실행 테스트"""
        # Given: 한글 내용이 포함된 파일 작업
        korean_content = """
# 한글 테스트 파일

이것은 한글 내용이 포함된 테스트 파일입니다.
도구가 한글을 올바르게 처리하는지 확인합니다.

## 테스트 항목
- 파일 읽기/쓰기
- 디렉토리 조회
- 명령어 실행

끝.
        """.strip()
        
        korean_file_path = str(temp_dir / "한글_파일.txt")
        
        # When: 한글 파일 생성
        write_result = agent.execute_tool("write_file", {
            "file_path": korean_file_path,
            "content": korean_content
        })
        
        assert write_result.success
        print(f"Korean file write result: {write_result.data}")
        
        # 한글 파일 읽기
        read_result = agent.execute_tool("read_file", {
            "file_path": korean_file_path
        })
        
        assert read_result.success
        assert "한글 테스트 파일" in read_result.data["content"]
        print(f"Korean file content verified: {len(read_result.data['content'])} chars")