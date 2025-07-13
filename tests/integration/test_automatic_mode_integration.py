"""Automatic mode 통합 테스트

실제 환경에서 automatic_mode의 전체 플로우를 테스트합니다.
"""

import tempfile
import shutil
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from selvage_eval.agent.core_agent import SelvageEvaluationAgent
from selvage_eval.config.settings import EvaluationConfig


class TestAutomaticModeIntegration:
    """Automatic mode 통합 테스트 클래스"""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 설정"""
        # 임시 디렉토리 생성
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_repo_dir = self.temp_dir / "test_repo"
        self.output_dir = self.temp_dir / "output"
        
        # 테스트용 Git 저장소 생성
        self._create_test_git_repo()
        
        # 테스트용 설정 생성
        self.config = self._create_test_config()
        
    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_test_git_repo(self):
        """테스트용 Git 저장소 생성"""
        self.test_repo_dir.mkdir(parents=True)
        
        # .git 디렉토리 생성 (실제 git init은 하지 않고 구조만 생성)
        git_dir = self.test_repo_dir / ".git"
        git_dir.mkdir()
        
        # 테스트용 파일 생성
        (self.test_repo_dir / "src").mkdir()
        (self.test_repo_dir / "src" / "main.py").write_text("""
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")
        
        (self.test_repo_dir / "README.md").write_text("# Test Repository")
    
    def _create_test_config(self) -> EvaluationConfig:
        """테스트용 설정 생성"""
        from selvage_eval.config.settings import load_config
        import yaml
        
        # 테스트용 YAML 설정 생성
        config_data = {
            "agent_model": "gemini-2.5-flash",
            "evaluation": {
                "session_prefix": "test",
                "output_dir": str(self.output_dir)
            },
            "target_repositories": [
                {
                    "name": "test_repo",
                    "path": str(self.test_repo_dir),
                    "tech_stack": "python",
                    "description": "Test repository",
                    "access_mode": "readonly"
                }
            ],
            "review_models": ["gemini-2.5-flash-preview-05-20"],
            "commits_per_repo": 3,
            "commit_filters": {
                "stats": {
                    "min_files": 1,
                    "max_files": 10,
                    "min_lines": 10
                },
                "merge_handling": {
                    "fast_forward": "exclude",
                    "conflict_resolution": "include"
                }
            },
            "workflow": {
                "skip_existing": {
                    "commit_filtering": True,
                    "review_results": True
                },
                "parallel_execution": {
                    "max_concurrent_repos": 2,
                    "max_concurrent_models": 3
                }
            },
            "selvage": {
                "binary_path": "/Users/demin_coder/.local/bin/selvage",
                "version": "0.1.2",
                "timeout": 300
            },
            "deepeval": {
                "metrics": []
            },
            "security": {
                "allowed_paths": ["/tmp", str(self.temp_dir)],
                "forbidden_commands": ["rm", "sudo"]
            },
            "resource_limits": {
                "max_memory_mb": 2048,
                "max_cpu_percent": 80,
                "max_disk_gb": 10,
                "max_execution_time": 3600
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        # 임시 설정 파일 생성
        config_file = self.temp_dir / "test_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        return load_config(str(config_file))
    
    def _create_test_commits_data(self):
        """테스트용 커밋 데이터 생성"""
        commits_data = {
            "repositories": [
                {
                    "name": "test_repo",
                    "path": str(self.test_repo_dir),
                    "commits": [
                        {
                            "id": "abc123",
                            "message": "feat: add main function",
                            "author": "Test Author <test@example.com>",
                            "date": "2024-01-01T12:00:00",
                            "stats": {
                                "files_changed": 1,
                                "lines_added": 5,
                                "lines_deleted": 0
                            },
                            "score": {
                                "base_score": 10,
                                "file_type_penalty": 0,
                                "scale_appropriateness_score": 5,
                                "commit_characteristics_score": 8,
                                "time_weight_score": 2,
                                "additional_adjustments": 0,
                                "total_score": 25
                            },
                            "file_paths": ["src/main.py"]
                        }
                    ],
                    "collection_metadata": {
                        "total_commits_found": 10,
                        "commits_after_filtering": 5,
                        "commits_selected": 1,
                        "collection_time": "2024-01-01T12:00:00"
                    }
                }
            ]
        }
        
        # meaningful_commits.json 파일 생성
        commits_file = self.output_dir / "meaningful_commits.json"
        commits_file.parent.mkdir(parents=True, exist_ok=True)
        commits_file.write_text(json.dumps(commits_data, indent=2))
        
        return commits_file
    
    @patch.dict(os.environ, {
        'GEMINI_API_KEY': 'test_key',
        'OPENAI_API_KEY': 'test_key',
        'ANTHROPIC_API_KEY': 'test_key'
    })
    @patch('selvage_eval.agent.core_agent.Path')
    def test_precondition_validation_success(self, mock_path):
        """사전 조건 검증 성공 테스트"""
        # Path.exists() 모킹
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        # Selvage 바이너리 존재 모킹
        with patch('os.path.exists') as mock_exists, \
             patch('os.access') as mock_access:
            mock_exists.return_value = True
            mock_access.return_value = True
            
            agent = SelvageEvaluationAgent(self.config)
            
            result = agent._validate_preconditions()
            
            assert result["valid"]
            assert len(result["errors"]) == 0
    
    def test_precondition_validation_missing_repo(self):
        """저장소 경로 없음 검증 테스트"""
        # 존재하지 않는 저장소 경로로 설정 수정
        # 기존 설정을 복사하고 저장소 경로만 수정
        import copy
        invalid_config = copy.deepcopy(self.config)
        invalid_config.target_repositories[0].path = "/nonexistent/path"
        
        agent = SelvageEvaluationAgent(invalid_config)
        
        result = agent._validate_preconditions()
        
        assert not result["valid"]
        assert any("does not exist" in error for error in result["errors"])
    
    def test_precondition_validation_missing_api_keys(self):
        """API 키 없음 검증 테스트"""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'temp_key'}), \
             patch('os.path.exists') as mock_exists, \
             patch('os.path.isdir') as mock_isdir, \
             patch('selvage_eval.agent.core_agent.Path') as mock_path:
            
            # 기본 파일 시스템 모킹
            mock_exists.return_value = True
            mock_isdir.return_value = True
            
            # Path 모킹
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            agent = SelvageEvaluationAgent(self.config)
            
            # 에이전트 생성 후 API 키 제거하고 검증
            with patch.dict(os.environ, {}, clear=True):
                result = agent._validate_preconditions()
            
            assert not result["valid"]
            assert any("Missing API keys" in error for error in result["errors"])
    
    @patch('selvage_eval.agent.core_agent.CommitCollector')
    def test_phase1_execution_success(self, mock_collector_class):
        """Phase 1 실행 성공 테스트"""
        # CommitCollector 모킹
        mock_collector = MagicMock()
        mock_commits_data = MagicMock()
        mock_commits_data.total_commits = 5
        mock_collector.collect_commits.return_value = mock_commits_data
        mock_collector_class.return_value = mock_collector
        
        agent = SelvageEvaluationAgent(self.config)
        
        result = agent._execute_phase1_commit_collection()
        
        assert result["status"] == "completed"
        assert result["phase"] == "commit_collection"
        assert result["commits_processed"] == 5
        assert "execution_time_seconds" in result
        assert "output_file" in result
        assert "timestamp" in result
    
    @patch('selvage_eval.agent.core_agent.CommitCollector')
    def test_phase1_execution_failure(self, mock_collector_class):
        """Phase 1 실행 실패 테스트"""
        # CommitCollector에서 예외 발생 모킹
        mock_collector_class.side_effect = Exception("Test error")
        
        agent = SelvageEvaluationAgent(self.config)
        
        result = agent._execute_phase1_commit_collection()
        
        assert result["status"] == "failed"
        assert result["phase"] == "commit_collection"
        assert "error" in result
        assert result["error"] == "Test error"
        assert "execution_time_seconds" in result
    
    @patch('selvage_eval.agent.core_agent.ReviewExecutorTool')
    def test_phase2_execution_success(self, mock_executor_class):
        """Phase 2 실행 성공 테스트"""
        # 테스트용 커밋 데이터 생성
        self._create_test_commits_data()
        
        # ReviewExecutorTool 모킹
        mock_executor = MagicMock()
        mock_executor.execute.return_value = {"review_result": "success"}
        mock_executor_class.return_value = mock_executor
        
        agent = SelvageEvaluationAgent(self.config)
        
        result = agent._execute_phase2_review_execution({})
        
        assert result["status"] == "completed"
        assert result["phase"] == "review_execution"
        assert result["models_processed"] == 1
        assert result["successful_models"] == 1
        assert result["failed_models"] == 0
        assert "execution_time_seconds" in result
        assert "output_directory" in result
    
    def test_phase2_execution_missing_commits_file(self):
        """Phase 2 실행 - 커밋 파일 없음 테스트"""
        agent = SelvageEvaluationAgent(self.config)
        
        result = agent._execute_phase2_review_execution({})
        
        assert result["status"] == "failed"
        assert result["phase"] == "review_execution"
        assert "error" in result
        assert "Required file not found" in result["error"]
    
    @patch('selvage_eval.agent.core_agent.ReviewExecutorTool')
    def test_phase2_execution_partial_failure(self, mock_executor_class):
        """Phase 2 실행 - 일부 모델 실패 테스트"""
        # 테스트용 커밋 데이터 생성
        self._create_test_commits_data()
        
        # 다중 모델 설정
        import copy
        multi_model_config = copy.deepcopy(self.config)
        multi_model_config.review_models = ["model1", "model2"]
        
        # ReviewExecutorTool 모킹 - 첫 번째 성공, 두 번째 실패
        mock_executor = MagicMock()
        mock_executor.execute.side_effect = [
            {"review_result": "success"},
            Exception("Model 2 failed")
        ]
        mock_executor_class.return_value = mock_executor
        
        agent = SelvageEvaluationAgent(multi_model_config)
        
        result = agent._execute_phase2_review_execution({})
        
        assert result["status"] == "completed"  # 일부 성공이면 completed
        assert result["phase"] == "review_execution"
        assert result["models_processed"] == 2
        assert result["successful_models"] == 1
        assert result["failed_models"] == 1
        assert len(result["review_results"]) == 2
    
    @patch.dict(os.environ, {
        'GEMINI_API_KEY': 'test_key'
    })
    @patch('selvage_eval.agent.core_agent.Path')
    @patch('selvage_eval.agent.core_agent.CommitCollector')
    @patch('selvage_eval.agent.core_agent.ReviewExecutorTool')
    def test_full_automatic_mode_flow(self, mock_executor_class, mock_collector_class, mock_path):
        """전체 automatic mode 플로우 테스트"""
        # 모든 모킹 설정
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        # CommitCollector 모킹
        mock_collector = MagicMock()
        mock_commits_data = MagicMock()
        mock_commits_data.total_commits = 3
        mock_collector.collect_commits.return_value = mock_commits_data
        mock_collector_class.return_value = mock_collector
        
        # ReviewExecutorTool 모킹
        mock_executor = MagicMock()
        mock_executor.execute.return_value = {"review_result": "success"}
        mock_executor_class.return_value = mock_executor
        
        # 파일 시스템 모킹
        with patch('os.path.exists') as mock_exists, \
             patch('os.path.isdir') as mock_isdir, \
             patch('os.access') as mock_access:
            
            mock_exists.return_value = True
            mock_isdir.return_value = True
            mock_access.return_value = True
            
            agent = SelvageEvaluationAgent(self.config)
            
            # execute_evaluation이 실제로는 복잡한 상태 기계이므로
            # 여기서는 각 phase 메서드만 개별적으로 테스트
            
            # Phase 1 테스트
            phase1_result = agent._execute_phase1_commit_collection()
            assert phase1_result["status"] == "completed"
            
            # meaningful_commits.json 파일 생성 (Phase 2를 위해)
            self._create_test_commits_data()
            
            # Phase 2 테스트
            phase2_result = agent._execute_phase2_review_execution({})
            assert phase2_result["status"] == "completed"