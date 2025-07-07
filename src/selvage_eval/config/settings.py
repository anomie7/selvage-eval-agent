"""설정 관리 모듈

YAML 설정 파일을 로드하고 검증하는 기능을 제공합니다.
Pydantic을 사용하여 타입 안전성과 검증을 보장합니다.
"""

import os
from typing import List, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


class FilterOverrides(BaseModel):
    """저장소별 필터 오버라이드 설정"""
    min_changed_lines: Optional[int] = None


class TargetRepository(BaseModel):
    """대상 저장소 설정"""
    name: str
    path: str
    tech_stack: str
    description: str
    access_mode: str = "readonly"
    filter_overrides: Optional[FilterOverrides] = None
    
    @field_validator('path')
    @classmethod
    def validate_path_exists(cls, v):
        if not os.path.exists(v):
            logger.warning(f"Repository path does not exist: {v}")
        return v



class CommitStats(BaseModel):
    """커밋 통계 필터"""
    min_files: int = 2
    max_files: int = 10
    min_lines: int = 50


class MergeHandling(BaseModel):
    """머지 처리 방식"""
    fast_forward: str = "exclude"
    conflict_resolution: str = "include"
    squash_merge: str = "include"
    feature_branch: str = "conditional"


class CommitFilters(BaseModel):
    """커밋 필터링 설정"""
    stats: CommitStats
    merge_handling: MergeHandling


class SkipExisting(BaseModel):
    """기존 결과 스킵 설정"""
    commit_filtering: bool = True
    review_results: bool = True


class ParallelExecution(BaseModel):
    """병렬 실행 설정"""
    max_concurrent_repos: int = 2
    max_concurrent_models: int = 3


class WorkflowConfig(BaseModel):
    """워크플로우 설정"""
    skip_existing: SkipExisting
    parallel_execution: ParallelExecution
    cache_enabled: bool = True


class EvaluationSettings(BaseModel):
    """평가 설정"""
    output_dir: str = "./selvage-eval-results"
    auto_session_id: bool = True


class SelvageConfig(BaseModel):
    """Selvage 실행 설정"""
    binary_path: str = "/Users/demin_coder/.local/bin/selvage"
    timeout_seconds: int = 300
    
    @field_validator('binary_path')
    @classmethod
    def validate_binary_exists(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Selvage binary not found at: {v}")
        return v




class EvaluationConfig(BaseModel):
    """전체 평가 설정"""
    agent_model: str = "gemini-2.5-flash"
    evaluation: EvaluationSettings
    target_repositories: List[TargetRepository]
    review_models: List[str]
    commit_filters: CommitFilters
    commits_per_repo: int = 5
    workflow: WorkflowConfig
    selvage: SelvageConfig
    
    @model_validator(mode='after')
    def validate_config(self):
        """전체 설정 검증"""
        # API 키 확인
        required_env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
        
        return self
    
    def get_repository_by_name(self, name: str) -> Optional[TargetRepository]:
        """이름으로 저장소 설정 조회"""
        for repo in self.target_repositories:
            if repo.name == name:
                return repo
        return None
    
    def get_output_path(self, *path_parts: str) -> str:
        """출력 디렉토리 기준 경로 생성"""
        return os.path.join(self.evaluation.output_dir, *path_parts)
    
    def get_review_logs_path(self, session_id: Optional[str] = None) -> str:
        """리뷰 로그 디렉토리 경로 반환"""
        base_path = os.path.expanduser("~/Library/selvage-eval/review_logs")
        if session_id:
            return os.path.join(base_path, session_id)
        return base_path
    
    def create_output_dirs(self) -> None:
        """필요한 출력 디렉토리 생성"""
        dirs_to_create = [
            self.evaluation.output_dir,
            self.get_output_path("review_logs"),
            self.get_output_path("evaluations"),
            self.get_output_path("analysis"),
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")


def load_config(config_path: str) -> EvaluationConfig:
    """설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        로드된 설정 객체
        
    Raises:
        FileNotFoundError: 설정 파일이 존재하지 않는 경우
        ValueError: 설정 파일 형식이 잘못된 경우
    """
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        config = EvaluationConfig(**config_data)
        logger.info(f"Loaded configuration from: {config_path}")
        
        # 출력 디렉토리 생성
        config.create_output_dirs()
        
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load config: {e}")


def get_default_config_path() -> str:
    """기본 설정 파일 경로 반환"""
    # 현재 디렉토리에서 설정 파일 찾기
    current_dir_config = "./configs/selvage-eval-config.yml"
    if os.path.exists(current_dir_config):
        return current_dir_config
    
    # 패키지 디렉토리에서 설정 파일 찾기
    package_dir = Path(__file__).parent.parent.parent.parent
    package_config = package_dir / "configs" / "selvage-eval-config.yml"
    if package_config.exists():
        return str(package_config)
    
    raise FileNotFoundError("Default config file not found")
