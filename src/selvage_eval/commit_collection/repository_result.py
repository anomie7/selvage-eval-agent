from dataclasses import dataclass
from typing import List, Dict, Any
from .commit_data import CommitData
from .repository_metadata import RepositoryMetadata

@dataclass
class RepositoryResult:
    """저장소별 커밋 수집 결과"""
    repo_name: str
    repo_path: str
    commits: List[CommitData]
    metadata: RepositoryMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화를 위한 딕셔너리 변환"""
        return {
            'repo_name': self.repo_name,
            'repo_path': self.repo_path,
            'commits': [commit.to_dict() for commit in self.commits],
            'metadata': self.metadata.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RepositoryResult':
        """딕셔너리에서 RepositoryResult 객체 생성"""
        return cls(
            repo_name=data['repo_name'],
            repo_path=data['repo_path'],
            commits=[CommitData.from_dict(commit_data) for commit_data in data['commits']],
            metadata=RepositoryMetadata.from_dict(data['metadata'])
        ) 