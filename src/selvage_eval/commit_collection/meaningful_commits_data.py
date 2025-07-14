from dataclasses import dataclass
from typing import List, Dict, Any
import json
from .repository_result import RepositoryResult

@dataclass
class MeaningfulCommitsData:
    """전체 커밋 수집 결과"""
    repositories: List[RepositoryResult]
    
    def save_to_json(self, filepath: str) -> None:
        """JSON 파일로 저장"""
        data = {
            'repositories': [repo.to_dict() for repo in self.repositories]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @property
    def total_commits(self) -> int:
        """전체 선별된 커밋 수"""
        return sum(len(repo.commits) for repo in self.repositories)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'MeaningfulCommitsData':
        """JSON 파일에서 MeaningfulCommitsData 객체 생성"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        repositories = [
            RepositoryResult.from_dict(repo_data) 
            for repo_data in data['repositories']
        ]
        
        return cls(repositories=repositories) 