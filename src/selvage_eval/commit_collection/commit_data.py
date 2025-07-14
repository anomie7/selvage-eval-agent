from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from datetime import datetime
from .commit_stats import CommitStats
from .commit_score import CommitScore

@dataclass
class CommitData:
    """개별 커밋 데이터"""
    id: str
    message: str
    author: str
    date: datetime
    stats: CommitStats
    score: CommitScore
    file_paths: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화를 위한 딕셔너리 변환"""
        data = asdict(self)
        data['date'] = self.date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommitData':
        """딕셔너리에서 CommitData 객체 생성"""
        return cls(
            id=data['id'],
            message=data['message'],
            author=data['author'],
            date=datetime.fromisoformat(data['date']),
            stats=CommitStats.from_dict(data['stats']),
            score=CommitScore.from_dict(data['score']),
            file_paths=data['file_paths']
        ) 