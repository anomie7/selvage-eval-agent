from dataclasses import dataclass, asdict
from typing import Dict, Any
from datetime import datetime

@dataclass
class RepositoryMetadata:
    """저장소 메타데이터"""
    total_commits: int
    filtered_commits: int
    selected_commits: int
    filter_timestamp: datetime
    processing_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화를 위한 딕셔너리 변환"""
        data = asdict(self)
        data['filter_timestamp'] = self.filter_timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RepositoryMetadata':
        """딕셔너리에서 RepositoryMetadata 객체 생성"""
        return cls(
            total_commits=data['total_commits'],
            filtered_commits=data['filtered_commits'],
            selected_commits=data['selected_commits'],
            filter_timestamp=datetime.fromisoformat(data['filter_timestamp']),
            processing_time_seconds=data['processing_time_seconds']
        ) 