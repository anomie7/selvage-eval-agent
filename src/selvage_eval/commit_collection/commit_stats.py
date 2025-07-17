from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CommitStats:
    """커밋 변경 통계"""
    files_changed: int
    lines_added: int
    lines_deleted: int
    
    @property
    def total_lines_changed(self) -> int:
        """총 변경 라인 수"""
        return self.lines_added + self.lines_deleted
    
    @property
    def addition_ratio(self) -> float:
        """추가 라인 비율 (0.0 ~ 1.0)"""
        total = self.total_lines_changed
        return self.lines_added / total if total > 0 else 0.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommitStats':
        """딕셔너리에서 CommitStats 객체 생성"""
        return cls(
            files_changed=data['files_changed'],
            lines_added=data['lines_added'],
            lines_deleted=data['lines_deleted']
        ) 