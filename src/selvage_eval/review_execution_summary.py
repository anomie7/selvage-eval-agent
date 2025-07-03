from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class ReviewExecutionSummary:
    """리뷰 실행 요약 (ToolResult.data 용)"""
    total_commits_reviewed: int
    total_reviews_executed: int
    total_successes: int
    total_failures: int
    execution_time_seconds: float
    output_directory: str
    success_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewExecutionSummary':
        return cls(**data)
    
    @property
    def summary_message(self) -> str:
        """실행 요약 메시지"""
        return (f"리뷰 완료: {self.total_commits_reviewed}개 커밋, "
                f"{self.total_reviews_executed}개 리뷰 ({self.success_rate:.1%} 성공)")