from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CommitScore:
    """커밋 점수 상세 정보"""
    total_score: int
    file_type_penalty: int
    scale_appropriateness_score: int
    commit_characteristics_score: int
    time_weight_score: int
    additional_adjustments: int
    
    def __post_init__(self):
        """점수 범위 검증"""
        self.total_score = max(0, min(100, self.total_score))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommitScore':
        """딕셔너리에서 CommitScore 객체 생성"""
        return cls(
            total_score=data['total_score'],
            file_type_penalty=data['file_type_penalty'],
            scale_appropriateness_score=data['scale_appropriateness_score'],
            commit_characteristics_score=data['commit_characteristics_score'],
            time_weight_score=data['time_weight_score'],
            additional_adjustments=data['additional_adjustments']
        ) 