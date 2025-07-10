"""
메트릭 가중치 상수 정의 모듈

이 모듈은 평가 메트릭의 가중치를 중앙에서 관리합니다.
"""

from typing import Dict


# 메트릭 가중치 상수
METRIC_WEIGHTS: Dict[str, float] = {
    'correctness': 0.3,
    'clarity': 0.3,
    'actionability': 0.3,
    'json_correctness': 0.1
}