"""실패 패턴 분석기

실패한 테스트 케이스의 개수를 집계합니다.
"""

from typing import List, Dict, Any
import logging

from .deepeval_log_parser import TestCaseResult

logger = logging.getLogger(__name__)


class FailurePatternAnalyzer:
    """실패 패턴 분석기 (단순 집계)"""
    
    def __init__(self):
        pass
    
    def analyze_failure_patterns(self, 
                               failed_cases: List[TestCaseResult]) -> Dict[str, Any]:
        """실패 패턴 단순 집계
        
        Args:
            failed_cases: 실패한 테스트 케이스 결과 리스트
            
        Returns:
            Dict: 실패 패턴 분석 결과 (total_failures만 포함)
        """
        logger.info(f"실패 패턴 분석 시작 - {len(failed_cases)}개 실패 케이스 분석")
        
        total_failures = len(failed_cases)
        
        logger.info(f"총 실패 케이스: {total_failures}개")
        
        return {
            'total_failures': total_failures
        }
    
    def get_failure_summary(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """실패 패턴 요약 정보 생성
        
        Args:
            patterns: 분석된 패턴 데이터
            
        Returns:
            Dict: 요약 정보
        """
        return {
            'total_failures': patterns.get('total_failures', 0)
        }