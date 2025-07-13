"""메트릭 점수 집계기

테스트 케이스 결과들을 집계하여 모델별 성능 통계를 계산합니다.
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

from .deepeval_log_parser import TestCaseResult


class MetricAggregator:
    """메트릭 점수 집계기"""
    
    def aggregate_model_performance(self, 
                                  test_results: List[TestCaseResult]) -> Dict[str, Any]:
        """모델별 종합 성능 계산
        
        Args:
            test_results: 테스트 케이스 결과 리스트
            
        Returns:
            Dict: 집계된 성능 데이터
        """
        if not test_results:
            return self._empty_performance_data()
        
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        aggregated: Dict[str, Any] = {}
        
        for metric in metrics:
            scores = [getattr(result, metric).score for result in test_results]
            passed_count = sum(1 for result in test_results 
                             if getattr(result, metric).passed)
            
            aggregated[metric] = {
                'mean_score': float(np.mean(scores)),
                'median_score': float(np.median(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'pass_rate': passed_count / len(test_results),
                'total_cases': len(test_results),
                'passed_cases': passed_count,
                'failed_cases': len(test_results) - passed_count,
                'failure_count': len(test_results) - passed_count  # 호환성을 위해 추가
            }
        
        # 종합 점수 계산 (가중평균)
        from ..constants import METRIC_WEIGHTS
        weights = METRIC_WEIGHTS
        
        overall_score = sum(
            aggregated[metric]['mean_score'] * weight 
            for metric, weight in weights.items()
        )
        
        # 전체 합격률 계산
        all_passed = sum(1 for result in test_results 
                        if all(getattr(result, metric).passed for metric in metrics))
        overall_pass_rate = all_passed / len(test_results)
        
        aggregated['overall'] = {
            'weighted_score': float(overall_score),
            'grade': self._assign_grade(overall_score),
            'consistency': float(1.0 - np.mean([aggregated[m]['std_score'] for m in metrics])),
            'total_cases': len(test_results),
            'pass_rate': overall_pass_rate,
            'total_failures': len(test_results) - all_passed
        }
        
        return aggregated
    
    def _empty_performance_data(self) -> Dict[str, Any]:
        """빈 성능 데이터 반환"""
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        aggregated: Dict[str, Any] = {}
        
        for metric in metrics:
            aggregated[metric] = {
                'mean_score': 0.0,
                'median_score': 0.0,
                'std_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'pass_rate': 0.0,
                'total_cases': 0,
                'passed_cases': 0,
                'failed_cases': 0,
                'failure_count': 0
            }
        
        aggregated['overall'] = {
            'weighted_score': 0.0,
            'grade': 'F',
            'consistency': 0.0,
            'total_cases': 0,
            'pass_rate': 0.0,
            'total_failures': 0
        }
        
        return aggregated
    
    def _assign_grade(self, score: float) -> str:
        """점수 기반 등급 할당
        
        Args:
            score: 종합 점수 (0.0~1.0)
            
        Returns:
            str: 등급 (A+, A, B+, B, C+, C, D, F)
        """
        if score >= 0.9:
            return 'A+'
        elif score >= 0.85:
            return 'A'
        elif score >= 0.8:
            return 'B+'
        elif score >= 0.75:
            return 'B'
        elif score >= 0.7:
            return 'C+'
        elif score >= 0.65:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def calculate_metric_statistics(self, test_results: List[TestCaseResult]) -> Dict[str, Dict[str, float]]:
        """메트릭별 상세 통계 계산
        
        Args:
            test_results: 테스트 케이스 결과 리스트
            
        Returns:
            Dict: 메트릭별 통계 데이터
        """
        if not test_results:
            return {}
        
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        statistics = {}
        
        for metric in metrics:
            scores = [getattr(result, metric).score for result in test_results]
            
            statistics[metric] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores)),
                'q1': float(np.percentile(scores, 25)),
                'q3': float(np.percentile(scores, 75)),
                'iqr': float(np.percentile(scores, 75) - np.percentile(scores, 25)),
                'coefficient_of_variation': float(np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0.0
            }
        
        return statistics
    
    def identify_performance_outliers(self, test_results: List[TestCaseResult]) -> Dict[str, List[Dict[str, Any]]]:
        """성능 이상치 식별
        
        Args:
            test_results: 테스트 케이스 결과 리스트
            
        Returns:
            Dict: 메트릭별 이상치 정보
        """
        if not test_results:
            return {}
        
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        outliers: Dict[str, List[Dict[str, Any]]] = {}
        
        for metric in metrics:
            scores = [getattr(result, metric).score for result in test_results]
            
            # 최소 3개 이상의 데이터가 있어야 이상치 탐지 가능
            if len(scores) < 3:
                outliers[metric] = []
                continue
            
            # IQR 방법을 사용한 이상치 탐지
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            iqr = q3 - q1
            
            # IQR이 0인 경우 (모든 값이 동일한 경우) 표준편차 방법 사용
            if iqr == 0:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                if std_score == 0:
                    outliers[metric] = []
                    continue
                lower_bound = mean_score - 2 * std_score
                upper_bound = mean_score + 2 * std_score
            else:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
            
            metric_outliers = []
            for i, result in enumerate(test_results):
                score = getattr(result, metric).score
                if score < lower_bound or score > upper_bound:
                    metric_outliers.append({
                        'test_case_index': i,
                        'score': score,
                        'reason': getattr(result, metric).reason,
                        'type': 'low' if score < lower_bound else 'high'
                    })
            
            outliers[metric] = metric_outliers
        
        return outliers
    
    def calculate_consistency_metrics(self, test_results: List[TestCaseResult]) -> Dict[str, Any]:
        """일관성 메트릭 계산
        
        Args:
            test_results: 테스트 케이스 결과 리스트
            
        Returns:
            Dict: 일관성 관련 메트릭
        """
        if not test_results:
            return {}
        
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        
        # 메트릭별 변동계수 계산
        consistency_scores = []
        for metric in metrics:
            scores = [getattr(result, metric).score for result in test_results]
            if np.mean(scores) > 0:
                cv = np.std(scores) / np.mean(scores)
                consistency_scores.append(1.0 - cv)  # 변동계수가 낮을수록 일관성이 높음
            else:
                consistency_scores.append(0.0)
        
        # 전체 일관성 점수
        overall_consistency = np.mean(consistency_scores)
        
        # 메트릭 간 균형 점수
        mean_scores = [np.mean([getattr(result, metric).score for result in test_results]) 
                      for metric in metrics]
        balance_score = 1.0 - (np.std(mean_scores) / np.mean(mean_scores)) if np.mean(mean_scores) > 0 else 0.0
        
        return {
            'overall_consistency': float(overall_consistency),
            'metric_balance': float(balance_score),
            'consistency_by_metric': {
                metric: float(score) for metric, score in zip(metrics, consistency_scores)
            }
        }