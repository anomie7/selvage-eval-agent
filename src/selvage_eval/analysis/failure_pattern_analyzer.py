"""실패 패턴 분석기

GeminiFailureAnalyzer를 사용하여 실패 패턴을 통합 분석합니다.
"""

import numpy as np
from typing import List, Dict, Any
import logging

from .deepeval_log_parser import TestCaseResult
from .gemini_failure_analyzer import GeminiFailureAnalyzer

logger = logging.getLogger(__name__)


class FailurePatternAnalyzer:
    """실패 패턴 분석기 (통합 인터페이스)"""
    
    def __init__(self):
        try:
            self.gemini_analyzer = GeminiFailureAnalyzer()
        except RuntimeError as e:
            logger.warning(f"GeminiFailureAnalyzer 초기화 실패: {e}")
            self.gemini_analyzer = None
    
    def analyze_failure_patterns(self, 
                               failed_cases: List[TestCaseResult]) -> Dict[str, Any]:
        """실패 패턴 종합 분석
        
        Args:
            failed_cases: 실패한 테스트 케이스 결과 리스트
            
        Returns:
            Dict: 실패 패턴 분석 결과
        """
        if not failed_cases:
            return {
                'total_failures': 0,
                'by_metric': {},
                'by_category': {},
                'critical_patterns': [],
                'confidence_scores': {},
                'gemini_available': self.gemini_analyzer is not None
            }
        
        patterns = {
            'total_failures': len(failed_cases),
            'by_metric': {},
            'by_category': {},
            'critical_patterns': [],
            'confidence_scores': {},
            'gemini_available': self.gemini_analyzer is not None
        }
        
        # 메트릭별 실패 분석
        for metric in ['correctness', 'clarity', 'actionability', 'json_correctness']:
            metric_failures = [
                case for case in failed_cases 
                if not getattr(case, metric).passed
            ]
            
            if not metric_failures:
                patterns['by_metric'][metric] = {
                    'total_failures': 0,
                    'failure_rate': 0.0,
                    'categories': {},
                    'worst_cases': [],
                    'avg_confidence': 0.0
                }
                continue
            
            categories = {}
            confidences = []
            
            # Gemini 분석이 가능한 경우
            if self.gemini_analyzer:
                for case in metric_failures:
                    reason = getattr(case, metric).reason
                    try:
                        category, confidence = self.gemini_analyzer.categorize_failure(reason, metric)
                        categories[category] = categories.get(category, 0) + 1
                        confidences.append(confidence)
                    except Exception as e:
                        logger.warning(f"Gemini 분류 실패: {e}")
                        # fallback 분류
                        category = self._fallback_categorize_failure(reason, metric)
                        categories[category] = categories.get(category, 0) + 1
                        confidences.append(0.5)  # 낮은 신뢰도
            else:
                # Gemini가 없는 경우 fallback 분류만 사용
                for case in metric_failures:
                    reason = getattr(case, metric).reason
                    category = self._fallback_categorize_failure(reason, metric)
                    categories[category] = categories.get(category, 0) + 1
                    confidences.append(0.5)  # 낮은 신뢰도
            
            patterns['by_metric'][metric] = {
                'total_failures': len(metric_failures),
                'failure_rate': len(metric_failures) / len(failed_cases) if failed_cases else 0,
                'categories': categories,
                'worst_cases': self._extract_worst_cases(metric_failures, metric),
                'avg_confidence': np.mean(confidences) if confidences else 0.0
            }
        
        # 전체 카테고리별 분석
        all_categories = {}
        all_confidences = []
        
        for case in failed_cases:
            for metric in ['correctness', 'clarity', 'actionability', 'json_correctness']:
                if not getattr(case, metric).passed:
                    reason = getattr(case, metric).reason
                    
                    if self.gemini_analyzer:
                        try:
                            category, confidence = self.gemini_analyzer.categorize_failure(reason, metric)
                            all_categories[category] = all_categories.get(category, 0) + 1
                            all_confidences.append(confidence)
                        except Exception as e:
                            logger.warning(f"Gemini 분류 실패: {e}")
                            category = self._fallback_categorize_failure(reason, metric)
                            all_categories[category] = all_categories.get(category, 0) + 1
                            all_confidences.append(0.5)
                    else:
                        category = self._fallback_categorize_failure(reason, metric)
                        all_categories[category] = all_categories.get(category, 0) + 1
                        all_confidences.append(0.5)
        
        patterns['by_category'] = all_categories
        patterns['confidence_scores']['overall'] = np.mean(all_confidences) if all_confidences else 0.0
        
        # 중요한 패턴 식별
        patterns['critical_patterns'] = self._identify_critical_patterns(patterns)
        
        return patterns
    
    def _fallback_categorize_failure(self, reason: str, metric: str) -> str:
        """Gemini 없이 기본 규칙 기반 분류
        
        Args:
            reason: 실패 사유
            metric: 메트릭 이름
            
        Returns:
            str: 분류된 카테고리
        """
        reason_lower = reason.lower()
        
        # 메트릭별 기본 분류 규칙
        if metric == 'correctness':
            if any(keyword in reason_lower for keyword in ['missing', 'not identified', 'failed to identify']):
                return 'missing_issues'
            elif any(keyword in reason_lower for keyword in ['incorrect', 'wrong', 'inaccurate']):
                return 'incorrect_analysis'
            elif any(keyword in reason_lower for keyword in ['severity', 'importance', 'priority']):
                return 'severity_misjudgment'
            else:
                return 'general_correctness_issue'
        
        elif metric == 'clarity':
            if any(keyword in reason_lower for keyword in ['unclear', 'confusing', 'ambiguous']):
                return 'unclear_explanation'
            elif any(keyword in reason_lower for keyword in ['technical', 'jargon', 'complex']):
                return 'technical_jargon'
            elif any(keyword in reason_lower for keyword in ['structure', 'organization', 'format']):
                return 'poor_structure'
            else:
                return 'general_clarity_issue'
        
        elif metric == 'actionability':
            if any(keyword in reason_lower for keyword in ['vague', 'general', 'abstract']):
                return 'vague_suggestions'
            elif any(keyword in reason_lower for keyword in ['specific', 'concrete', 'detailed']):
                return 'lack_of_specificity'
            elif any(keyword in reason_lower for keyword in ['implement', 'execute', 'practical']):
                return 'impractical_suggestions'
            else:
                return 'general_actionability_issue'
        
        elif metric == 'json_correctness':
            if any(keyword in reason_lower for keyword in ['format', 'structure', 'syntax']):
                return 'json_format_error'
            elif any(keyword in reason_lower for keyword in ['schema', 'field', 'property']):
                return 'schema_violation'
            elif any(keyword in reason_lower for keyword in ['missing', 'required']):
                return 'missing_fields'
            else:
                return 'general_json_issue'
        
        return 'unknown_failure'
    
    def _extract_worst_cases(self, failures: List[TestCaseResult], 
                           metric: str) -> List[Dict[str, Any]]:
        """가장 낮은 점수의 실패 케이스 추출
        
        Args:
            failures: 실패 케이스 리스트
            metric: 메트릭 이름
            
        Returns:
            List: 최악 케이스들
        """
        metric_scores = [(case, getattr(case, metric).score) for case in failures]
        worst_cases = sorted(metric_scores, key=lambda x: x[1])[:5]
        
        return [
            {
                'score': score,
                'reason': getattr(case, metric).reason,
                'input_preview': case.input_data[:200] + '...' if len(case.input_data) > 200 else case.input_data
            }
            for case, score in worst_cases
        ]
    
    def _identify_critical_patterns(self, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """중요한 패턴 식별
        
        Args:
            patterns: 분석된 패턴 데이터
            
        Returns:
            List: 중요한 패턴들
        """
        critical_patterns = []
        
        # 전체 카테고리별 빈도 분석
        by_category = patterns.get('by_category', {})
        total_failures = patterns.get('total_failures', 0)
        
        if total_failures == 0:
            return critical_patterns
        
        # 빈도가 높은 카테고리 식별 (전체 실패의 20% 이상)
        high_frequency_threshold = max(3, total_failures * 0.2)
        
        for category, count in by_category.items():
            if count >= high_frequency_threshold:
                critical_patterns.append({
                    'category': category,
                    'count': count,
                    'percentage': (count / total_failures) * 100,
                    'severity': 'high',
                    'reason': f"전체 실패의 {(count / total_failures) * 100:.1f}%를 차지하는 주요 패턴"
                })
        
        # 메트릭별 심각한 패턴 식별
        by_metric = patterns.get('by_metric', {})
        for metric, metric_data in by_metric.items():
            metric_failures = metric_data.get('total_failures', 0)
            if metric_failures >= 5:  # 최소 5개 이상의 실패가 있는 메트릭
                critical_patterns.append({
                    'category': f"{metric}_high_failure_rate",
                    'count': metric_failures,
                    'percentage': (metric_failures / total_failures) * 100,
                    'severity': 'medium',
                    'reason': f"{metric} 메트릭에서 {metric_failures}개의 실패 발생"
                })
        
        # 신뢰도가 낮은 분류 식별
        confidence_scores = patterns.get('confidence_scores', {})
        overall_confidence = confidence_scores.get('overall', 1.0)
        
        if overall_confidence < 0.7:
            critical_patterns.append({
                'category': 'low_classification_confidence',
                'count': total_failures,
                'percentage': 100.0,
                'severity': 'low',
                'reason': f"분류 신뢰도가 낮음 (평균: {overall_confidence:.2f})"
            })
        
        # 심각도 순으로 정렬
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        critical_patterns.sort(key=lambda x: (severity_order[x['severity']], -x['count']))
        
        return critical_patterns
    
    def get_failure_summary(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """실패 패턴 요약 정보 생성
        
        Args:
            patterns: 분석된 패턴 데이터
            
        Returns:
            Dict: 요약 정보
        """
        summary = {
            'total_failures': patterns.get('total_failures', 0),
            'top_categories': [],
            'most_problematic_metric': None,
            'critical_patterns_count': len(patterns.get('critical_patterns', [])),
            'gemini_available': patterns.get('gemini_available', False)
        }
        
        # 상위 카테고리 추출
        by_category = patterns.get('by_category', {})
        if by_category:
            sorted_categories = sorted(by_category.items(), key=lambda x: x[1], reverse=True)
            summary['top_categories'] = [
                {'category': category, 'count': count}
                for category, count in sorted_categories[:5]
            ]
        
        # 가장 문제가 많은 메트릭 식별
        by_metric = patterns.get('by_metric', {})
        if by_metric:
            most_problematic = max(
                by_metric.items(),
                key=lambda x: x[1].get('total_failures', 0)
            )
            summary['most_problematic_metric'] = {
                'metric': most_problematic[0],
                'failures': most_problematic[1].get('total_failures', 0),
                'failure_rate': most_problematic[1].get('failure_rate', 0.0)
            }
        
        return summary