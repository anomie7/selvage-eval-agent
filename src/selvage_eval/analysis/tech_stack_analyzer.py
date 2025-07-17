"""기술스택별 모델 성능 분석기

저장소별로 다른 기술스택에서의 모델 성능을 분석하고 비교합니다.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

from .deepeval_log_parser import TestCaseResult
from .metric_aggregator import MetricAggregator
from ..config.settings import get_tech_stack_mapping

logger = logging.getLogger(__name__)


class TechStackAnalyzer:
    """기술스택별 모델 성능 분석기"""
    
    def __init__(self, config_path: Optional[str] = None):
        # 설정 파일에서 기술스택 매핑 로드
        self.tech_stack_mapping = get_tech_stack_mapping(config_path)
        self.aggregator = MetricAggregator()
    
    def analyze_tech_stack_performance(self, 
                                     repo_results: Dict[str, Dict[str, List[TestCaseResult]]]) -> Dict[str, Any]:
        """저장소/기술스택별 모델 성능 분석
        
        Args:
            repo_results: 저장소별 모델 결과 딕셔너리
                         {repository: {model_name: [TestCaseResult, ...]}}
            
        Returns:
            Dict: 기술스택별 분석 결과
        """
        logger.info(f"기술스택 성능 분석 시작 - {len(repo_results)}개 저장소 분석")
        
        if not repo_results:
            logger.info("분석할 저장소 데이터가 없음")
            return {
                'by_tech_stack': {},
                'cross_stack_comparison': {},
                'recommendations': []
            }
        
        # 기술스택별 성능 분석
        tech_stack_analysis = {}
        
        for repo_idx, (repository, model_results) in enumerate(repo_results.items(), 1):
            logger.info(f"저장소 {repo_idx}/{len(repo_results)} '{repository}' 분석 중...")
            # 저장소명을 기술스택으로 매핑
            tech_stack = self.tech_stack_mapping.get(repository, repository)
            logger.debug(f"저장소 '{repository}' -> 기술스택 '{tech_stack}' 매핑")
            
            if not model_results:
                logger.warning(f"저장소 '{repository}'에 모델 결과가 없음")
                continue
            
            # 모델별 성능 계산
            logger.debug(f"기술스택 '{tech_stack}' - {len(model_results)}개 모델 성능 계산 중...")
            model_performance = {}
            for model_name, test_results in model_results.items():
                logger.debug(f"모델 '{model_name}' 성능 계산 중 ({len(test_results)}개 테스트 케이스)...")
                if test_results:
                    model_performance[model_name] = self.aggregator.aggregate_model_performance(test_results)
                else:
                    model_performance[model_name] = self.aggregator.aggregate_model_performance([])
            
            # 최고 성능 모델 식별
            logger.debug(f"기술스택 '{tech_stack}' 최고 성능 모델 식별 중...")
            best_model = self._find_best_model(model_performance)
            
            # 성능 격차 계산
            logger.debug(f"기술스택 '{tech_stack}' 성능 격차 계산 중...")
            performance_gap = self._calculate_performance_gap(model_performance)
            
            # 기술스택별 권장사항 생성
            logger.debug(f"기술스택 '{tech_stack}' 권장사항 생성 중...")
            recommendations = self._generate_tech_stack_recommendations(tech_stack, {
                'model_performance': model_performance,
                'best_model': best_model,
                'performance_gap': performance_gap
            })
            
            tech_stack_analysis[tech_stack] = {
                'repository': repository,
                'model_performance': model_performance,
                'best_model': best_model,
                'performance_gap': performance_gap,
                'recommendations': recommendations,
                'model_count': len(model_performance)
            }
            
            if best_model and best_model['name']:
                logger.info(f"저장소 '{repository}' ({tech_stack}) 분석 완료 - 최고 모델: {best_model['name']} (점수: {best_model['score']:.3f})")
            else:
                logger.info(f"저장소 '{repository}' ({tech_stack}) 분석 완료 - 유효한 모델 없음")
        
        # 기술스택 간 교차 비교
        logger.info("기술스택 간 교차 비교 분석 중...")
        cross_stack_comparison = self._cross_stack_comparison(tech_stack_analysis)
        
        # 전체 권장사항 생성
        logger.info("전체 권장사항 생성 중...")
        overall_recommendations = self._generate_overall_recommendations(tech_stack_analysis, cross_stack_comparison)
        
        logger.info(f"기술스택 성능 분석 완료 - {len(tech_stack_analysis)}개 기술스택 분석됨")
        
        return {
            'by_tech_stack': tech_stack_analysis,
            'cross_stack_comparison': cross_stack_comparison,
            'recommendations': overall_recommendations
        }
    
    def _find_best_model(self, model_performance: Dict[str, Dict]) -> Dict[str, Any]:
        """최고 성능 모델 찾기
        
        Args:
            model_performance: 모델별 성능 데이터
            
        Returns:
            Dict: 최고 성능 모델 정보
        """
        if not model_performance:
            return {
                'name': None,
                'score': 0.0,
                'grade': 'F',
                'metrics': {}
            }
        
        best_model = None
        best_score = -1
        
        for model_name, performance in model_performance.items():
            if 'overall' in performance:
                score = performance['overall']['weighted_score']
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        if best_model:
            best_performance = model_performance[best_model]
            return {
                'name': best_model,
                'score': best_score,
                'grade': best_performance['overall']['grade'],
                'metrics': {
                    metric: best_performance[metric]['mean_score']
                    for metric in ['correctness', 'clarity', 'actionability', 'json_correctness']
                    if metric in best_performance
                }
            }
        
        return {
            'name': None,
            'score': 0.0,
            'grade': 'F',
            'metrics': {}
        }
    
    def _calculate_performance_gap(self, model_performance: Dict[str, Dict]) -> Dict[str, float]:
        """성능 격차 계산
        
        Args:
            model_performance: 모델별 성능 데이터
            
        Returns:
            Dict: 성능 격차 정보
        """
        if not model_performance:
            return {
                'max_score': 0.0,
                'min_score': 0.0,
                'gap': 0.0,
                'coefficient_of_variation': 0.0
            }
        
        # 전체 점수 수집
        scores = []
        for model_name, performance in model_performance.items():
            if 'overall' in performance:
                scores.append(performance['overall']['weighted_score'])
        
        if not scores:
            return {
                'max_score': 0.0,
                'min_score': 0.0,
                'gap': 0.0,
                'coefficient_of_variation': 0.0
            }
        
        max_score = max(scores)
        min_score = min(scores)
        gap = max_score - min_score
        
        # 변동계수 계산
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        coefficient_of_variation = std_score / mean_score if mean_score > 0 else 0.0
        
        return {
            'max_score': float(max_score),
            'min_score': float(min_score),
            'gap': float(gap),
            'coefficient_of_variation': float(coefficient_of_variation)
        }
    
    def _generate_tech_stack_recommendations(self, tech_stack: str, 
                                           performance: Dict[str, Any]) -> List[str]:
        """기술스택별 권장사항 생성
        
        Args:
            tech_stack: 기술스택명
            performance: 성능 분석 결과
            
        Returns:
            List: 권장사항 리스트
        """
        recommendations = []
        
        best_model = performance.get('best_model')
        performance_gap = performance.get('performance_gap', {})
        
        # 최고 성능 모델 추천
        if best_model and best_model['name']:
            recommendations.append(
                f"🥇 {tech_stack} 최적 모델: {best_model['name']} "
                f"(점수: {best_model['score']:.3f}, 등급: {best_model['grade']})"
            )
        
        # 성능 격차 분석
        gap = performance_gap.get('gap', 0.0)
        if gap > 0.2:
            recommendations.append(
                f"⚠️ 모델 간 성능 격차가 큼 (격차: {gap:.3f}) - 모델 선택이 중요"
            )
        elif gap < 0.1:
            recommendations.append(
                f"✅ 모델 간 성능이 균등함 (격차: {gap:.3f}) - 안정적인 선택 가능"
            )
        
        # 메트릭별 특화 분석
        if best_model and best_model['metrics']:
            metrics = best_model['metrics']
            metric_names = {
                'correctness': '정확성',
                'clarity': '명확성',
                'actionability': '실행가능성',
                'json_correctness': 'JSON 정확성'
            }
            
            # 약한 메트릭 식별
            weak_metrics = [
                korean_name for metric, korean_name in metric_names.items()
                if metric in metrics and metrics[metric] < 0.7
            ]
            
            if weak_metrics:
                recommendations.append(
                    f"📈 {tech_stack}에서 개선 필요 영역: {', '.join(weak_metrics)}"
                )
        
        # 변동계수 기반 일관성 분석
        cv = performance_gap.get('coefficient_of_variation', 0.0)
        if cv > 0.3:
            recommendations.append(
                f"📊 성능 변동성이 높음 (CV: {cv:.3f}) - 모델 선택 시 신중 검토 필요"
            )
        elif cv < 0.1:
            recommendations.append(
                f"🎯 성능 일관성이 우수함 (CV: {cv:.3f}) - 안정적인 기술스택"
            )
        
        return recommendations
    
    def _cross_stack_comparison(self, tech_stack_analysis: Dict[str, Dict]) -> Dict[str, Any]:
        """기술스택 간 교차 비교
        
        Args:
            tech_stack_analysis: 기술스택별 분석 결과
            
        Returns:
            Dict: 교차 비교 결과
        """
        if len(tech_stack_analysis) < 2:
            return {
                'comparison_performed': False,
                'reason': 'Need at least 2 tech stacks for comparison'
            }
        
        # 기술스택별 최고 점수 수집
        stack_scores = {}
        stack_best_models = {}
        
        for tech_stack, analysis in tech_stack_analysis.items():
            best_model = analysis.get('best_model')
            if best_model and best_model['name']:
                stack_scores[tech_stack] = best_model['score']
                stack_best_models[tech_stack] = best_model['name']
        
        if not stack_scores:
            return {
                'comparison_performed': False,
                'reason': 'No valid performance data for comparison'
            }
        
        # 순위 생성
        ranked_stacks = sorted(stack_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 메트릭별 비교
        metric_comparison = self._compare_metrics_across_stacks(tech_stack_analysis)
        
        return {
            'comparison_performed': True,
            'stack_ranking': [
                {
                    'rank': i + 1,
                    'tech_stack': tech_stack,
                    'score': score,
                    'best_model': stack_best_models.get(tech_stack, 'Unknown')
                }
                for i, (tech_stack, score) in enumerate(ranked_stacks)
            ],
            'metric_comparison': metric_comparison,
            'performance_summary': {
                'best_stack': ranked_stacks[0][0] if ranked_stacks else None,
                'worst_stack': ranked_stacks[-1][0] if ranked_stacks else None,
                'score_range': {
                    'max': max(stack_scores.values()) if stack_scores else 0.0,
                    'min': min(stack_scores.values()) if stack_scores else 0.0,
                    'gap': max(stack_scores.values()) - min(stack_scores.values()) if stack_scores else 0.0
                }
            }
        }
    
    def _compare_metrics_across_stacks(self, tech_stack_analysis: Dict[str, Dict]) -> Dict[str, Dict]:
        """메트릭별 기술스택 간 비교
        
        Args:
            tech_stack_analysis: 기술스택별 분석 결과
            
        Returns:
            Dict: 메트릭별 비교 결과
        """
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        metric_comparison = {}
        
        for metric in metrics:
            stack_metric_scores = {}
            
            for tech_stack, analysis in tech_stack_analysis.items():
                best_model = analysis.get('best_model')
                if best_model and best_model['metrics'] and metric in best_model['metrics']:
                    stack_metric_scores[tech_stack] = best_model['metrics'][metric]
            
            if stack_metric_scores:
                # 메트릭별 순위
                ranked_stacks = sorted(stack_metric_scores.items(), key=lambda x: x[1], reverse=True)
                
                metric_comparison[metric] = {
                    'rankings': [
                        {
                            'rank': i + 1,
                            'tech_stack': tech_stack,
                            'score': score
                        }
                        for i, (tech_stack, score) in enumerate(ranked_stacks)
                    ],
                    'best_stack': ranked_stacks[0][0] if ranked_stacks else None,
                    'worst_stack': ranked_stacks[-1][0] if ranked_stacks else None,
                    'score_statistics': {
                        'mean': float(np.mean(list(stack_metric_scores.values()))),
                        'std': float(np.std(list(stack_metric_scores.values()))),
                        'max': float(max(stack_metric_scores.values())),
                        'min': float(min(stack_metric_scores.values()))
                    }
                }
        
        return metric_comparison
    
    def _generate_overall_recommendations(self, tech_stack_analysis: Dict[str, Dict], 
                                        cross_stack_comparison: Dict[str, Any]) -> List[str]:
        """전체 권장사항 생성
        
        Args:
            tech_stack_analysis: 기술스택별 분석 결과
            cross_stack_comparison: 교차 비교 결과
            
        Returns:
            List: 전체 권장사항 리스트
        """
        recommendations = []
        
        if not tech_stack_analysis:
            return ["분석할 기술스택 데이터가 없습니다."]
        
        # 기술스택별 요약
        recommendations.append(f"📊 분석된 기술스택: {len(tech_stack_analysis)}개")
        
        # 교차 비교 결과
        if cross_stack_comparison.get('comparison_performed'):
            performance_summary = cross_stack_comparison.get('performance_summary', {})
            best_stack = performance_summary.get('best_stack')
            worst_stack = performance_summary.get('worst_stack')
            
            if best_stack and worst_stack:
                recommendations.append(
                    f"🏆 최고 성능 기술스택: {best_stack}"
                )
                recommendations.append(
                    f"📈 개선 필요 기술스택: {worst_stack}"
                )
                
                score_range = performance_summary.get('score_range', {})
                gap = score_range.get('gap', 0.0)
                if gap > 0.2:
                    recommendations.append(
                        f"⚠️ 기술스택 간 성능 격차가 큼 (격차: {gap:.3f})"
                    )
        
        # 메트릭별 특화 권장사항
        if cross_stack_comparison.get('metric_comparison'):
            metric_comparison = cross_stack_comparison['metric_comparison']
            metric_names = {
                'correctness': '정확성',
                'clarity': '명확성',
                'actionability': '실행가능성',
                'json_correctness': 'JSON 정확성'
            }
            
            for metric, korean_name in metric_names.items():
                if metric in metric_comparison:
                    best_stack = metric_comparison[metric].get('best_stack')
                    if best_stack:
                        recommendations.append(
                            f"📈 {korean_name} 최고 기술스택: {best_stack}"
                        )
        
        # 개별 기술스택 권장사항 수집
        for tech_stack, analysis in tech_stack_analysis.items():
            stack_recommendations = analysis.get('recommendations', [])
            if stack_recommendations:
                recommendations.append(f"\n[{tech_stack}]")
                recommendations.extend(stack_recommendations)
        
        return recommendations