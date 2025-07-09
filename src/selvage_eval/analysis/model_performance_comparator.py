"""모델 성능 비교기

여러 AI 모델의 성능을 종합적으로 비교하고 분석합니다.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from scipy import stats

from .deepeval_log_parser import TestCaseResult
from .metric_aggregator import MetricAggregator

logger = logging.getLogger(__name__)


class ModelPerformanceComparator:
    """모델 성능 비교 분석기"""
    
    def __init__(self):
        self.significance_level = 0.05
        self.effect_size_thresholds = {
            'small': 0.2,
            'medium': 0.5,
            'large': 0.8
        }
        self.aggregator = MetricAggregator()
    
    def compare_models(self, model_results: Dict[str, List[TestCaseResult]]) -> Dict[str, Any]:
        """n개 모델 종합 성능 비교 분석
        
        Args:
            model_results: 모델별 테스트 결과 딕셔너리
            
        Returns:
            Dict: 모델 통계, 비교 표, 순위, 통계 분석, 권장사항을 포함한 종합 결과
        """
        if not model_results:
            return {
                'model_count': 0,
                'model_statistics': {},
                'comparison_table': {},
                'rankings': {},
                'statistical_analysis': {},
                'recommendations': []
            }
        
        # 모델별 기본 통계 계산
        model_statistics = {}
        for model_name, results in model_results.items():
            if results:
                model_statistics[model_name] = self.aggregator.aggregate_model_performance(results)
            else:
                model_statistics[model_name] = self.aggregator.aggregate_model_performance([])
        
        # 순위 계산
        rankings = self._calculate_model_rankings(model_statistics)
        
        # 비교 표 생성
        comparison_table = self._create_comparison_table(model_statistics, rankings)
        
        # 통계 분석
        statistical_analysis = self._n_model_statistical_analysis(model_results)
        
        # 권장사항 생성
        recommendations = self._generate_model_recommendations(model_statistics, rankings)
        
        return {
            'model_count': len(model_results),
            'model_statistics': model_statistics,
            'comparison_table': comparison_table,
            'rankings': rankings,
            'statistical_analysis': statistical_analysis,
            'recommendations': recommendations
        }
    
    def _create_comparison_table(self, model_stats: Dict[str, Dict], 
                               rankings: Dict[str, Any]) -> Dict[str, Any]:
        """n개 모델 종합 비교 표 생성
        
        Args:
            model_stats: 모델별 통계 데이터
            rankings: 모델 순위 정보
            
        Returns:
            Dict: 비교 표 데이터
        """
        comparison_data = []
        
        for model_name, stats in model_stats.items():
            if not stats or 'overall' not in stats:
                continue
                
            overall = stats['overall']
            
            # 티어 분류
            tier = self._classify_tier(overall['weighted_score'])
            
            model_data = {
                'model_name': model_name,
                'overall_score': round(overall['weighted_score'], 4),
                'overall_rank': rankings['overall_ranking'].get(model_name, len(model_stats)),
                'grade': overall['grade'],
                'tier': tier,
                'total_cases': overall['total_cases'],
                'overall_pass_rate': round(overall['pass_rate'], 4),
                'total_failures': overall['total_failures'],
                'consistency_score': round(overall['consistency'], 4)
            }
            
            # 메트릭별 상세 점수 및 순위 추가
            for metric in ['correctness', 'clarity', 'actionability', 'json_correctness']:
                if metric in stats:
                    model_data[f'{metric}_score'] = round(stats[metric]['mean_score'], 4)
                    model_data[f'{metric}_rank'] = rankings['metric_rankings'][metric].get(model_name, len(model_stats))
                    model_data[f'{metric}_pass_rate'] = round(stats[metric]['pass_rate'], 4)
                else:
                    model_data[f'{metric}_score'] = 0.0
                    model_data[f'{metric}_rank'] = len(model_stats)
                    model_data[f'{metric}_pass_rate'] = 0.0
            
            comparison_data.append(model_data)
        
        # 종합 점수 순으로 정렬
        comparison_data.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return {
            'table_data': comparison_data,
            'metrics': ['correctness', 'clarity', 'actionability', 'json_correctness'],
            'tier_distribution': self._calculate_tier_distribution(comparison_data),
            'summary': self._generate_comparison_summary(comparison_data)
        }
    
    def _calculate_model_rankings(self, model_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """모델 순위 계산
        
        Args:
            model_stats: 모델별 통계 데이터
            
        Returns:
            Dict: 순위 정보 (메트릭별, 종합)
        """
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        
        # 메트릭별 순위 계산
        metric_rankings = {}
        for metric in metrics:
            metric_scores = []
            for model_name, stats in model_stats.items():
                if metric in stats:
                    metric_scores.append((model_name, stats[metric]['mean_score']))
                else:
                    metric_scores.append((model_name, 0.0))
            
            # 점수 순으로 정렬 (높은 점수가 1위)
            metric_scores.sort(key=lambda x: x[1], reverse=True)
            
            metric_rankings[metric] = {
                model_name: rank + 1 
                for rank, (model_name, score) in enumerate(metric_scores)
            }
        
        # 종합 순위 계산
        overall_scores = []
        for model_name, stats in model_stats.items():
            if 'overall' in stats:
                overall_scores.append((model_name, stats['overall']['weighted_score']))
            else:
                overall_scores.append((model_name, 0.0))
        
        overall_scores.sort(key=lambda x: x[1], reverse=True)
        overall_ranking = {
            model_name: rank + 1 
            for rank, (model_name, score) in enumerate(overall_scores)
        }
        
        return {
            'metric_rankings': metric_rankings,
            'overall_ranking': overall_ranking,
            'ranking_summary': self._generate_ranking_summary(metric_rankings, overall_ranking)
        }
    
    def _n_model_statistical_analysis(self, model_results: Dict[str, List[TestCaseResult]]) -> Dict[str, Any]:
        """ANOVA 및 Kruskal-Wallis 검정 수행
        
        Args:
            model_results: 모델별 테스트 결과
            
        Returns:
            Dict: 통계 분석 결과
        """
        if len(model_results) < 2:
            return {
                'analysis_performed': False,
                'reason': 'Insufficient models for statistical comparison (need at least 2 models)'
            }
        
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        analysis_results = {
            'analysis_performed': True,
            'model_count': len(model_results),
            'metric_analyses': {}
        }
        
        for metric in metrics:
            metric_data = []
            model_names = []
            
            # 각 모델별 메트릭 점수 수집
            for model_name, results in model_results.items():
                if results:
                    scores = [getattr(result, metric).score for result in results]
                    metric_data.append(scores)
                    model_names.append(model_name)
            
            if len(metric_data) < 2 or any(len(scores) == 0 for scores in metric_data):
                analysis_results['metric_analyses'][metric] = {
                    'test_performed': False,
                    'reason': 'Insufficient data for analysis'
                }
                continue
            
            # ANOVA 검정
            try:
                anova_f_stat, anova_p_value = stats.f_oneway(*metric_data)
                anova_significant = anova_p_value < self.significance_level
            except Exception as e:
                logger.warning(f"ANOVA 검정 실패 ({metric}): {e}")
                anova_f_stat, anova_p_value, anova_significant = None, None, False
            
            # Kruskal-Wallis 검정
            try:
                kw_h_stat, kw_p_value = stats.kruskal(*metric_data)
                kw_significant = kw_p_value < self.significance_level
            except Exception as e:
                logger.warning(f"Kruskal-Wallis 검정 실패 ({metric}): {e}")
                kw_h_stat, kw_p_value, kw_significant = None, None, False
            
            analysis_results['metric_analyses'][metric] = {
                'test_performed': True,
                'sample_sizes': [len(scores) for scores in metric_data],
                'model_names': model_names,
                'anova': {
                    'f_statistic': anova_f_stat,
                    'p_value': anova_p_value,
                    'significant': anova_significant,
                    'interpretation': 'Models show significant differences' if anova_significant else 'No significant differences between models'
                },
                'kruskal_wallis': {
                    'h_statistic': kw_h_stat,
                    'p_value': kw_p_value,
                    'significant': kw_significant,
                    'interpretation': 'Models show significant differences' if kw_significant else 'No significant differences between models'
                }
            }
        
        return analysis_results
    
    def _generate_model_recommendations(self, model_stats: Dict[str, Dict], 
                                       rankings: Dict[str, Any]) -> List[str]:
        """모델 선택 권장사항 생성
        
        Args:
            model_stats: 모델별 통계 데이터
            rankings: 순위 정보
            
        Returns:
            List: 권장사항 문자열 리스트
        """
        recommendations = []
        
        if not model_stats or not rankings:
            return ["충분한 데이터가 없어 권장사항을 생성할 수 없습니다."]
        
        # 전체 최고 성능 모델
        overall_ranking = rankings['overall_ranking']
        if overall_ranking:
            best_model = min(overall_ranking.items(), key=lambda x: x[1])
            best_model_name = best_model[0]
            best_score = model_stats[best_model_name]['overall']['weighted_score']
            
            recommendations.append(
                f"🏆 전체 최고 성능: {best_model_name} (점수: {best_score:.3f}, 등급: {model_stats[best_model_name]['overall']['grade']})"
            )
        
        # 메트릭별 특화 모델
        metric_rankings = rankings['metric_rankings']
        metric_names = {
            'correctness': '정확성',
            'clarity': '명확성',
            'actionability': '실행가능성',
            'json_correctness': 'JSON 정확성'
        }
        
        for metric, korean_name in metric_names.items():
            if metric in metric_rankings:
                best_metric_model = min(metric_rankings[metric].items(), key=lambda x: x[1])
                best_metric_model_name = best_metric_model[0]
                best_metric_score = model_stats[best_metric_model_name][metric]['mean_score']
                
                recommendations.append(
                    f"📊 {korean_name} 최고: {best_metric_model_name} (점수: {best_metric_score:.3f})"
                )
        
        # 일관성 최고 모델
        consistency_scores = [
            (model_name, stats['overall']['consistency'])
            for model_name, stats in model_stats.items()
            if 'overall' in stats
        ]
        
        if consistency_scores:
            most_consistent = max(consistency_scores, key=lambda x: x[1])
            recommendations.append(
                f"🎯 일관성 최고: {most_consistent[0]} (일관성: {most_consistent[1]:.3f})"
            )
        
        # 성능 티어 분석
        tier_counts = {}
        for model_name, stats in model_stats.items():
            if 'overall' in stats:
                tier = self._classify_tier(stats['overall']['weighted_score'])
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        if tier_counts:
            recommendations.append(
                f"📈 성능 분포: " + ", ".join([f"{tier} {count}개" for tier, count in tier_counts.items()])
            )
        
        return recommendations
    
    def _classify_tier(self, score: float) -> str:
        """점수 기반 성능 티어 분류
        
        Args:
            score: 성능 점수
            
        Returns:
            str: 성능 티어
        """
        if score >= 0.85:
            return "Tier 1 (우수)"
        elif score >= 0.75:
            return "Tier 2 (양호)"
        elif score >= 0.65:
            return "Tier 3 (보통)"
        else:
            return "Tier 4 (개선필요)"
    
    def _calculate_tier_distribution(self, comparison_data: List[Dict]) -> Dict[str, int]:
        """티어 분포 계산
        
        Args:
            comparison_data: 비교 테이블 데이터
            
        Returns:
            Dict: 티어별 모델 수
        """
        tier_distribution = {}
        for model_data in comparison_data:
            tier = model_data['tier']
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
        
        return tier_distribution
    
    def _generate_comparison_summary(self, comparison_data: List[Dict]) -> Dict[str, Any]:
        """비교 요약 정보 생성
        
        Args:
            comparison_data: 비교 테이블 데이터
            
        Returns:
            Dict: 요약 정보
        """
        if not comparison_data:
            return {}
        
        scores = [data['overall_score'] for data in comparison_data]
        
        return {
            'total_models': len(comparison_data),
            'score_statistics': {
                'mean': round(np.mean(scores), 4),
                'std': round(np.std(scores), 4),
                'min': round(np.min(scores), 4),
                'max': round(np.max(scores), 4),
                'median': round(np.median(scores), 4)
            },
            'best_model': comparison_data[0]['model_name'],
            'worst_model': comparison_data[-1]['model_name'],
            'score_gap': round(comparison_data[0]['overall_score'] - comparison_data[-1]['overall_score'], 4)
        }
    
    def _generate_ranking_summary(self, metric_rankings: Dict[str, Dict], 
                                 overall_ranking: Dict[str, int]) -> Dict[str, Any]:
        """순위 요약 정보 생성
        
        Args:
            metric_rankings: 메트릭별 순위
            overall_ranking: 종합 순위
            
        Returns:
            Dict: 순위 요약 정보
        """
        summary = {
            'total_models': len(overall_ranking),
            'top_model': min(overall_ranking.items(), key=lambda x: x[1])[0] if overall_ranking else None,
            'metric_leaders': {}
        }
        
        # 메트릭별 1위 모델
        for metric, rankings in metric_rankings.items():
            if rankings:
                leader = min(rankings.items(), key=lambda x: x[1])[0]
                summary['metric_leaders'][metric] = leader
        
        return summary