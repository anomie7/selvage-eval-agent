"""버전 비교 분석기

Selvage 버전별 성능 변화 분석 및 추적을 담당합니다.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import numpy as np
from sklearn.linear_model import LinearRegression

from .deepeval_log_parser import DeepEvalLogParser, TestCaseResult
from .metric_aggregator import MetricAggregator

logger = logging.getLogger(__name__)


class VersionComparisonAnalyzer:
    """버전 비교 분석기"""
    
    def __init__(self):
        self.version_pattern = re.compile(r'selvage\s+(\d+\.\d+\.\d+)')
        self.parser = DeepEvalLogParser()
        self.aggregator = MetricAggregator()
        
        # 임계값 설정
        self.regression_threshold = 0.05  # 5% 성능 저하
        self.improvement_threshold = 0.03  # 3% 성능 개선
        self.excellent_threshold = 0.8     # 우수 성능 80%
        self.needs_improvement_threshold = 0.7  # 개선 필요 70%
    
    def collect_version_data(self, base_path: str) -> Dict[str, Any]:
        """버전별 평가 데이터 수집
        
        Args:
            base_path: 평가 세션들이 저장된 기본 경로
            
        Returns:
            Dict: 버전별 수집된 데이터
        """
        logger.info(f"버전 데이터 수집 시작 - 기본 경로: {base_path}")
        
        base_dir = Path(base_path)
        if not base_dir.exists():
            logger.warning(f"Base path does not exist: {base_dir}")
            return {}
        
        version_data: Dict[str, Any] = {}
        session_dirs = list(base_dir.glob('*/'))
        logger.info(f"스캔할 세션 디렉토리 {len(session_dirs)}개 발견")
        
        # 모든 평가 세션 스캔
        for session_idx, session_dir in enumerate(session_dirs, 1):
            logger.debug(f"세션 {session_idx}/{len(session_dirs)} 처리 중: {session_dir.name}")
            if not session_dir.is_dir():
                continue
            
            try:
                # 메타데이터에서 버전 정보 추출
                metadata_path = session_dir / 'metadata.json'
                if not metadata_path.exists():
                    logger.debug(f"No metadata.json in {session_dir}")
                    continue
                
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 버전 정보 추출
                version = self._extract_version_from_metadata(metadata)
                if not version:
                    logger.debug(f"No version info in {session_dir}")
                    continue
                
                logger.debug(f"세션 '{session_dir.name}'에서 버전 '{version}' 감지")
                
                # 세션 결과 수집
                session_results = self._collect_session_results(session_dir)
                if not session_results:
                    logger.debug(f"No results in {session_dir}")
                    continue
                
                logger.debug(f"세션 '{session_dir.name}'에서 {len(session_results)}개 테스트 케이스 수집")
                
                # 실행 날짜 추출
                execution_date = self._extract_execution_date(metadata)
                
                if version not in version_data:
                    sessions_list: List[Dict[str, Any]] = []
                    dates_list: List[Optional[datetime]] = []
                    version_data[version] = {
                        'version': version,
                        'sessions': sessions_list,
                        'execution_dates': dates_list,
                        'latest_execution_date': None
                    }
                    logger.debug(f"새 버전 '{version}' 추가")
                
                version_data[version]['sessions'].append({
                    'session_dir': str(session_dir),
                    'results': session_results,
                    'execution_date': execution_date
                })
                
                version_data[version]['execution_dates'].append(execution_date)
                
            except Exception as e:
                logger.error(f"Error processing session {session_dir}: {e}")
                continue
        
        # 각 버전별로 평균 실행 날짜 계산
        for version, data in version_data.items():
            dates = [d for d in data['execution_dates'] if d]
            if dates:
                # 가장 최근 실행 날짜 사용
                data['latest_execution_date'] = max(dates)
                logger.debug(f"버전 '{version}' 최신 실행 날짜: {data['latest_execution_date']}")
            else:
                data['latest_execution_date'] = None
                logger.debug(f"버전 '{version}' 실행 날짜 정보 없음")
        
        total_sessions = sum(len(data['sessions']) for data in version_data.values())
        logger.info(f"버전 데이터 수집 완료 - {len(version_data)}개 버전, 총 {total_sessions}개 세션")
        
        return version_data
    
    def _extract_version_from_metadata(self, metadata: Dict) -> Optional[str]:
        """메타데이터에서 버전 정보 추출
        
        Args:
            metadata: 메타데이터 딕셔너리
            
        Returns:
            Optional[str]: 추출된 버전 정보
        """
        # 다양한 위치에서 버전 정보 탐색
        version_fields = [
            'selvage_version',
            'version',
            'tool_version',
            'selvage_tool_version'
        ]
        
        for field in version_fields:
            if field in metadata:
                version_str = str(metadata[field])
                match = self.version_pattern.search(version_str)
                if match:
                    return match.group(1)
        
        # 명령어나 설정에서 버전 정보 탐색
        if 'command' in metadata:
            command_str = str(metadata['command'])
            match = self.version_pattern.search(command_str)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_execution_date(self, metadata: Dict) -> Optional[datetime]:
        """메타데이터에서 실행 날짜 추출
        
        Args:
            metadata: 메타데이터 딕셔너리
            
        Returns:
            Optional[datetime]: 추출된 실행 날짜
        """
        date_fields = [
            'execution_date',
            'timestamp',
            'created_at',
            'start_time'
        ]
        
        for field in date_fields:
            if field in metadata:
                try:
                    date_str = str(metadata[field])
                    # ISO 형식 시도
                    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                except ValueError:
                    try:
                        # 다른 형식들 시도
                        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        continue
        
        return None
    
    def _collect_session_results(self, session_dir: Path) -> List[TestCaseResult]:
        """세션 디렉토리에서 모든 결과 수집
        
        Args:
            session_dir: 세션 디렉토리 경로
            
        Returns:
            List[TestCaseResult]: 수집된 테스트 케이스 결과
        """
        results = []
        
        # 로그 파일 탐색
        for log_file in session_dir.glob('**/*.log'):
            try:
                file_results = list(self.parser.parse_log_file(log_file))
                results.extend(file_results)
            except Exception as e:
                logger.warning(f"Error parsing log file {log_file}: {e}")
                continue
        
        return results
    
    def analyze_version_progression(self, version_data: Dict[str, Any]) -> Dict[str, Any]:
        """버전별 성능 발전 분석
        
        Args:
            version_data: 버전별 데이터
            
        Returns:
            Dict: 버전 발전 분석 결과
        """
        logger.info(f"버전 발전 분석 시작 - {len(version_data)}개 버전 분석")
        
        if not version_data:
            logger.info("분석할 버전 데이터가 없음")
            return {
                'version_timeline': [],
                'performance_trends': {},
                'regression_analysis': {},
                'improvement_highlights': [],
                'version_recommendations': {}
            }
        
        # 버전별 종합 성능 계산
        logger.info("버전별 종합 성능 계산 중...")
        version_performance = {}
        for version_idx, (version, data) in enumerate(version_data.items(), 1):
            logger.debug(f"버전 {version_idx}/{len(version_data)} '{version}' 성능 계산 중...")
            
            all_results = []
            for session in data['sessions']:
                all_results.extend(session['results'])
            
            if all_results:
                logger.debug(f"버전 '{version}' - {len(all_results)}개 테스트 케이스로 성능 계산")
                performance = self.aggregator.aggregate_model_performance(all_results)
                version_performance[version] = {
                    'version': version,
                    'performance': performance,
                    'latest_execution_date': data['latest_execution_date'],
                    'total_test_cases': len(all_results)
                }
                
                overall_score = performance.get('overall', {}).get('weighted_score', 0.0)
                logger.debug(f"버전 '{version}' 성능 계산 완료 - 전체 점수: {overall_score:.3f}")
            else:
                logger.warning(f"버전 '{version}'에 유효한 테스트 케이스가 없음")
        
        # 시간순 정렬
        logger.info("버전을 시간순으로 정렬 중...")
        sorted_versions = self._sort_versions_by_date(version_performance)
        logger.info(f"시간순 정렬 완료 - {len(sorted_versions)}개 버전")
        
        # 성능 트렌드 분석
        logger.info("성능 트렌드 분석 중...")
        performance_trends = self._analyze_performance_trends(sorted_versions)
        
        # 회귀 분석
        logger.info("성능 회귀 분석 중...")
        regression_analysis = self._detect_regressions(sorted_versions)
        
        # 개선 사항 식별
        logger.info("성능 개선사항 식별 중...")
        improvement_highlights = self._identify_improvements(sorted_versions)
        
        # 버전 권장사항 생성
        logger.info("버전 권장사항 생성 중...")
        version_recommendations = self._generate_version_recommendations(
            sorted_versions, {
                'performance_trends': performance_trends,
                'regression_analysis': regression_analysis,
                'improvement_highlights': improvement_highlights
            }
        )
        
        logger.info(f"버전 발전 분석 완료 - {len(improvement_highlights)}개 개선사항, {regression_analysis.get('total_regressions', 0)}개 회귀 발견")
        
        return {
            'version_timeline': sorted_versions,
            'performance_trends': performance_trends,
            'regression_analysis': regression_analysis,
            'improvement_highlights': improvement_highlights,
            'version_recommendations': version_recommendations
        }
    
    def _sort_versions_by_date(self, version_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """버전을 날짜순으로 정렬
        
        Args:
            version_performance: 버전별 성능 데이터
            
        Returns:
            List: 날짜순으로 정렬된 버전 리스트
        """
        versions_with_dates = []
        
        for version, data in version_performance.items():
            execution_date = data['latest_execution_date']
            if execution_date:
                versions_with_dates.append((execution_date, data))
            else:
                # 날짜가 없는 경우 버전 번호로 정렬
                version_tuple = tuple(map(int, version.split('.')))
                # 임의의 오래된 날짜 할당
                fake_date = datetime(2020, 1, 1)
                versions_with_dates.append((fake_date, data))
        
        # 날짜순 정렬
        versions_with_dates.sort(key=lambda x: x[0])
        
        return [data for _, data in versions_with_dates]
    
    def _analyze_performance_trends(self, sorted_versions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """성능 트렌드 분석
        
        Args:
            sorted_versions: 시간순 정렬된 버전 리스트
            
        Returns:
            Dict: 성능 트렌드 분석 결과
        """
        if len(sorted_versions) < 2:
            return {
                'analysis_performed': False,
                'reason': 'Need at least 2 versions for trend analysis'
            }
        
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        trends = {}
        
        for metric in metrics:
            scores = []
            for version_data in sorted_versions:
                performance = version_data['performance']
                if metric in performance:
                    scores.append(performance[metric]['mean_score'])
                else:
                    scores.append(0.0)
            
            # 선형 회귀 분석
            if len(scores) >= 2:
                X = np.array(range(len(scores))).reshape(-1, 1)
                y = np.array(scores)
                
                model = LinearRegression()
                model.fit(X, y)
                
                slope = model.coef_[0]
                r_squared = model.score(X, y)
                
                # 트렌드 분류
                if slope > 0.01:
                    trend_direction = 'improving'
                elif slope < -0.01:
                    trend_direction = 'declining'
                else:
                    trend_direction = 'stable'
                
                trends[metric] = {
                    'slope': float(slope),
                    'r_squared': float(r_squared),
                    'trend_direction': trend_direction,
                    'scores': scores,
                    'improvement_rate': float(slope * len(scores)) if len(scores) > 1 else 0.0
                }
        
        return {
            'analysis_performed': True,
            'metric_trends': trends,
            'overall_trend': self._calculate_overall_trend(trends)
        }
    
    def _calculate_overall_trend(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """전체 트렌드 계산
        
        Args:
            trends: 메트릭별 트렌드 데이터
            
        Returns:
            Dict: 전체 트렌드 정보
        """
        if not trends:
            return {'direction': 'unknown', 'strength': 0.0}
        
        # 가중 평균 계산
        from ..constants import METRIC_WEIGHTS
        weights = METRIC_WEIGHTS
        
        weighted_slope = sum(
            trends[metric]['slope'] * weights.get(metric, 0.25)
            for metric in trends
        )
        
        avg_r_squared = np.mean([trends[metric]['r_squared'] for metric in trends])
        
        # 전체 트렌드 방향 결정
        if weighted_slope > 0.01:
            direction = 'improving'
        elif weighted_slope < -0.01:
            direction = 'declining'
        else:
            direction = 'stable'
        
        return {
            'direction': direction,
            'strength': float(abs(weighted_slope)),
            'confidence': float(avg_r_squared)
        }
    
    def _detect_regressions(self, sorted_versions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """성능 회귀 탐지
        
        Args:
            sorted_versions: 시간순 정렬된 버전 리스트
            
        Returns:
            Dict: 회귀 분석 결과
        """
        if len(sorted_versions) < 2:
            return {
                'regressions_detected': False,
                'reason': 'Need at least 2 versions for regression detection'
            }
        
        regressions = []
        
        for i in range(1, len(sorted_versions)):
            current_version = sorted_versions[i]
            previous_version = sorted_versions[i-1]
            
            current_score = current_version['performance']['overall']['weighted_score']
            previous_score = previous_version['performance']['overall']['weighted_score']
            
            # 성능 저하 검사
            if previous_score > 0:
                regression_ratio = (previous_score - current_score) / previous_score
                
                if regression_ratio > self.regression_threshold:
                    # 회귀 심각도 평가
                    if regression_ratio > 0.15:
                        severity = 'critical'
                    elif regression_ratio > 0.10:
                        severity = 'major'
                    elif regression_ratio > 0.05:
                        severity = 'minor'
                    else:
                        severity = 'negligible'
                    
                    regressions.append({
                        'from_version': previous_version['version'],
                        'to_version': current_version['version'],
                        'regression_ratio': float(regression_ratio),
                        'severity': severity,
                        'previous_score': float(previous_score),
                        'current_score': float(current_score),
                        'affected_metrics': self._analyze_affected_metrics(previous_version, current_version)
                    })
        
        return {
            'regressions_detected': len(regressions) > 0,
            'total_regressions': len(regressions),
            'regressions': regressions,
            'stability_assessment': self._assess_stability(regressions)
        }
    
    def _analyze_affected_metrics(self, previous_version: Dict, current_version: Dict) -> List[Dict[str, Any]]:
        """영향받은 메트릭 분석
        
        Args:
            previous_version: 이전 버전 데이터
            current_version: 현재 버전 데이터
            
        Returns:
            List: 영향받은 메트릭 정보
        """
        affected_metrics = []
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        
        for metric in metrics:
            if metric in previous_version['performance'] and metric in current_version['performance']:
                prev_score = previous_version['performance'][metric]['mean_score']
                curr_score = current_version['performance'][metric]['mean_score']
                
                if prev_score > 0:
                    change_ratio = (curr_score - prev_score) / prev_score
                    
                    if abs(change_ratio) > 0.03:  # 3% 이상 변화
                        affected_metrics.append({
                            'metric': metric,
                            'previous_score': float(prev_score),
                            'current_score': float(curr_score),
                            'change_ratio': float(change_ratio),
                            'change_type': 'improvement' if change_ratio > 0 else 'regression'
                        })
        
        return affected_metrics
    
    def _assess_stability(self, regressions: List[Dict]) -> Dict[str, Any]:
        """안정성 평가
        
        Args:
            regressions: 회귀 정보 리스트
            
        Returns:
            Dict: 안정성 평가 결과
        """
        if not regressions:
            return {
                'stability_level': 'stable',
                'description': 'No significant regressions detected'
            }
        
        # 심각도별 카운트
        severity_counts: Dict[str, int] = {}
        for regression in regressions:
            severity = regression['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # 안정성 등급 결정
        if severity_counts.get('critical', 0) > 0:
            stability_level = 'unstable'
            description = f"Critical regressions detected: {severity_counts['critical']}"
        elif severity_counts.get('major', 0) > 1:
            stability_level = 'unstable'
            description = f"Multiple major regressions detected: {severity_counts['major']}"
        elif severity_counts.get('major', 0) > 0 or severity_counts.get('minor', 0) > 2:
            stability_level = 'moderately_stable'
            description = "Some regressions detected but manageable"
        else:
            stability_level = 'stable'
            description = "Minor regressions only"
        
        return {
            'stability_level': stability_level,
            'description': description,
            'severity_counts': severity_counts
        }
    
    def _identify_improvements(self, sorted_versions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """주목할 만한 개선사항 식별
        
        Args:
            sorted_versions: 시간순 정렬된 버전 리스트
            
        Returns:
            List: 개선사항 리스트
        """
        improvements = []
        
        for i in range(1, len(sorted_versions)):
            current_version = sorted_versions[i]
            previous_version = sorted_versions[i-1]
            
            current_score = current_version['performance']['overall']['weighted_score']
            previous_score = previous_version['performance']['overall']['weighted_score']
            
            # 성능 개선 검사
            if previous_score > 0:
                improvement_ratio = (current_score - previous_score) / previous_score
                
                if improvement_ratio > self.improvement_threshold:
                    # 개선 정도 평가
                    if improvement_ratio > 0.15:
                        improvement_level = 'breakthrough'
                    elif improvement_ratio > 0.08:
                        improvement_level = 'significant'
                    else:
                        improvement_level = 'moderate'
                    
                    improvements.append({
                        'from_version': previous_version['version'],
                        'to_version': current_version['version'],
                        'improvement_ratio': float(improvement_ratio),
                        'improvement_level': improvement_level,
                        'previous_score': float(previous_score),
                        'current_score': float(current_score),
                        'improved_metrics': self._analyze_improved_metrics(previous_version, current_version)
                    })
        
        return improvements
    
    def _analyze_improved_metrics(self, previous_version: Dict, current_version: Dict) -> List[Dict[str, Any]]:
        """개선된 메트릭 분석
        
        Args:
            previous_version: 이전 버전 데이터
            current_version: 현재 버전 데이터
            
        Returns:
            List: 개선된 메트릭 정보
        """
        improved_metrics = []
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        
        for metric in metrics:
            if metric in previous_version['performance'] and metric in current_version['performance']:
                prev_score = previous_version['performance'][metric]['mean_score']
                curr_score = current_version['performance'][metric]['mean_score']
                
                if prev_score > 0:
                    change_ratio = (curr_score - prev_score) / prev_score
                    
                    if change_ratio > 0.03:  # 3% 이상 개선
                        improved_metrics.append({
                            'metric': metric,
                            'previous_score': float(prev_score),
                            'current_score': float(curr_score),
                            'improvement_ratio': float(change_ratio)
                        })
        
        return improved_metrics
    
    def _generate_version_recommendations(self, sorted_versions: List[Dict[str, Any]], 
                                        analysis: Dict[str, Any]) -> Dict[str, Any]:
        """버전별 권장사항 생성
        
        Args:
            sorted_versions: 시간순 정렬된 버전 리스트
            analysis: 분석 결과
            
        Returns:
            Dict: 버전별 권장사항
        """
        if not sorted_versions:
            return {
                'recommended_version': None,
                'recommendations': ['분석할 버전 데이터가 없습니다.']
            }
        
        # 최고 성능 버전 식별
        best_version = max(
            sorted_versions, 
            key=lambda x: x['performance']['overall']['weighted_score']
        )
        
        # 최신 버전
        latest_version = sorted_versions[-1]
        
        recommendations = []
        
        # 성능 기반 권장사항
        best_score = best_version['performance']['overall']['weighted_score']
        if best_score >= self.excellent_threshold:
            recommendations.append(
                f"🏆 최고 성능 버전: {best_version['version']} "
                f"(점수: {best_score:.3f}, 등급: {best_version['performance']['overall']['grade']})"
            )
        
        # 안정성 기반 권장사항
        regression_analysis = analysis.get('regression_analysis', {})
        stability = regression_analysis.get('stability_assessment', {})
        
        if stability.get('stability_level') == 'stable':
            recommendations.append(f"✅ 안정성 우수: 전체적으로 안정적인 성능")
        elif stability.get('stability_level') == 'unstable':
            recommendations.append(f"⚠️ 안정성 주의: {stability.get('description', '')}")
        
        # 트렌드 기반 권장사항
        performance_trends = analysis.get('performance_trends', {})
        if performance_trends.get('analysis_performed'):
            overall_trend = performance_trends.get('overall_trend', {})
            trend_direction = overall_trend.get('direction', 'unknown')
            
            if trend_direction == 'improving':
                recommendations.append("📈 성능 개선 트렌드: 지속적인 성능 향상")
            elif trend_direction == 'declining':
                recommendations.append("📉 성능 하락 트렌드: 성능 저하 추세")
            else:
                recommendations.append("📊 성능 안정 트렌드: 일정한 성능 유지")
        
        # 버전 선택 권장사항
        if best_version['version'] == latest_version['version']:
            recommended_version = latest_version['version']
            recommendations.append(f"🎯 권장 버전: {recommended_version} (최고 성능 + 최신 버전)")
        else:
            latest_score = latest_version['performance']['overall']['weighted_score']
            score_diff = best_score - latest_score
            
            if score_diff > 0.05:  # 5% 이상 차이
                recommended_version = best_version['version']
                recommendations.append(
                    f"🎯 권장 버전: {recommended_version} (최고 성능 우선)"
                )
                recommendations.append(
                    f"💡 최신 버전 {latest_version['version']}은 성능이 {score_diff:.3f}점 낮음"
                )
            else:
                recommended_version = latest_version['version']
                recommendations.append(
                    f"🎯 권장 버전: {recommended_version} (최신 버전, 성능 차이 미미)"
                )
        
        # 개선사항 하이라이트
        improvements = analysis.get('improvement_highlights', [])
        if improvements:
            major_improvements = [
                imp for imp in improvements 
                if imp['improvement_level'] in ['breakthrough', 'significant']
            ]
            
            if major_improvements:
                recommendations.append(
                    f"🚀 주요 개선사항: {len(major_improvements)}개의 의미있는 성능 개선"
                )
        
        return {
            'recommended_version': recommended_version,
            'best_performance_version': best_version['version'],
            'latest_version': latest_version['version'],
            'recommendations': recommendations
        }