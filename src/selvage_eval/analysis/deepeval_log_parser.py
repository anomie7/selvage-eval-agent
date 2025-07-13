"""DeepEval 로그 파일 파서

대용량 DeepEval 로그 파일을 스트리밍 방식으로 파싱하여 메트릭 정보를 추출합니다.
"""

import re
import json
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetricScore:
    """개별 메트릭 점수"""
    score: float
    passed: bool
    reason: str
    error: Optional[str] = None


@dataclass
class TestCaseResult:
    """테스트 케이스 결과"""
    correctness: MetricScore
    clarity: MetricScore
    actionability: MetricScore
    json_correctness: MetricScore
    input_data: str
    actual_output: str
    raw_content: str


class DeepEvalLogParser:
    """대용량 DeepEval 로그 파일 파서"""
    
    def __init__(self):
        self.test_case_separator = "=" * 70
        self.metrics_pattern = re.compile(
            r'(✅|❌)\s+([^[]+?)(?:\s*\[[^\]]*\])?\s+\(score:\s+([\d.]+),.*?reason:\s+(.*?),\s*error:\s+([^)]*)\)',
            re.MULTILINE | re.DOTALL
        )
        
    def parse_log_file(self, log_path: Path) -> Iterator[TestCaseResult]:
        """로그 파일을 스트리밍 방식으로 파싱
        
        Args:
            log_path: 파싱할 로그 파일 경로
            
        Yields:
            TestCaseResult: 개별 테스트 케이스 결과
        """
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 테스트 케이스 분리
        test_cases = content.split(self.test_case_separator)
        
        for test_case_content in test_cases:
            test_case_content = test_case_content.strip()
            if test_case_content:
                # "Overall Metric Pass Rates" 섹션 제외
                if "Overall Metric Pass Rates" in test_case_content:
                    logger.debug("'Overall Metric Pass Rates' 섹션 건너뛰기")
                    continue
                    
                result = self._parse_test_case(test_case_content.split('\n'))
                if result:
                    yield result
    
    def _parse_test_case(self, lines: list[str]) -> Optional[TestCaseResult]:
        """개별 테스트 케이스 파싱
        
        Args:
            lines: 테스트 케이스 라인들
            
        Returns:
            TestCaseResult: 파싱된 테스트 케이스 결과
        """
        content = ''.join(lines)
        
        # 메트릭 정보 추출
        metrics = {}
        matches = list(self.metrics_pattern.finditer(content))
        
        if not matches:
            logger.warning(f"메트릭 정보를 찾을 수 없습니다. 컨텐츠 미리보기: {content[:200]}...")
            logger.debug(f"전체 컨텐츠: {content}")
        
        logger.debug(f"총 {len(matches)}개의 메트릭 매칭 발견")
        
        for i, match in enumerate(matches):
            status_icon, metric_name, score, reason, error = match.groups()
            
            logger.debug(f"매칭 {i+1}: icon={status_icon}, name='{metric_name}', score={score}")
            logger.debug(f"reason 길이: {len(reason)} 문자")
            
            # 메트릭명 정규화
            metric_key = metric_name.strip().lower().replace(' ', '_')
            if 'json' in metric_key and 'correctness' in metric_key:
                metric_key = 'json_correctness'
            elif 'correctness' in metric_key:
                metric_key = 'correctness'
            elif 'clarity' in metric_key:
                metric_key = 'clarity'
            elif 'actionability' in metric_key:
                metric_key = 'actionability'
            
            logger.debug(f"메트릭 파싱 성공: {metric_key} = {score} ({status_icon})")
            
            metrics[metric_key] = MetricScore(
                score=float(score),
                passed=status_icon == '✅',
                reason=reason.strip(),
                error=error.strip() if error and error != 'None' else None
            )
        
        # 필수 메트릭 확인
        required_metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        missing_metrics = []
        for metric in required_metrics:
            if metric not in metrics:
                missing_metrics.append(metric)
                # 누락된 메트릭은 기본값으로 설정
                metrics[metric] = MetricScore(
                    score=0.0,
                    passed=False,
                    reason="메트릭 정보 없음",
                    error="로그에서 메트릭 정보를 찾을 수 없음"
                )
        
        if missing_metrics:
            logger.warning(f"누락된 메트릭: {missing_metrics}")
        
        # 입력/출력 데이터 추출
        input_match = re.search(r'input:\s*(\[.*?\])', content, re.DOTALL)
        output_match = re.search(r'actual output:\s*(\{.*?\})', content, re.DOTALL)
        
        input_data = input_match.group(1) if input_match else "{}"
        actual_output = output_match.group(1) if output_match else "{}"
        
        # TestCaseResult 생성
        try:
            result = TestCaseResult(
                correctness=metrics['correctness'],
                clarity=metrics['clarity'],
                actionability=metrics['actionability'],
                json_correctness=metrics['json_correctness'],
                input_data=input_data,
                actual_output=actual_output,
                raw_content=content
            )
            logger.debug(f"테스트 케이스 파싱 성공: {len(metrics)}개 메트릭")
            return result
        except KeyError as e:
            logger.error(f"필수 메트릭 누락: {e}")
            return None
    
    def convert_to_test_case_result(self, test_case_data: Dict[str, Any]) -> Optional[TestCaseResult]:
        """딕셔너리 형태의 테스트 케이스 데이터를 TestCaseResult로 변환
        
        Args:
            test_case_data: 파싱된 테스트 케이스 데이터
            
        Returns:
            TestCaseResult: 변환된 테스트 케이스 결과
        """
        metrics = test_case_data.get('metrics', {})
        
        # 각 메트릭을 MetricScore로 변환
        metric_scores = {}
        for metric_name in ['correctness', 'clarity', 'actionability', 'json_correctness']:
            if metric_name in metrics:
                metric_data = metrics[metric_name]
                metric_scores[metric_name] = MetricScore(
                    score=metric_data.get('score', 0.0),
                    passed=metric_data.get('passed', False),
                    reason=metric_data.get('reason', ''),
                    error=metric_data.get('error')
                )
            else:
                metric_scores[metric_name] = MetricScore(
                    score=0.0,
                    passed=False,
                    reason="메트릭 정보 없음",
                    error="로그에서 메트릭 정보를 찾을 수 없음"
                )
        
        return TestCaseResult(
            correctness=metric_scores['correctness'],
            clarity=metric_scores['clarity'],
            actionability=metric_scores['actionability'],
            json_correctness=metric_scores['json_correctness'],
            input_data=test_case_data.get('input', '{}'),
            actual_output=test_case_data.get('actual_output', '{}'),
            raw_content=test_case_data.get('raw_content', '')
        )