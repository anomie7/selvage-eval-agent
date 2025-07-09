"""DeepEval 로그 파일 파서

대용량 DeepEval 로그 파일을 스트리밍 방식으로 파싱하여 메트릭 정보를 추출합니다.
"""

import re
import json
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from dataclasses import dataclass


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
            r'(✅|❌)\s+(\w+(?:\s+\w+)*)\s+.*?\(score:\s+([\d.]+),.*?reason:\s+"([^"]*)".*?error:\s+([^)]*)\)',
            re.MULTILINE
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
        for match in self.metrics_pattern.finditer(content):
            status_icon, metric_name, score, reason, error = match.groups()
            
            # 메트릭명 정규화
            metric_key = metric_name.lower().replace(' ', '_')
            if metric_key == 'json_correctness':
                metric_key = 'json_correctness'
            elif 'correctness' in metric_key:
                metric_key = 'correctness'
            
            metrics[metric_key] = MetricScore(
                score=float(score),
                passed=status_icon == '✅',
                reason=reason,
                error=error if error != 'None' else None
            )
        
        # 필수 메트릭 확인
        required_metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        for metric in required_metrics:
            if metric not in metrics:
                # 누락된 메트릭은 기본값으로 설정
                metrics[metric] = MetricScore(
                    score=0.0,
                    passed=False,
                    reason="메트릭 정보 없음",
                    error="로그에서 메트릭 정보를 찾을 수 없음"
                )
        
        # 입력/출력 데이터 추출
        input_match = re.search(r'input:\s*(\[.*?\])', content, re.DOTALL)
        output_match = re.search(r'actual output:\s*(\{.*?\})', content, re.DOTALL)
        
        input_data = input_match.group(1) if input_match else "{}"
        actual_output = output_match.group(1) if output_match else "{}"
        
        # TestCaseResult 생성
        try:
            return TestCaseResult(
                correctness=metrics['correctness'],
                clarity=metrics['clarity'],
                actionability=metrics['actionability'],
                json_correctness=metrics['json_correctness'],
                input_data=input_data,
                actual_output=actual_output,
                raw_content=content
            )
        except KeyError as e:
            print(f"WARNING: 필수 메트릭 누락: {e}")
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