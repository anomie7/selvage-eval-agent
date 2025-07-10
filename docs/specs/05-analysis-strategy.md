# Selvage 평가 에이전트 - DeepEval 결과 분석 전략 (5단계)

## 개요

DeepEval 평가 결과를 체계적으로 분석하여 모델별 성능 비교, 실패 패턴 분석, Selvage 버전별 개선 추적을 위한 종합 분석 전략을 제시합니다.

## 1. DeepEval 결과 파일 구조 분석

### 1.1 실제 파일 구조

**기본 디렉토리 구조:**
```
~/Library/selvage-eval/deepeval_results/{session_id}/
├── metadata.json                    # 세션 메타데이터
├── {model_name_1}/                  # 모델별 디렉토리
│   ├── test_run_20250708_004754.log # 개별 평가 실행 로그
│   ├── test_run_20250708_005356.log
│   └── ...
├── {model_name_2}/
│   ├── test_run_20250708_004441.log
│   └── ...
└── ...
```

**실제 확인된 모델 디렉토리:**
- `gemini-2.5-flash/`
- `gemini-2.5-pro/`
- `o3/`
- `o4-mini-high/`
(주의 : 모델 디렉토리는 고정된 것이 아닌 세션마다 다를 수 있음 )

### 1.2 메타데이터 스키마

**metadata.json 구조:**
```json
{
  "selvage_version": "selvage 0.1.2",
  "execution_date": "2025-07-08T01:07:09.148180",
  "session_id": "eval_20250707_174243_e4df05f6",
  "deep_eval_test_case_path": "/Users/demin_coder/Library/selvage-eval/deep_eval_test_case/eval_20250707_174243_e4df05f6",
  "tool_name": "deepeval_executor",
  "created_by": "DeepEvalExecutorTool"
}
```

### 1.3 로그 파일 내용 구조

**파일 특성:**
- 크기: 평균 500KB+ (대용량 파일)
- 형식: 텍스트 로그 (JSON 형태 아님)
- 인코딩: UTF-8

**각 테스트 케이스 구조:**
```
======================================================================

Metrics Summary

  - ✅/❌ Correctness [GEval] (score: 1.0, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: "상세한 평가 이유...", error: None)
  - ✅/❌ Clarity [GEval] (score: 1.0, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: "상세한 평가 이유...", error: None)  
  - ✅/❌ Actionability [GEval] (score: 1.0, threshold: 0.7, strict: False, evaluation model: gemini-2.5-pro, reason: "상세한 평가 이유...", error: None)
  - ✅/❌ Json Correctness (score: 1.0, threshold: 1.0, strict: True, evaluation model: None, reason: "상세한 평가 이유...", error: None)

For test case:

  - input: [시스템 메시지 및 사용자 입력 JSON]
  - actual output: {Selvage가 생성한 실제 리뷰 결과 JSON}
```

## 2. 데이터 추출 및 처리 전략

### 2.1 대용량 로그 파일 파싱 전략

**문제점:**
- 단일 파일 크기가 500KB+ 로 매우 큼
- 메모리 효율적 처리 필요
- 구조화되지 않은 텍스트 형태

**해결 방안:**

```python
from typing import Iterator, Dict, Any, Optional
import re
from pathlib import Path

class DeepEvalLogParser:
    """대용량 DeepEval 로그 파일 파서"""
    
    def __init__(self):
        self.test_case_separator = "=" * 70
        self.metrics_pattern = re.compile(
            r'(✅|❌)\s+(\w+)\s+.*?\(score:\s+([\d.]+),.*?reason:\s+"([^"]*)".*?error:\s+([^)]*)\)'
        )
        
    def parse_log_file(self, log_path: Path) -> Iterator[Dict[str, Any]]:
        """로그 파일을 스트리밍 방식으로 파싱"""
        with open(log_path, 'r', encoding='utf-8') as f:
            current_test_case = None
            buffer = []
            
            for line in f:
                if self.test_case_separator in line:
                    if current_test_case:
                        yield self._parse_test_case(buffer)
                    current_test_case = {}
                    buffer = []
                else:
                    buffer.append(line)
            
            # 마지막 테스트 케이스 처리
            if buffer:
                yield self._parse_test_case(buffer)
    
    def _parse_test_case(self, lines: list[str]) -> Dict[str, Any]:
        """개별 테스트 케이스 파싱"""
        content = ''.join(lines)
        
        # 메트릭 정보 추출
        metrics = {}
        for match in self.metrics_pattern.finditer(content):
            status_icon, metric_name, score, reason, error = match.groups()
            metrics[metric_name.lower()] = {
                'status': status_icon,
                'score': float(score),
                'reason': reason,
                'error': error if error != 'None' else None,
                'passed': status_icon == '✅'
            }
        
        # 입력/출력 데이터 추출
        input_match = re.search(r'input:\s*(\[.*?\])', content, re.DOTALL)
        output_match = re.search(r'actual output:\s*(\{.*?\})', content, re.DOTALL)
        
        return {
            'metrics': metrics,
            'input': input_match.group(1) if input_match else None,
            'actual_output': output_match.group(1) if output_match else None,
            'raw_content': content
        }
```

### 2.2 메트릭 점수 집계 알고리즘

```python
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

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

class MetricAggregator:
    """메트릭 점수 집계기"""
    
    def aggregate_model_performance(self, 
                                  test_results: List[TestCaseResult]) -> Dict[str, Any]:
        """모델별 종합 성능 계산"""
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        aggregated = {}
        
        for metric in metrics:
            scores = [getattr(result, metric).score for result in test_results]
            passed_count = sum(1 for result in test_results 
                             if getattr(result, metric).passed)
            
            aggregated[metric] = {
                'mean_score': np.mean(scores),
                'median_score': np.median(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'pass_rate': passed_count / len(test_results),
                'total_cases': len(test_results),
                'passed_cases': passed_count,
                'failed_cases': len(test_results) - passed_count
            }
        
        # 종합 점수 계산 (가중평균)
        weights = {
            'correctness': 0.4,
            'clarity': 0.25, 
            'actionability': 0.25,
            'json_correctness': 0.1
        }
        
        overall_score = sum(
            aggregated[metric]['mean_score'] * weight 
            for metric, weight in weights.items()
        )
        
        aggregated['overall'] = {
            'weighted_score': overall_score,
            'grade': self._assign_grade(overall_score),
            'consistency': 1.0 - np.mean([aggregated[m]['std_score'] for m in metrics])
        }
        
        return aggregated
    
    def _assign_grade(self, score: float) -> str:
        """점수 기반 등급 할당"""
        if score >= 0.9: return 'A+'
        elif score >= 0.85: return 'A'
        elif score >= 0.8: return 'B+'
        elif score >= 0.75: return 'B'
        elif score >= 0.7: return 'C+'
        elif score >= 0.65: return 'C'
        elif score >= 0.6: return 'D'
        else: return 'F'
```

## 3. 실패 패턴 분석 체계

### 3.1 실패 사유 분류 체계

```python
from typing import List, Dict, Tuple
import numpy as np
import json

# FailureCategory enum 제거: Gemini가 자유롭게 카테고리를 생성하도록 함
# 더 유연하고 정확한 실패 패턴 분석을 위해 미리 정의된 카테고리 제한을 제거

class GeminiFailureAnalyzer:
    """Gemini 기반 실패 패턴 분석기"""
    
    def __init__(self):
        self.gemini_client = self._initialize_gemini_client()
        self.cache = {}  # 비용 효율성을 위한 캐시
        
        # Gemini 클라이언트가 초기화되지 않으면 예외 발생
        if not self.gemini_client:
            raise RuntimeError(
                "Gemini 클라이언트 초기화에 실패했습니다. "
                "GEMINI_API_KEY 환경 변수를 확인하고 API 키가 유효한지 확인해주세요."
            )
        
    def _initialize_gemini_client(self):
        """Gemini 클라이언트 초기화 (structured output 지원)"""
        try:
            import google.generativeai as genai
            import os
            
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("ERROR: GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
                return None
                
            genai.configure(api_key=api_key)
            
            # structured output을 위한 생성 설정
            generation_config = {
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "실패 사유를 분류한 카테고리명 (영어 스네이크케이스)"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "분류 신뢰도 (0.0-1.0)"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "카테고리 선택 이유 및 Selvage 개선 방향"
                        }
                    },
                    "required": ["category", "confidence", "explanation"]
                }
            }
            
            model = genai.GenerativeModel(
                'gemini-2.5-flash',
                generation_config=generation_config
            )
            
            # structured output 테스트
            test_prompt = """
테스트용 실패 사유: "The review output format is incorrect"
위 실패 사유를 JSON 형식으로 분류해주세요.
            """
            test_response = model.generate_content(test_prompt)
            
            if not test_response or not test_response.text:
                print("ERROR: Gemini structured output 테스트 실패")
                return None
                
            # JSON 파싱 테스트
            try:
                json.loads(test_response.text)
                print("Gemini 클라이언트 초기화 성공 (structured output 지원)")
            except json.JSONDecodeError:
                print("WARNING: structured output 테스트에서 JSON 파싱 실패, 계속 진행")
                
            return model
            
        except Exception as e:
            print(f"ERROR: Gemini 클라이언트 초기화 실패: {e}")
            return None
    
    def categorize_failure(self, reason: str, failed_metric: str = None) -> Tuple[str, float]:
        """실패 사유를 자유 형식 카테고리로 분류 (신뢰도 점수 포함)
        
        Args:
            reason: 실패 사유 텍스트
            failed_metric: 실패한 메트릭 이름 (correctness, clarity, actionability, json_correctness)
        """
        # 캐시 키에 failed_metric 포함
        cache_key = hash(f"{reason}:{failed_metric or 'unknown'}")
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Gemini 분류 실행 (필수)
        try:
            category, confidence = self._gemini_categorize_failure(reason, failed_metric)
            self.cache[cache_key] = (category, confidence)
            return category, confidence
        except Exception as e:
            error_msg = f"Gemini 분류 실패 - 메트릭: {failed_metric or 'unknown'}, 실패 사유: '{reason[:100]}{'...' if len(reason) > 100 else ''}', 오류: {e}"
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _gemini_categorize_failure(self, reason: str, failed_metric: str = None) -> Tuple[str, float]:
        """Gemini를 사용한 실패 사유 분류"""
        # 메트릭별 전문 컨텍스트 설명
        metric_contexts = {
            'correctness': {
                'description': '리뷰의 정확성 - 코드 이슈를 올바르게 식별했는가?',
                'common_failures': '이슈 미탐지, 잘못된 진단, 중요도 오판, 그말싸 오탐'
            },
            'clarity': {
                'description': '리뷰의 명확성 - 개발자가 이해하기 쉬운 설명인가?',
                'common_failures': '모호한 설명, 전문용어 남용, 불명확한 표현, 이해하기 어려운 구조'
            },
            'actionability': {
                'description': '실행가능성 - 구체적이고 실행 가능한 개선 방안을 제시했는가?',
                'common_failures': '추상적 제안, 비현실적 조치, 구체성 부족, 실행 방법 미제시'
            },
            'json_correctness': {
                'description': 'JSON 형식 유효성 - 올바른 JSON 스키마를 따르는가?',
                'common_failures': 'JSON 구문 오류, 스키마 불일치, 필드 누락, 데이터 타입 오류'
            }
        }
        
        # 메트릭 컨텍스트 추가
        metric_context = ""
        if failed_metric and failed_metric in metric_contexts:
            context = metric_contexts[failed_metric]
            metric_context = f"""
## 실패한 메트릭: {failed_metric.upper()}
**메트릭 설명:** {context['description']}
**일반적 실패 패턴:** {context['common_failures']}

이 메트릭에서 실패했다는 점을 고려하여 분석해주세요.
"""
        elif failed_metric:
            metric_context = f"""
## 실패한 메트릭: {failed_metric.upper()}
이 메트릭에서 실패했다는 점을 고려하여 분석해주세요.
"""
        
        prompt = f"""
## 컨텍스트
당신은 LLM 기반 코드 리뷰 도구 'Selvage'의 성능을 평가하는 전문가입니다.

### 평가 시스템 구조:
1. **Selvage**: AI 기반 코드 리뷰 도구 (피평가 시스템)
   - 코드 변경사항을 분석하여 리뷰 피드백 생성
   - JSON 형식으로 구조화된 리뷰 결과 출력

2. **DeepEval**: LLM 평가 프레임워크 (평가 시스템)
   - Selvage가 생성한 리뷰의 품질을 4가지 메트릭으로 평가
   - 각 메트릭은 0.7 이상 점수 시 통과, 미달 시 실패

### DeepEval 평가 메트릭:
- **Correctness**: 리뷰의 정확성 (코드 이슈를 올바르게 식별했는가?)
- **Clarity**: 리뷰의 명확성 (개발자가 이해하기 쉬운 설명인가?)
- **Actionability**: 실행가능성 (구체적이고 실행 가능한 개선 방안을 제시했는가?)
- **JSON Correctness**: JSON 형식 유효성 (올바른 JSON 스키마를 따르는가?)
{metric_context}
## 분석 대상
다음은 DeepEval이 Selvage의 리뷰 결과를 평가한 후, 특정 메트릭에서 **실패(threshold 0.7 미달)**한 사례의 실패 사유입니다.

**실패 사유:**
{reason}

## 분석 요청
위 실패 사유를 분석하여 Selvage의 코드 리뷰 품질 문제를 가장 잘 설명하는 카테고리명을 생성해주세요.

### 분류 관점:
- **코드 분석 능력**: 코드 이슈 탐지, 라인 번호 정확성, 중요도 판단 등
- **의사소통 능력**: 설명의 명확성, 개발자 친화적 표현, 전문용어 사용 등  
- **실용성**: 구체적 해결방안 제시, 실행 가능한 제안, 우선순위 가이드 등
- **기술적 정확성**: JSON 형식 준수, 스키마 일치성, 구조적 완성도 등
- **도메인 특화**: 보안, 성능, 가독성, 유지보수성 등 특정 영역 전문성

### 지침:
- 실패의 근본 원인을 반영하는 구체적인 카테고리명 생성
- 영어 스네이크케이스 사용 (예: "missing_security_vulnerabilities", "vague_improvement_suggestions")
- Selvage 개선에 직접적으로 도움이 되는 실행 가능한 카테고리명 선호
- 유사한 실패 패턴들을 그룹핑할 수 있는 일반적 수준의 추상화

응답 형식:
JSON 형식으로 다음 스키마를 따라 응답해주세요:
```json
{
  "category": "카테고리명 (영어 스네이크케이스)",
  "confidence": 0.95,
  "explanation": "카테고리 선택 이유 및 Selvage 개선 방향"
}
\`\`\`
"""
        
        response = self.gemini_client.generate_content(prompt)
        return self._parse_gemini_response(response.text)
    
    def _parse_gemini_response(self, response_text: str) -> Tuple[str, float]:
        """Gemini structured output 응답 파싱"""
        try:
            # JSON 파싱
            response_data = json.loads(response_text.strip())
            
            category = response_data.get('category', 'unknown_failure')
            confidence = float(response_data.get('confidence', 0.8))
            explanation = response_data.get('explanation', '')
            
            # 카테고리명 정규화 (공백을 언더스코어로 변환, 소문자 변환)
            category = category.lower().replace(' ', '_').replace('-', '_')
            
            # 신뢰도 범위 검증
            confidence = max(0.0, min(1.0, confidence))
            
            print(f"Gemini 분류 결과: {category} (신뢰도: {confidence:.3f}) - {explanation[:100]}")
            
            return category, confidence
            
        except json.JSONDecodeError as e:
            print(f"WARNING: Gemini JSON 파싱 실패: {e}")
            print(f"Raw response: {response_text[:200]}...")
            
            # fallback: 기본 문자열 파싱 시도
            return self._fallback_parse_response(response_text)
        
        except (KeyError, ValueError, TypeError) as e:
            print(f"WARNING: Gemini 응답 파싱 오류: {e}")
            return self._fallback_parse_response(response_text)
    
    def _fallback_parse_response(self, response_text: str) -> Tuple[str, float]:
        """JSON 파싱 실패 시 fallback 파싱"""
        # 기본 문자열 파싱 시도
        lines = response_text.strip().split('\n')
        category_str = "unknown_failure"
        confidence = 0.8
        
        for line in lines:
            line = line.strip()
            if '카테고리' in line and ':' in line:
                category_str = line.split(':')[1].strip()
            elif '신뢰도' in line and ':' in line:
                try:
                    confidence = float(line.split(':')[1].strip())
                except ValueError:
                    pass
        
        # 카테고리명 정규화
        category_str = category_str.lower().replace(' ', '_').replace('-', '_')
        print(f"Fallback 파싱 결과: {category_str} (신뢰도: {confidence})")
        
        return category_str, confidence


class FailurePatternAnalyzer:
    """실패 패턴 분석기 (통합 인터페이스)"""
    
    def __init__(self):
        self.gemini_analyzer = GeminiFailureAnalyzer()
    
    def analyze_failure_patterns(self, 
                               failed_cases: List[TestCaseResult]) -> Dict[str, Any]:
        """실패 패턴 종합 분석"""
        patterns = {
            'total_failures': len(failed_cases),
            'by_metric': {},
            'by_category': {},
            'critical_patterns': [],
            'confidence_scores': {}  # 분류 신뢰도 추가
        }
        
        # 메트릭별 실패 분석
        for metric in ['correctness', 'clarity', 'actionability', 'json_correctness']:
            metric_failures = [
                case for case in failed_cases 
                if not getattr(case, metric).passed
            ]
            
            categories = {}
            confidences = []
            for case in metric_failures:
                reason = getattr(case, metric).reason
                category, confidence = self.gemini_analyzer.categorize_failure(reason, metric)
                categories[category] = categories.get(category, 0) + 1
                confidences.append(confidence)
            
            patterns['by_metric'][metric] = {
                'total_failures': len(metric_failures),
                'failure_rate': len(metric_failures) / len(failed_cases) if failed_cases else 0,
                'categories': categories,
                'worst_cases': self._extract_worst_cases(metric_failures, metric),
                'avg_confidence': np.mean(confidences) if confidences else 0
            }
        
        # 전체 카테고리별 분석
        all_categories = {}
        all_confidences = []
        for case in failed_cases:
            for metric in ['correctness', 'clarity', 'actionability', 'json_correctness']:
                if not getattr(case, metric).passed:
                    reason = getattr(case, metric).reason
                    category, confidence = self.gemini_analyzer.categorize_failure(reason, metric)
                    all_categories[category] = all_categories.get(category, 0) + 1
                    all_confidences.append(confidence)
        
        patterns['by_category'] = all_categories
        patterns['confidence_scores']['overall'] = np.mean(all_confidences) if all_confidences else 0
        
        
        return patterns
    
    def _extract_worst_cases(self, failures: List[TestCaseResult], 
                           metric: str) -> List[Dict[str, Any]]:
        """가장 낮은 점수의 실패 케이스 추출"""
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
    
    # 개선 제안 기능 제거 - 현재 단계에서는 모델별/버전별 비교 분석에 집중
    # 개선 제안은 추후 필요 시 추가 예정
    

## 4. 모델별 성능 비교 분석

### 4.1 통계적 비교 방법론

```python
from scipy import stats
from typing import List, Dict, Tuple
import pandas as pd

class ModelPerformanceComparator:
    """모델 성능 비교 분석기"""
    
    def __init__(self):
        self.significance_level = 0.05
        self.effect_size_thresholds = {
            'small': 0.2,
            'medium': 0.5, 
            'large': 0.8
        }
    
    def compare_models(self, 
                      model_results: Dict[str, List[TestCaseResult]]) -> Dict[str, Any]:
        """n개 모델 종합 성능 비교 분석"""
        model_names = list(model_results.keys())
        
        # 모델별 기본 통계
        model_stats = {}
        for model_name, results in model_results.items():
            aggregator = MetricAggregator()
            model_stats[model_name] = aggregator.aggregate_model_performance(results)
        
        # 전체 순위 계산
        rankings = self._calculate_model_rankings(model_stats)
        
        # 종합 비교 표 생성
        comparison_table = self._create_comparison_table(model_stats, rankings)
        
        # n-모델 통계 분석 (ANOVA)
        statistical_analysis = self._n_model_statistical_analysis(model_results)
        
        return {
            'model_statistics': model_stats,
            'comparison_table': comparison_table,
            'rankings': rankings,
            'statistical_analysis': statistical_analysis,
            'recommendations': self._generate_model_recommendations(model_stats, rankings)
        }
    
    def _create_comparison_table(self, 
                               model_stats: Dict[str, Any],
                               rankings: Dict[str, Any]) -> Dict[str, Any]:
        """n개 모델 종합 비교 표 생성"""
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        models = list(model_stats.keys())
        
        # 비교 표 데이터 구성
        table_data = []
        for model in models:
            stats = model_stats[model]
            row = {
                'model': model,
                'overall_score': stats['overall']['weighted_score'],
                'overall_rank': next((entry['rank'] for entry in rankings['overall'] 
                                    if entry['model'] == model), 'N/A'),
                'total_test_cases': stats['overall']['total_cases'],
                'pass_rate': stats['overall']['pass_rate'],
                'failure_count': stats['overall']['total_failures'],
                'grade': stats['overall']['grade'],
                'tier': self._get_model_tier(stats['overall']['weighted_score'])
            }
            
            # 메트릭별 상세 점수 추가
            for metric in metrics:
                metric_data = stats[metric]
                row[f'{metric}_score'] = metric_data['mean_score']
                row[f'{metric}_rank'] = next((entry['rank'] for entry in rankings['by_metric'][metric] 
                                            if entry['model'] == model), 'N/A')
                row[f'{metric}_failures'] = metric_data['failure_count']
            
            table_data.append(row)
        
        # 점수별 정렬
        table_data.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return {
            'table_data': table_data,
            'column_headers': self._get_table_headers(),
            'summary_stats': self._calculate_table_summary(table_data),
            'performance_gaps': self._calculate_performance_gaps(table_data)
        }
    
    def _get_model_tier(self, score: float) -> str:
        """점수 기반 모델 등급 결정"""
        if score >= 0.85:
            return "Tier 1 (우수)"
        elif score >= 0.75:
            return "Tier 2 (양호)"
        elif score >= 0.65:
            return "Tier 3 (보통)"
        else:
            return "Tier 4 (개선필요)"
    
    def _get_table_headers(self) -> List[str]:
        """비교 표 헤더 정의"""
        return [
            'Model', 'Overall Score', 'Rank', 'Test Cases', 'Pass Rate', 
            'Failures', 'Grade', 'Tier', 'Correctness', 'Clarity', 
            'Actionability', 'JSON Correctness'
        ]
    
    def _calculate_table_summary(self, table_data: List[Dict]) -> Dict[str, Any]:
        """비교 표 요약 통계"""
        scores = [row['overall_score'] for row in table_data]
        return {
            'best_model': table_data[0]['model'] if table_data else None,
            'worst_model': table_data[-1]['model'] if table_data else None,
            'score_range': {
                'max': max(scores) if scores else 0,
                'min': min(scores) if scores else 0,
                'mean': np.mean(scores) if scores else 0,
                'std': np.std(scores) if scores else 0
            },
            'tier_distribution': self._count_tier_distribution(table_data)
        }
    
    def _count_tier_distribution(self, table_data: List[Dict]) -> Dict[str, int]:
        """등급별 모델 분포 계산"""
        tier_count = {}
        for row in table_data:
            tier = row['tier']
            tier_count[tier] = tier_count.get(tier, 0) + 1
        return tier_count
    
    def _calculate_performance_gaps(self, table_data: List[Dict]) -> Dict[str, float]:
        """성능 격차 분석"""
        if len(table_data) < 2:
            return {}
        
        best_score = table_data[0]['overall_score']
        worst_score = table_data[-1]['overall_score']
        
        return {
            'max_gap': best_score - worst_score,
            'relative_gap_percentage': ((best_score - worst_score) / worst_score) * 100 if worst_score > 0 else 0,
            'tier1_threshold_gap': max(0, 0.85 - worst_score),
            'competitive_threshold': 0.1  # 상위 10% 이내를 경쟁력 있는 모델로 간주
        }
    
    def _n_model_statistical_analysis(self, 
                                    model_results: Dict[str, List[TestCaseResult]]) -> Dict[str, Any]:
        """n개 모델 통계 분석 (ANOVA/Kruskal-Wallis)"""
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        analysis_results = {}
        
        for metric in metrics:
            # 각 모델별 점수 수집
            model_scores = {}
            for model_name, results in model_results.items():
                scores = [getattr(result, metric).score for result in results]
                model_scores[model_name] = scores
            
            # ANOVA 검정 (정규성 가정)
            score_groups = list(model_scores.values())
            if len(score_groups) >= 2 and all(len(group) > 0 for group in score_groups):
                try:
                    f_stat, p_value_anova = stats.f_oneway(*score_groups)
                    
                    # Kruskal-Wallis 검정 (비모수적 대안)
                    h_stat, p_value_kw = stats.kruskal(*score_groups)
                    
                    analysis_results[metric] = {
                        'anova': {
                            'f_statistic': f_stat,
                            'p_value': p_value_anova,
                            'significant': p_value_anova < self.significance_level
                        },
                        'kruskal_wallis': {
                            'h_statistic': h_stat,
                            'p_value': p_value_kw,
                            'significant': p_value_kw < self.significance_level
                        },
                        'interpretation': self._interpret_n_model_test_results(p_value_anova, p_value_kw)
                    }
                except Exception as e:
                    analysis_results[metric] = {
                        'error': f"통계 분석 실패: {str(e)}",
                        'interpretation': "충분한 데이터가 없거나 분석에 실패했습니다."
                    }
        
        return {
            'by_metric': analysis_results,
            'overall_conclusion': self._generate_statistical_conclusion(analysis_results)
        }
    
    def _interpret_n_model_test_results(self, p_anova: float, p_kw: float) -> str:
        """n-모델 통계 검정 결과 해석"""
        if p_anova < 0.001 and p_kw < 0.001:
            return "모델 간 매우 유의미한 성능 차이 존재"
        elif p_anova < 0.01 and p_kw < 0.01:
            return "모델 간 유의미한 성능 차이 존재"
        elif p_anova < 0.05 or p_kw < 0.05:
            return "모델 간 성능 차이가 존재할 가능성 있음"
        else:
            return "모델 간 유의미한 성능 차이 없음"
    
    def _generate_statistical_conclusion(self, analysis_results: Dict[str, Any]) -> str:
        """통계 분석 종합 결론"""
        significant_metrics = []
        for metric, result in analysis_results.items():
            if isinstance(result, dict) and 'anova' in result:
                if result['anova']['significant'] or result['kruskal_wallis']['significant']:
                    significant_metrics.append(metric)
        
        if len(significant_metrics) >= 3:
            return "대부분 메트릭에서 모델 간 유의미한 성능 차이가 확인됨"
        elif len(significant_metrics) >= 1:
            return f"{', '.join(significant_metrics)} 메트릭에서 모델 간 성능 차이 확인"
        else:
            return "모든 메트릭에서 모델 간 성능이 유사함"
    
    # 기존 pairwise comparison 관련 메서드들 - n-model 비교로 변경하면서 제거됨
    # def _calculate_cohens_d(): Cohen's d는 2개 그룹 비교용이므로 n-model 분석에서는 불필요
    # def _interpret_effect_size(): 마찬가지로 pairwise comparison용
    # def _assess_practical_significance(): 실용적 유의성도 pairwise comparison 맥락
    
    def _calculate_model_rankings(self, model_stats: Dict[str, Any]) -> Dict[str, Any]:
        """모델 순위 계산"""
        models = list(model_stats.keys())
        
        # 메트릭별 순위
        metric_rankings = {}
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness', 'overall']
        
        for metric in metrics:
            if metric == 'overall':
                scores = [(model, stats['overall']['weighted_score']) for model, stats in model_stats.items()]
            else:
                scores = [(model, stats[metric]['mean_score']) for model, stats in model_stats.items()]
            
            # 점수 순으로 정렬 (내림차순)
            ranked = sorted(scores, key=lambda x: x[1], reverse=True)
            metric_rankings[metric] = [
                {
                    'rank': i + 1,
                    'model': model,
                    'score': score,
                    'grade': model_stats[model].get('overall', {}).get('grade', 'N/A') if metric == 'overall' else 'N/A'
                }
                for i, (model, score) in enumerate(ranked)
            ]
        
        # 종합 순위 (가중 평균)
        overall_ranking = metric_rankings['overall']
        
        return {
            'by_metric': metric_rankings,
            'overall': overall_ranking,
            'champion': overall_ranking[0] if overall_ranking else None,
            'summary': self._create_ranking_summary(metric_rankings)
        }
    
    def _create_ranking_summary(self, metric_rankings: Dict[str, List]) -> Dict[str, Any]:
        """순위 요약 생성"""
        summary = {}
        
        # 각 메트릭별 1위 모델
        champions = {}
        for metric, ranking in metric_rankings.items():
            if ranking:
                champions[metric] = ranking[0]['model']
        
        # 가장 일관성 있는 모델 (여러 메트릭에서 상위권)
        model_positions = {}
        for metric, ranking in metric_rankings.items():
            for entry in ranking:
                model = entry['model']
                rank = entry['rank']
                if model not in model_positions:
                    model_positions[model] = []
                model_positions[model].append(rank)
        
        # 평균 순위 계산
        avg_rankings = {
            model: np.mean(positions) 
            for model, positions in model_positions.items()
        }
        most_consistent = min(avg_rankings.items(), key=lambda x: x[1])[0]
        
        return {
            'metric_champions': champions,
            'most_consistent_model': most_consistent,
            'average_rankings': avg_rankings,
            'performance_tiers': self._create_performance_tiers(metric_rankings['overall'])
        }
    
    def _create_performance_tiers(self, overall_ranking: List[Dict]) -> Dict[str, List[str]]:
        """성능 등급별 모델 분류"""
        tiers = {
            'Tier 1 (우수)': [],
            'Tier 2 (양호)': [],
            'Tier 3 (보통)': [],
            'Tier 4 (개선필요)': []
        }
        
        for entry in overall_ranking:
            score = entry['score']
            model = entry['model']
            
            if score >= 0.85:
                tiers['Tier 1 (우수)'].append(model)
            elif score >= 0.75:
                tiers['Tier 2 (양호)'].append(model)
            elif score >= 0.65:
                tiers['Tier 3 (보통)'].append(model)
            else:
                tiers['Tier 4 (개선필요)'].append(model)
        
        return tiers
    
    def _generate_model_recommendations(self, 
                                      model_stats: Dict[str, Any],
                                      rankings: Dict[str, Any]) -> List[str]:
        """모델 선택 권장사항 생성"""
        recommendations = []
        
        # 전체 최고 성능 모델
        best_overall = rankings['overall'][0] if rankings['overall'] else None
        if best_overall:
            recommendations.append(
                f"🏆 전체 최고 성능: {best_overall['model']} "
                f"(종합 점수: {best_overall['score']:.3f}, 등급: {best_overall['grade']})"
            )
        
        # 메트릭별 특화 모델
        metric_champions = rankings['summary']['metric_champions']
        for metric, champion in metric_champions.items():
            if metric != 'overall':
                score = model_stats[champion][metric]['mean_score']
                recommendations.append(
                    f"📊 {metric.title()} 최고: {champion} (점수: {score:.3f})"
                )
        
        # 일관성 최고 모델
        most_consistent = rankings['summary']['most_consistent_model']
        avg_rank = rankings['summary']['average_rankings'][most_consistent]
        recommendations.append(
            f"🎯 가장 일관성 있는 모델: {most_consistent} (평균 순위: {avg_rank:.1f})"
        )
        
        # 성능 개선 필요 모델
        poor_performers = rankings['summary']['performance_tiers']['Tier 4 (개선필요)']
        if poor_performers:
            recommendations.append(
                f"⚠️ 성능 개선 필요: {', '.join(poor_performers)}"
            )
        
        return recommendations

### 4.2 기술스택별 성능 차이 분석

```python
class TechStackAnalyzer:
    """기술스택별 모델 성능 분석"""
    
    def __init__(self):
        self.tech_stack_mapping = {
            'cline': 'TypeScript/JavaScript',
            'ecommerce-microservices': 'Java/Spring',
            'kotlin-realworld': 'Kotlin/JPA',
            'selvage-deprecated': 'Python'
        }
    
    def analyze_tech_stack_performance(self, 
                                     repo_results: Dict[str, Dict[str, List[TestCaseResult]]]) -> Dict[str, Any]:
        """저장소/기술스택별 모델 성능 분석"""
        
        analysis = {
            'by_tech_stack': {},
            'cross_stack_comparison': {},
            'recommendations': {}
        }
        
        # 기술스택별 분석
        for repo_name, model_results in repo_results.items():
            tech_stack = self.tech_stack_mapping.get(repo_name, 'Unknown')
            
            # 해당 기술스택에서의 모델별 성능
            stack_performance = {}
            for model_name, results in model_results.items():
                aggregator = MetricAggregator()
                stack_performance[model_name] = aggregator.aggregate_model_performance(results)
            
            # 최고 성능 모델 식별
            best_model = max(
                stack_performance.items(),
                key=lambda x: x[1]['overall']['weighted_score']
            )
            
            analysis['by_tech_stack'][tech_stack] = {
                'repository': repo_name,
                'model_performance': stack_performance,
                'best_model': {
                    'name': best_model[0],
                    'score': best_model[1]['overall']['weighted_score'],
                    'grade': best_model[1]['overall']['grade']
                },
                'performance_gap': self._calculate_performance_gap(stack_performance),
                'recommendations': self._generate_tech_stack_recommendations(
                    tech_stack, stack_performance
                )
            }
        
        # 기술스택 간 교차 비교
        analysis['cross_stack_comparison'] = self._cross_stack_comparison(analysis['by_tech_stack'])
        
        return analysis
    
    def _calculate_performance_gap(self, stack_performance: Dict[str, Any]) -> Dict[str, float]:
        """성능 격차 계산"""
        overall_scores = [
            perf['overall']['weighted_score'] 
            for perf in stack_performance.values()
        ]
        
        return {
            'max_score': max(overall_scores),
            'min_score': min(overall_scores),
            'gap': max(overall_scores) - min(overall_scores),
            'coefficient_variation': np.std(overall_scores) / np.mean(overall_scores)
        }
    
    def _generate_tech_stack_recommendations(self, 
                                           tech_stack: str,
                                           performance: Dict[str, Any]) -> List[str]:
        """기술스택별 권장사항"""
        recommendations = []
        
        # 최고 성능 모델 추천
        best_model = max(
            performance.items(),
            key=lambda x: x[1]['overall']['weighted_score']
        )
        recommendations.append(
            f"{tech_stack}에는 {best_model[0]} 모델 사용 권장 "
            f"(점수: {best_model[1]['overall']['weighted_score']:.3f})"
        )
        
        # 특정 메트릭 강화 필요성
        for model_name, perf in performance.items():
            weak_metrics = [
                metric for metric, data in perf.items() 
                if metric != 'overall' and data.get('mean_score', 0) < 0.7
            ]
            if weak_metrics:
                recommendations.append(
                    f"{model_name}의 {', '.join(weak_metrics)} 성능 개선 필요"
                )
        
        return recommendations
```

## 5. Selvage 버전별 성능 변화 추적

### 5.1 버전별 비교 분석 방법론

```python
from datetime import datetime, timedelta
import json
from pathlib import Path

class VersionComparisonAnalyzer:
    """Selvage 버전별 성능 변화 분석"""
    
    def __init__(self):
        self.version_pattern = re.compile(r'selvage\s+([\d.]+)')
        
    def collect_version_data(self, base_path: str) -> Dict[str, List[Dict]]:
        """버전별 평가 데이터 수집"""
        base_path = Path(base_path).expanduser()
        version_data = {}
        
        # 모든 평가 세션 스캔
        for session_dir in base_path.glob('eval_*'):
            if not session_dir.is_dir():
                continue
                
            # 메타데이터에서 버전 정보 추출
            metadata_file = session_dir / 'metadata.json'
            if not metadata_file.exists():
                continue
                
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            version = metadata.get('selvage_version', 'unknown')
            execution_date = metadata.get('execution_date')
            
            # 해당 세션의 모든 모델 결과 수집
            session_results = self._collect_session_results(session_dir)
            
            if version not in version_data:
                version_data[version] = []
            
            version_data[version].append({
                'session_id': session_dir.name,
                'execution_date': execution_date,
                'metadata': metadata,
                'results': session_results
            })
        
        return version_data
    
    def analyze_version_progression(self, 
                                  version_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """버전별 성능 발전 분석"""
        
        analysis = {
            'version_timeline': [],
            'performance_trends': {},
            'regression_analysis': {},
            'improvement_highlights': [],
            'version_recommendations': {}
        }
        
        # 버전별 종합 성능 계산
        version_performance = {}
        for version, sessions in version_data.items():
            aggregated_results = self._aggregate_version_results(sessions)
            
            aggregator = MetricAggregator()
            version_performance[version] = {
                'performance': aggregator.aggregate_model_performance(aggregated_results),
                'session_count': len(sessions),
                'latest_date': max(s['execution_date'] for s in sessions if s['execution_date'])
            }
        
        # 시간순 정렬
        sorted_versions = self._sort_versions_chronologically(version_performance)
        analysis['version_timeline'] = sorted_versions
        
        # 성능 트렌드 분석
        analysis['performance_trends'] = self._analyze_performance_trends(sorted_versions)
        
        # 회귀 분석 (성능 저하 탐지)
        analysis['regression_analysis'] = self._detect_regressions(sorted_versions)
        
        # 개선 하이라이트
        analysis['improvement_highlights'] = self._identify_improvements(sorted_versions)
        
        # 버전별 권장사항
        analysis['version_recommendations'] = self._generate_version_recommendations(
            sorted_versions, analysis
        )
        
        return analysis
    
    def _collect_session_results(self, session_dir: Path) -> List[TestCaseResult]:
        """세션 디렉토리에서 모든 결과 수집"""
        all_results = []
        
        for model_dir in session_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == 'metadata.json':
                continue
                
            parser = DeepEvalLogParser()
            for log_file in model_dir.glob('*.log'):
                for test_case_data in parser.parse_log_file(log_file):
                    # TestCaseResult 객체로 변환
                    result = self._convert_to_test_case_result(test_case_data)
                    if result:
                        all_results.append(result)
        
        return all_results
    
    def _aggregate_version_results(self, sessions: List[Dict]) -> List[TestCaseResult]:
        """버전 내 모든 세션 결과 통합"""
        all_results = []
        for session in sessions:
            all_results.extend(session['results'])
        return all_results
    
    def _sort_versions_chronologically(self, 
                                     version_performance: Dict[str, Any]) -> List[Dict]:
        """버전을 시간순으로 정렬"""
        version_list = []
        
        for version, data in version_performance.items():
            # 버전 번호 파싱 (예: "selvage 0.1.2" -> [0, 1, 2])
            version_match = self.version_pattern.search(version)
            if version_match:
                version_number = version_match.group(1)
                version_parts = [int(x) for x in version_number.split('.')]
            else:
                version_parts = [0, 0, 0]  # 알 수 없는 버전
            
            version_list.append({
                'version': version,
                'version_parts': version_parts,
                'latest_date': data['latest_date'],
                'performance': data['performance'],
                'session_count': data['session_count']
            })
        
        # 버전 번호 순으로 정렬
        return sorted(version_list, key=lambda x: x['version_parts'])
    
    def _analyze_performance_trends(self, sorted_versions: List[Dict]) -> Dict[str, Any]:
        """성능 트렌드 분석"""
        trends = {}
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        
        for metric in metrics:
            scores = []
            versions = []
            
            for version_data in sorted_versions:
                score = version_data['performance'][metric]['mean_score']
                scores.append(score)
                versions.append(version_data['version'])
            
            # 선형 회귀를 통한 트렌드 분석
            if len(scores) >= 2:
                x = np.arange(len(scores))
                z = np.polyfit(x, scores, 1)
                trend_slope = z[0]
                
                trends[metric] = {
                    'scores': scores,
                    'versions': versions,
                    'trend_slope': trend_slope,
                    'trend_direction': 'improving' if trend_slope > 0.01 else 'declining' if trend_slope < -0.01 else 'stable',
                    'total_change': scores[-1] - scores[0] if scores else 0,
                    'best_version': versions[scores.index(max(scores))] if scores else None,
                    'worst_version': versions[scores.index(min(scores))] if scores else None
                }
        
        # 전체 트렌드 요약
        overall_trend = np.mean([trends[m]['trend_slope'] for m in metrics])
        trends['overall'] = {
            'trend_slope': overall_trend,
            'trend_direction': 'improving' if overall_trend > 0.01 else 'declining' if overall_trend < -0.01 else 'stable'
        }
        
        return trends
    
    def _detect_regressions(self, sorted_versions: List[Dict]) -> Dict[str, Any]:
        """성능 회귀(저하) 탐지"""
        regressions = {
            'detected_regressions': [],
            'regression_summary': {},
            'severity_assessment': {}
        }
        
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        
        for i in range(1, len(sorted_versions)):
            current_version = sorted_versions[i]
            previous_version = sorted_versions[i-1]
            
            version_regressions = []
            
            for metric in metrics:
                current_score = current_version['performance'][metric]['mean_score']
                previous_score = previous_version['performance'][metric]['mean_score']
                
                # 의미있는 성능 저하 임계값: 5% 이상
                regression_threshold = 0.05
                
                if previous_score - current_score > regression_threshold:
                    severity = self._assess_regression_severity(
                        previous_score, current_score, metric
                    )
                    
                    version_regressions.append({
                        'metric': metric,
                        'previous_score': previous_score,
                        'current_score': current_score,
                        'regression_amount': previous_score - current_score,
                        'regression_percentage': ((previous_score - current_score) / previous_score) * 100,
                        'severity': severity
                    })
            
            if version_regressions:
                regressions['detected_regressions'].append({
                    'from_version': previous_version['version'],
                    'to_version': current_version['version'],
                    'regressions': version_regressions,
                    'overall_severity': max(r['severity'] for r in version_regressions)
                })
        
        return regressions
    
    def _assess_regression_severity(self, previous_score: float, 
                                  current_score: float, metric: str) -> str:
        """회귀 심각도 평가"""
        regression_amount = previous_score - current_score
        
        if regression_amount > 0.15:
            return 'critical'
        elif regression_amount > 0.10:
            return 'major'
        elif regression_amount > 0.05:
            return 'minor'
        else:
            return 'negligible'
    
    def _identify_improvements(self, sorted_versions: List[Dict]) -> List[Dict]:
        """주목할 만한 개선사항 식별"""
        improvements = []
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        
        for i in range(1, len(sorted_versions)):
            current_version = sorted_versions[i]
            previous_version = sorted_versions[i-1]
            
            version_improvements = []
            
            for metric in metrics:
                current_score = current_version['performance'][metric]['mean_score']
                previous_score = previous_version['performance'][metric]['mean_score']
                
                # 의미있는 개선 임계값: 3% 이상
                improvement_threshold = 0.03
                
                if current_score - previous_score > improvement_threshold:
                    improvement_magnitude = self._assess_improvement_magnitude(
                        previous_score, current_score
                    )
                    
                    version_improvements.append({
                        'metric': metric,
                        'previous_score': previous_score,
                        'current_score': current_score,
                        'improvement_amount': current_score - previous_score,
                        'improvement_percentage': ((current_score - previous_score) / previous_score) * 100,
                        'magnitude': improvement_magnitude
                    })
            
            if version_improvements:
                improvements.append({
                    'from_version': previous_version['version'],
                    'to_version': current_version['version'],
                    'improvements': version_improvements,
                    'overall_magnitude': max(i['magnitude'] for i in version_improvements)
                })
        
        return improvements
    
    def _assess_improvement_magnitude(self, previous_score: float, current_score: float) -> str:
        """개선 정도 평가"""
        improvement_amount = current_score - previous_score
        
        if improvement_amount > 0.15:
            return 'breakthrough'
        elif improvement_amount > 0.10:
            return 'significant'
        elif improvement_amount > 0.05:
            return 'moderate'
        else:
            return 'minor'
    
    def _generate_version_recommendations(self, 
                                        sorted_versions: List[Dict],
                                        analysis: Dict[str, Any]) -> Dict[str, Any]:
        """버전별 권장사항 생성"""
        recommendations = {
            'recommended_version': None,
            'version_assessments': {},
            'upgrade_path': [],
            'risk_assessment': {}
        }
        
        # 최고 성능 버전 식별
        best_version = max(
            sorted_versions,
            key=lambda x: x['performance']['overall']['weighted_score']
        )
        recommendations['recommended_version'] = best_version['version']
        
        # 각 버전별 평가
        for version_data in sorted_versions:
            version = version_data['version']
            performance = version_data['performance']
            
            # 장단점 분석
            strengths = []
            weaknesses = []
            
            for metric in ['correctness', 'clarity', 'actionability', 'json_correctness']:
                score = performance[metric]['mean_score']
                if score >= 0.8:
                    strengths.append(f"{metric} 우수 ({score:.3f})")
                elif score < 0.7:
                    weaknesses.append(f"{metric} 개선 필요 ({score:.3f})")
            
            recommendations['version_assessments'][version] = {
                'overall_score': performance['overall']['weighted_score'],
                'grade': performance['overall']['grade'],
                'strengths': strengths,
                'weaknesses': weaknesses,
                'stability_rating': self._assess_version_stability(version, analysis)
            }
        
        return recommendations
    
    def _assess_version_stability(self, version: str, analysis: Dict[str, Any]) -> str:
        """버전 안정성 평가"""
        # 회귀 분석에서 해당 버전이 관련된 경우 확인
        regressions = analysis.get('regression_analysis', {}).get('detected_regressions', [])
        
        for regression in regressions:
            if regression['to_version'] == version:
                if regression['overall_severity'] in ['critical', 'major']:
                    return 'unstable'
                elif regression['overall_severity'] == 'minor':
                    return 'moderately_stable'
        
        return 'stable'
```

## 6. 시각화 및 보고서 생성 전략

### 6.1 대시보드 설계

```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class VisualizationGenerator:
    """분석 결과 시각화 생성기"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'success': '#2ca02c', 
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'info': '#17becf'
        }
        
    def generate_comprehensive_dashboard(self, 
                                       analysis_results: Dict[str, Any],
                                       output_dir: str) -> List[str]:
        """종합 분석 대시보드 생성"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # 1. 모델 성능 비교 레이더 차트
        radar_file = self._create_model_performance_radar(
            analysis_results.get('model_comparison', {}), output_path
        )
        generated_files.append(radar_file)
        
        # 2. 메트릭별 성능 히트맵
        heatmap_file = self._create_performance_heatmap(
            analysis_results.get('model_comparison', {}), output_path
        )
        generated_files.append(heatmap_file)
        
        # 3. 실패 패턴 분석 차트
        failure_file = self._create_failure_pattern_charts(
            analysis_results.get('failure_analysis', {}), output_path
        )
        generated_files.append(failure_file)
        
        # 4. 버전별 성능 트렌드
        trend_file = self._create_version_trend_chart(
            analysis_results.get('version_analysis', {}), output_path
        )
        generated_files.append(trend_file)
        
        # 5. 기술스택별 성능 비교
        tech_stack_file = self._create_tech_stack_comparison(
            analysis_results.get('tech_stack_analysis', {}), output_path
        )
        generated_files.append(tech_stack_file)
        
        return generated_files
    
    def _create_model_performance_radar(self, 
                                      model_comparison: Dict[str, Any],
                                      output_path: Path) -> str:
        """모델 성능 레이더 차트 생성"""
        fig = go.Figure()
        
        model_stats = model_comparison.get('model_statistics', {})
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        
        for model_name, stats in model_stats.items():
            scores = [stats[metric]['mean_score'] for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=metrics,
                fill='toself',
                name=model_name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="모델별 성능 비교 (레이더 차트)",
            width=800,
            height=600
        )
        
        file_path = output_path / "model_performance_radar.html"
        fig.write_html(str(file_path))
        return str(file_path)
    
    def _create_performance_heatmap(self, 
                                  model_comparison: Dict[str, Any],
                                  output_path: Path) -> str:
        """성능 히트맵 생성"""
        model_stats = model_comparison.get('model_statistics', {})
        
        # 데이터 준비
        models = list(model_stats.keys())
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        
        heatmap_data = []
        for model in models:
            row = [model_stats[model][metric]['mean_score'] for metric in metrics]
            heatmap_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=metrics,
            y=models,
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            text=[[f"{val:.3f}" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="성능 점수")
        ))
        
        fig.update_layout(
            title="모델별 메트릭 성능 히트맵",
            xaxis_title="메트릭",
            yaxis_title="모델",
            width=800,
            height=500
        )
        
        file_path = output_path / "performance_heatmap.html"
        fig.write_html(str(file_path))
        return str(file_path)
    
    def _create_failure_pattern_charts(self, 
                                     failure_analysis: Dict[str, Any],
                                     output_path: Path) -> str:
        """실패 패턴 분석 차트 생성"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('메트릭별 실패 분포', '실패 카테고리별 분포', 
                          '심각도별 실패 건수', '개선 우선순위'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 메트릭별 실패 분포
        if 'by_metric' in failure_analysis:
            metrics = list(failure_analysis['by_metric'].keys())
            failure_counts = [failure_analysis['by_metric'][m]['total_failures'] for m in metrics]
            
            fig.add_trace(
                go.Bar(x=metrics, y=failure_counts, name="실패 건수"),
                row=1, col=1
            )
        
        # 실패 카테고리별 분포 (파이 차트)
        if 'by_category' in failure_analysis:
            categories = list(failure_analysis['by_category'].keys())
            category_counts = list(failure_analysis['by_category'].values())
            
            fig.add_trace(
                go.Pie(labels=categories, values=category_counts, name="카테고리"),
                row=1, col=2
            )
        
        fig.update_layout(
            title_text="실패 패턴 종합 분석",
            showlegend=True,
            height=800,
            width=1200
        )
        
        file_path = output_path / "failure_pattern_analysis.html"
        fig.write_html(str(file_path))
        return str(file_path)
    
    def _create_version_trend_chart(self, 
                                  version_analysis: Dict[str, Any],
                                  output_path: Path) -> str:
        """버전별 성능 트렌드 차트 생성"""
        trends = version_analysis.get('performance_trends', {})
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Correctness 트렌드', 'Clarity 트렌드', 
                          'Actionability 트렌드', 'JSON Correctness 트렌드']
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        metrics = ['correctness', 'clarity', 'actionability', 'json_correctness']
        
        for i, metric in enumerate(metrics):
            if metric in trends:
                trend_data = trends[metric]
                versions = trend_data['versions']
                scores = trend_data['scores']
                
                row, col = positions[i]
                
                # 실제 점수
                fig.add_trace(
                    go.Scatter(
                        x=versions, y=scores,
                        mode='lines+markers',
                        name=f"{metric} 점수",
                        line=dict(width=3)
                    ),
                    row=row, col=col
                )
                
                # 트렌드 라인
                x_numeric = list(range(len(versions)))
                z = np.polyfit(x_numeric, scores, 1)
                trend_line = np.poly1d(z)
                
                fig.add_trace(
                    go.Scatter(
                        x=versions, y=trend_line(x_numeric),
                        mode='lines',
                        name=f"{metric} 트렌드",
                        line=dict(dash='dash', width=2)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title_text="Selvage 버전별 성능 트렌드",
            height=800,
            width=1200,
            showlegend=True
        )
        
        file_path = output_path / "version_trend_analysis.html"
        fig.write_html(str(file_path))
        return str(file_path)
```

## 7. 종합 분석 엔진

### 7.1 메인 분석 엔진 클래스

```python
"""DeepEval 분석 엔진

DeepEval 평가 결과를 분석하고 통합된 마크다운 보고서를 생성하는 엔진입니다.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """모델 성능 데이터"""
    model_name: str
    total_test_cases: int
    correctness_mean: float
    correctness_std: float
    clarity_mean: float
    clarity_std: float
    actionability_mean: float
    actionability_std: float
    json_correctness_mean: float
    json_correctness_std: float
    overall_score: float
    pass_rate: float
    grade: str


@dataclass
class FailurePattern:
    """실패 패턴 데이터"""
    metric_name: str
    total_failures: int
    missing_issues: int
    incorrect_line_numbers: int
    inappropriate_severity: int
    unclear_descriptions: int
    non_actionable_suggestions: int
    json_format_errors: int
    other: int


class DeepEvalAnalysisEngine:
    """DeepEval 분석 엔진"""
    
    def __init__(self, output_dir: str = "~/Library/selvage-eval/analyze_results"):
        """분석 엔진 초기화
        
        Args:
            output_dir: 분석 결과 출력 디렉토리
        """
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_session(self, session_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """세션 분석 실행
        
        Args:
            session_path: DeepEval 결과가 있는 세션 경로
            output_dir: 사용자 지정 출력 디렉토리 (선택사항)
            
        Returns:
            분석 결과 메타데이터
        """
        session_path = Path(session_path).expanduser()
        
        if not session_path.exists():
            raise FileNotFoundError(f"세션 경로가 존재하지 않습니다: {session_path}")
        
        # 출력 디렉토리 설정
        if output_dir:
            final_output_dir = Path(output_dir).expanduser()
        else:
            session_id = session_path.name
            final_output_dir = self.output_dir / session_id
        
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # DeepEval 결과 수집
        deepeval_results = self._collect_deepeval_results(session_path)
        
        if not deepeval_results:
            raise ValueError("DeepEval 결과를 찾을 수 없습니다")
        
        # 분석 실행
        analysis_data = self._perform_comprehensive_analysis(deepeval_results)
        
        # 통합 마크다운 보고서 생성
        markdown_report = self._generate_markdown_report(analysis_data)
        report_path = final_output_dir / "analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        # JSON 데이터 저장 (프로그래밍 활용용)
        json_data = self._prepare_json_data(analysis_data)
        json_path = final_output_dir / "analysis_data.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # 선택적 인터랙티브 대시보드 생성
        dashboard_path = None
        if self._should_generate_dashboard(analysis_data):
            dashboard_path = self._generate_interactive_dashboard(analysis_data, final_output_dir)
        
        return {
            "analysis_metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "session_path": str(session_path),
                "total_test_cases": analysis_data["data_summary"]["total_test_cases"],
                "models_analyzed": analysis_data["models_analyzed"]
            },
            "files_generated": {
                "markdown_report": str(report_path),
                "json_data": str(json_path),
                "interactive_dashboard": dashboard_path
            }
        }
    
    def _collect_deepeval_results(self, session_path: Path) -> List[Dict[str, Any]]:
        """DeepEval 결과 수집"""
        results = []
        
        # deepeval_results_*.json 패턴의 파일들 찾기
        for result_file in session_path.glob("deepeval_results_*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                # 파일명에서 모델명 추출
                filename = result_file.stem
                parts = filename.split('_')
                if len(parts) >= 3:
                    model_name = '_'.join(parts[2:])  # deepeval_results_이후 부분
                else:
                    model_name = "unknown"
                
                results.append({
                    "model_name": model_name,
                    "file_path": str(result_file),
                    "data": result_data
                })
                
            except Exception as e:
                logger.warning(f"DeepEval 결과 파일 로드 실패: {result_file} - {e}")
        
        return results
    
    def _perform_comprehensive_analysis(self, deepeval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """종합 분석 수행"""
        models_analyzed = [r["model_name"] for r in deepeval_results]
        
        # 모델별 성능 분석
        model_performances = []
        for result in deepeval_results:
            performance = self._analyze_model_performance(result)
            model_performances.append(performance)
        
        # 실패 패턴 분석
        failure_patterns = self._analyze_failure_patterns(deepeval_results)
        
        # 기술스택별 분석 (현재는 단순화)
        tech_stack_analysis = self._analyze_tech_stack_performance(model_performances)
        
        # 통계 분석
        statistical_analysis = self._perform_statistical_analysis(model_performances)
        
        # 권장사항 생성
        recommendations = self._generate_recommendations(model_performances, failure_patterns)
        
        return {
            "models_analyzed": models_analyzed,
            "model_performances": model_performances,
            "failure_patterns": failure_patterns,
            "tech_stack_analysis": tech_stack_analysis,
            "statistical_analysis": statistical_analysis,
            "recommendations": recommendations,
            "data_summary": {
                "total_test_cases": sum(len(r["data"].get("testCases", [])) for r in deepeval_results),
                "successful_evaluations": sum(
                    len([tc for tc in r["data"].get("testCases", []) if tc.get("success", True)])
                    for r in deepeval_results
                ),
                "models_count": len(deepeval_results)
            }
        }
    
    def _analyze_model_performance(self, result: Dict[str, Any]) -> ModelPerformance:
        """개별 모델 성능 분석"""
        model_name = result["model_name"]
        test_cases = result["data"].get("testCases", [])
        
        if not test_cases:
            return ModelPerformance(
                model_name=model_name,
                total_test_cases=0,
                correctness_mean=0.0, correctness_std=0.0,
                clarity_mean=0.0, clarity_std=0.0,
                actionability_mean=0.0, actionability_std=0.0,
                json_correctness_mean=0.0, json_correctness_std=0.0,
                overall_score=0.0, pass_rate=0.0, grade="F"
            )
        
        # 메트릭별 점수 수집
        metric_scores = {
            "correctness": [],
            "clarity": [],
            "actionability": [],
            "json_correctness": []
        }
        
        for test_case in test_cases:
            metrics_data = test_case.get("metricsData", [])
            for metric in metrics_data:
                metric_name = metric.get("name", "").lower().replace(" ", "_")
                if metric_name in metric_scores:
                    score = metric.get("score", 0.0)
                    if isinstance(score, (int, float)):
                        metric_scores[metric_name].append(float(score))
        
        # 통계 계산
        def calc_stats(scores):
            if not scores:
                return 0.0, 0.0
            return float(np.mean(scores)), float(np.std(scores))
        
        correctness_mean, correctness_std = calc_stats(metric_scores["correctness"])
        clarity_mean, clarity_std = calc_stats(metric_scores["clarity"])
        actionability_mean, actionability_std = calc_stats(metric_scores["actionability"])
        json_correctness_mean, json_correctness_std = calc_stats(metric_scores["json_correctness"])
        
        # 전체 점수 계산 (가중평균)
        weights = {"correctness": 0.4, "clarity": 0.25, "actionability": 0.25, "json_correctness": 0.1}
        overall_score = (
            correctness_mean * weights["correctness"] +
            clarity_mean * weights["clarity"] +
            actionability_mean * weights["actionability"] +
            json_correctness_mean * weights["json_correctness"]
        )
        
        # 합격률 계산 (0.7 이상)
        all_scores = []
        for scores in metric_scores.values():
            all_scores.extend(scores)
        pass_rate = len([s for s in all_scores if s >= 0.7]) / len(all_scores) if all_scores else 0.0
        
        # 등급 할당
        if overall_score >= 0.9:
            grade = "A"
        elif overall_score >= 0.8:
            grade = "B"
        elif overall_score >= 0.7:
            grade = "C"
        elif overall_score >= 0.6:
            grade = "D"
        else:
            grade = "F"
        
        return ModelPerformance(
            model_name=model_name,
            total_test_cases=len(test_cases),
            correctness_mean=correctness_mean,
            correctness_std=correctness_std,
            clarity_mean=clarity_mean,
            clarity_std=clarity_std,
            actionability_mean=actionability_mean,
            actionability_std=actionability_std,
            json_correctness_mean=json_correctness_mean,
            json_correctness_std=json_correctness_std,
            overall_score=overall_score,
            pass_rate=pass_rate,
            grade=grade
        )
    
    def _analyze_failure_patterns(self, deepeval_results: List[Dict[str, Any]]) -> List[FailurePattern]:
        """실패 패턴 분석"""
        patterns = []
        metrics = ["correctness", "clarity", "actionability", "json_correctness"]
        
        for metric in metrics:
            failure_reasons = []
            
            for result in deepeval_results:
                test_cases = result["data"].get("testCases", [])
                for test_case in test_cases:
                    metrics_data = test_case.get("metricsData", [])
                    for metric_data in metrics_data:
                        if (metric_data.get("name", "").lower().replace(" ", "_") == metric and
                            not metric_data.get("success", True)):
                            reason = metric_data.get("reason", "").lower()
                            failure_reasons.append(reason)
            
            # 실패 유형별 분류
            missing_issues = len([r for r in failure_reasons if "missing" in r or "not identified" in r])
            incorrect_line_numbers = len([r for r in failure_reasons if "line number" in r or "incorrect" in r])
            inappropriate_severity = len([r for r in failure_reasons if "severity" in r or "inappropriate" in r])
            unclear_descriptions = len([r for r in failure_reasons if "unclear" in r or "vague" in r])
            non_actionable_suggestions = len([r for r in failure_reasons if "actionable" in r or "implementable" in r])
            json_format_errors = len([r for r in failure_reasons if "json" in r or "format" in r])
            
            categorized_count = (missing_issues + incorrect_line_numbers + inappropriate_severity +
                               unclear_descriptions + non_actionable_suggestions + json_format_errors)
            other = len(failure_reasons) - categorized_count
            
            patterns.append(FailurePattern(
                metric_name=metric,
                total_failures=len(failure_reasons),
                missing_issues=missing_issues,
                incorrect_line_numbers=incorrect_line_numbers,
                inappropriate_severity=inappropriate_severity,
                unclear_descriptions=unclear_descriptions,
                non_actionable_suggestions=non_actionable_suggestions,
                json_format_errors=json_format_errors,
                other=max(0, other)
            ))
        
        return patterns
    
    def _analyze_tech_stack_performance(self, model_performances: List[ModelPerformance]) -> Dict[str, Any]:
        """기술스택별 성능 분석 (현재는 단순화)"""
        if not model_performances:
            return {}
        
        # 전체 평균 기준으로 분석
        overall_scores = [mp.overall_score for mp in model_performances]
        avg_score = np.mean(overall_scores) if overall_scores else 0.0
        
        return {
            "overall_average": float(avg_score),
            "best_performing_model": max(model_performances, key=lambda x: x.overall_score).model_name,
            "most_consistent_model": min(model_performances, key=lambda x: x.correctness_std).model_name
        }
    
    def _perform_statistical_analysis(self, model_performances: List[ModelPerformance]) -> Dict[str, Any]:
        """통계 분석"""
        if not model_performances:
            return {}
        
        scores = [mp.overall_score for mp in model_performances]
        
        return {
            "mean_performance": float(np.mean(scores)),
            "std_performance": float(np.std(scores)),
            "min_performance": float(np.min(scores)),
            "max_performance": float(np.max(scores)),
            "performance_range": float(np.max(scores) - np.min(scores)),
            "models_above_threshold": len([s for s in scores if s >= 0.7])
        }
    
    def _generate_recommendations(self, model_performances: List[ModelPerformance], 
                                failure_patterns: List[FailurePattern]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        if not model_performances:
            return recommendations
        
        # 최고 성능 모델
        best_model = max(model_performances, key=lambda x: x.overall_score)
        recommendations.append(
            f"🏆 전체 최고 성능: {best_model.model_name} (종합 점수: {best_model.overall_score:.3f}, 등급: {best_model.grade})"
        )
        
        # 메트릭별 최고 성능
        best_correctness = max(model_performances, key=lambda x: x.correctness_mean)
        recommendations.append(
            f"📊 Correctness 최고: {best_correctness.model_name} (점수: {best_correctness.correctness_mean:.3f})"
        )
        
        # 개선이 필요한 영역
        for pattern in failure_patterns:
            if pattern.total_failures > 5:
                recommendations.append(
                    f"⚠️ {pattern.metric_name} 개선 필요: {pattern.total_failures}개 실패 사례 발견"
                )
        
        # 일관성 관련 권장사항
        most_consistent = min(model_performances, key=lambda x: x.correctness_std)
        recommendations.append(
            f"🎯 가장 일관성 있는 모델: {most_consistent.model_name} (표준편차: {most_consistent.correctness_std:.3f})"
        )
        
        return recommendations
    
    def _generate_markdown_report(self, analysis_data: Dict[str, Any]) -> str:
        """통합 마크다운 보고서 생성"""
        # 실제 구현은 매우 길므로 주요 구조만 표시
        report_lines = [
            "# DeepEval 분석 보고서",
            "",
            f"**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**분석 대상 모델**: {', '.join(analysis_data['models_analyzed'])}",
            f"**총 테스트 케이스**: {analysis_data['data_summary']['total_test_cases']}개",
            "",
            "## 📋 핵심 결과 요약",
            ""
        ]
        
        # 권장사항 추가
        for recommendation in analysis_data['recommendations']:
            report_lines.append(f"- {recommendation}")
        
        # 모델별 성능 비교 테이블
        report_lines.extend([
            "",
            "## 🎯 모델별 성능 비교",
            "",
            "| 모델명 | 종합점수 | 등급 | Correctness | Clarity | Actionability | JSON Correctness | 합격률 |",
            "|--------|----------|------|-------------|---------|---------------|------------------|--------|"
        ])
        
        for perf in analysis_data['model_performances']:
            report_lines.append(
                f"| {perf.model_name} | {perf.overall_score:.3f} | {perf.grade} | "
                f"{perf.correctness_mean:.3f} | {perf.clarity_mean:.3f} | "
                f"{perf.actionability_mean:.3f} | {perf.json_correctness_mean:.3f} | "
                f"{perf.pass_rate:.1%} |"
            )
        
        # 추가 섹션들...
        report_lines.extend([
            "",
            "## 📊 메트릭별 상세 분석",
            "## 🔍 실패 패턴 분석", 
            "## 📈 통계 분석 결과",
            "## 💡 권장사항 및 결론"
        ])
        
        return "\n".join(report_lines)
    
    def _prepare_json_data(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """JSON 데이터 준비"""
        # ModelPerformance와 FailurePattern 객체를 딕셔너리로 변환
        return {
            "analysis_metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "models_analyzed": analysis_data["models_analyzed"]
            },
            "model_performances": [
                {
                    "model_name": perf.model_name,
                    "total_test_cases": perf.total_test_cases,
                    "overall_score": perf.overall_score,
                    "grade": perf.grade,
                    # 기타 메트릭들...
                }
                for perf in analysis_data["model_performances"]
            ],
            "failure_patterns": [
                {
                    "metric_name": pattern.metric_name,
                    "total_failures": pattern.total_failures,
                    # 기타 실패 유형들...
                }
                for pattern in analysis_data["failure_patterns"]
            ],
            "statistical_analysis": analysis_data["statistical_analysis"],
            "recommendations": analysis_data["recommendations"],
            "data_summary": analysis_data["data_summary"]
        }
    
    def _should_generate_dashboard(self, analysis_data: Dict[str, Any]) -> bool:
        """인터랙티브 대시보드 생성 여부 결정"""
        # 복잡한 분석이나 많은 모델이 있을 때만 생성
        return len(analysis_data["models_analyzed"]) >= 3
    
    def _generate_interactive_dashboard(self, analysis_data: Dict[str, Any], output_dir: Path) -> Optional[str]:
        """선택적 인터랙티브 대시보드 생성"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            # Plotly를 사용한 대시보드 생성
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('모델별 종합 성능', '메트릭별 성능 비교', '실패 패턴 분석', '성능 분포')
            )
            
            # 차트 구성...
            # HTML 파일로 저장
            dashboard_path = output_dir / "interactive_dashboard.html"
            fig.write_html(dashboard_path)
            
            return str(dashboard_path)
            
        except ImportError:
            logger.warning("plotly가 설치되지 않았습니다. 인터랙티브 대시보드를 생성할 수 없습니다.")
            return None
        except Exception as e:
            logger.error(f"대시보드 생성 실패: {e}")
            return None

### 7.2 분석 결과 파일 구조

`analyze_session()` 메서드를 실행하면 다음과 같은 간소화된 구조로 분석 결과 파일들이 생성됩니다:

#### 7.2.1 기본 디렉토리 구조

```
~/Library/selvage-eval/analyze_results/{session_id}/
├── analysis_report.md          # 통합 마크다운 보고서 (메인)
├── analysis_data.json          # 원시 데이터 (프로그래밍 활용용)
└── interactive_dashboard.html  # 선택적 통합 대시보드 (정말 필요시만)
```

#### 7.2.2 실제 예시

세션 ID가 `eval_20250708_004754_a1b2c3d4`인 경우:

```
~/Library/selvage-eval/analyze_results/eval_20250708_004754_a1b2c3d4/
├── analysis_report.md          # 20-50KB (통합 마크다운 보고서)
├── analysis_data.json          # 100-500KB (구조화된 데이터)
└── interactive_dashboard.html  # 200-800KB (복잡한 분석시에만 생성)
```

#### 7.2.3 각 파일별 상세 설명

**1. `analysis_report.md` - 통합 마크다운 보고서 (메인)**

모든 분석 결과를 하나의 마크다운 파일에 통합하여 제공합니다.

```markdown
# DeepEval 분석 보고서

**분석 시간**: 2025-07-08 10:30:45
**분석 대상 모델**: gemini-2.5-flash, gemini-2.5-pro, o3, o4-mini-high
**총 테스트 케이스**: 150개

## 📋 핵심 결과 요약
- 🏆 전체 최고 성능: gemini-2.5-pro (종합 점수: 0.892, 등급: A)
- 📊 Correctness 최고: gemini-2.5-pro (점수: 0.925)
- 🎯 가장 일관성 있는 모델: claude-sonnet-4 (표준편차: 0.045)
- ⚠️ actionability 개선 필요: 12개 실패 사례 발견

## 🎯 모델별 성능 비교
| 모델명 | 종합점수 | 등급 | Correctness | Clarity | Actionability | JSON Correctness | 합격률 |
|--------|----------|------|-------------|---------|---------------|------------------|--------|
| gemini-2.5-pro | 0.892 | A | 0.925 | 0.875 | 0.845 | 0.980 | 89.2% |
| claude-sonnet-4 | 0.834 | B | 0.815 | 0.890 | 0.820 | 0.960 | 83.1% |
| o3 | 0.798 | B | 0.780 | 0.835 | 0.775 | 0.945 | 79.5% |
| o4-mini-high | 0.745 | C | 0.720 | 0.785 | 0.730 | 0.935 | 72.8% |

## 📊 메트릭별 상세 분석
### gemini-2.5-pro
- **Correctness**: 0.925 ± 0.067
- **Clarity**: 0.875 ± 0.054
- **Actionability**: 0.845 ± 0.089
- **JSON Correctness**: 0.980 ± 0.023
- **총 테스트 케이스**: 150개

### [다른 모델들 상세 분석...]

## 🔍 실패 패턴 분석
### Correctness
- **총 실패 수**: 23개
- **이슈 누락**: 8개
- **잘못된 라인 번호**: 5개
- **부적절한 심각도**: 6개
- **기타**: 4개

### [다른 메트릭별 실패 패턴...]

## 📈 통계 분석 결과
- **평균 성능**: 0.817
- **성능 표준편차**: 0.061
- **최고 성능**: 0.892
- **최저 성능**: 0.745
- **기준점(0.7) 이상 모델**: 4개

## 💡 권장사항 및 결론
### 주요 권장사항
1. 🏆 전체 최고 성능: gemini-2.5-pro (종합 점수: 0.892, 등급: A)
2. 📊 Correctness 최고: gemini-2.5-pro (점수: 0.925)
3. 🎯 가장 일관성 있는 모델: claude-sonnet-4 (표준편차: 0.045)

### 개선 방향
- 실패 패턴이 많은 메트릭에 대한 프롬프트 개선
- 일관성이 낮은 모델의 안정성 향상 방안 검토
- 최고 성능 모델의 장점을 다른 모델에 적용
```

**2. `analysis_data.json` - 구조화된 원시 데이터 (프로그래밍 활용용)**

프로그래밍 방식으로 활용할 수 있는 구조화된 JSON 데이터를 제공합니다.

```json
{
  "analysis_metadata": {
    "analysis_timestamp": "2025-07-08T10:30:45.123456",
    "models_analyzed": ["gemini-2.5-flash", "gemini-2.5-pro", "o3", "o4-mini-high"]
  },
  "model_performances": [
    {
      "model_name": "gemini-2.5-pro",
      "total_test_cases": 150,
      "correctness_mean": 0.925,
      "correctness_std": 0.067,
      "clarity_mean": 0.875,
      "clarity_std": 0.054,
      "actionability_mean": 0.845,
      "actionability_std": 0.089,
      "json_correctness_mean": 0.980,
      "json_correctness_std": 0.023,
      "overall_score": 0.892,
      "pass_rate": 0.892,
      "grade": "A"
    }
  ],
  "failure_patterns": [
    {
      "metric_name": "correctness",
      "total_failures": 23,
      "missing_issues": 8,
      "incorrect_line_numbers": 5,
      "inappropriate_severity": 6,
      "unclear_descriptions": 2,
      "non_actionable_suggestions": 1,
      "json_format_errors": 1,
      "other": 0
    }
  ],
  "statistical_analysis": {
    "mean_performance": 0.817,
    "std_performance": 0.061,
    "min_performance": 0.745,
    "max_performance": 0.892,
    "performance_range": 0.147,
    "models_above_threshold": 4
  },
  "recommendations": [
    "🏆 전체 최고 성능: gemini-2.5-pro (종합 점수: 0.892, 등급: A)",
    "📊 Correctness 최고: gemini-2.5-pro (점수: 0.925)"
  ]
}
```

**3. `interactive_dashboard.html` - 선택적 통합 대시보드**

3개 이상의 모델이 분석될 때만 생성되는 인터랙티브 대시보드입니다.

- **Plotly 기반 통합 대시보드**
- **모델별 종합 성능, 메트릭별 레이더 차트, 실패 패턴, 성능 분포를 한 화면에서 확인**
- **브라우저에서 직접 열어서 확인 가능**
- **확대/축소, 필터링, 데이터 포인트 상세 정보 제공**

#### 7.2.4 사용자 가이드

**1. 빠른 확인 순서**
1. `analysis_report.md` - 통합 마크다운 보고서에서 모든 분석 결과 확인
2. `interactive_dashboard.html` - 복잡한 분석이 필요한 경우에만 인터랙티브 대시보드 참조

**2. 상세 분석**
- `analysis_data.json` - 프로그래밍 방식으로 데이터 활용
- `analysis_report.md` - 텍스트 기반 상세 분석 및 패턴 확인

**3. 보고서 활용**
- **경영진 보고**: `analysis_report.md`의 핵심 결과 요약 섹션 활용
- **기술팀 분석**: `analysis_report.md`의 메트릭별 상세 분석 및 실패 패턴 분석 활용
- **자동화 처리**: `analysis_data.json` 활용

**4. 장점**
- **통합성**: 한 파일(`analysis_report.md`)에서 모든 정보 확인 가능
- **접근성**: 텍스트 기반으로 검색/복사/공유 용이
- **효율성**: 로딩 시간 단축, 프린트 친화적
- **선택적 시각화**: 정말 필요시에만 인터랙티브 대시보드 참조

# 사용 예시

```python
from selvage_eval.analysis import DeepEvalAnalysisEngine

if __name__ == "__main__":
    # 분석 엔진 초기화 (기본 출력 디렉토리: ~/Library/selvage-eval/analyze_results)
    engine = DeepEvalAnalysisEngine()
    
    # 세션 분석 실행 (출력 디렉토리 자동 생성)
    session_path = "~/Library/selvage-eval/deepeval_results/eval_20250708_004754_a1b2c3d4"
    results = engine.analyze_session(session_path)
    # 결과는 ~/Library/selvage-eval/analyze_results/eval_20250708_004754_a1b2c3d4/ 에 저장됨
    
    # 커스텀 출력 디렉토리 지정도 가능
    # results = engine.analyze_session(session_path, output_dir="/custom/path")
    
    print("분석 완료!")
    print(f"분석된 모델: {results['analysis_metadata']['models_analyzed']}")
    print(f"생성된 파일:")
    print(f"  - 마크다운 보고서: {results['files_generated']['markdown_report']}")
    print(f"  - JSON 데이터: {results['files_generated']['json_data']}")
    if results['files_generated']['interactive_dashboard']:
        print(f"  - 인터랙티브 대시보드: {results['files_generated']['interactive_dashboard']}")
```