"""AI 기반 실패 분석 기능 테스트"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from selvage_eval.analysis.deepeval_analysis_engine import DeepEvalAnalysisEngine


@pytest.fixture
def mock_engine():
    """테스트용 분석 엔진 생성"""
    with patch('selvage_eval.analysis.deepeval_analysis_engine.os.getenv') as mock_getenv:
        mock_getenv.return_value = "test_api_key"
        
        with patch('selvage_eval.llm.gemini_client.GeminiClient') as mock_client_class:
            # Flash 클라이언트 모킹
            mock_flash_client = Mock()
            # Pro 클라이언트 모킹  
            mock_pro_client = Mock()
            
            # 클라이언트 생성 시 모델명에 따라 다른 인스턴스 반환
            def create_client(*args, **kwargs):
                if 'gemini-2.5-pro' in str(kwargs.get('model_name', '')):
                    return mock_pro_client
                return mock_flash_client
            
            mock_client_class.side_effect = create_client
            
            engine = DeepEvalAnalysisEngine()
            engine.gemini_client = mock_flash_client
            engine.gemini_pro_client = mock_pro_client
            
            return engine, mock_flash_client, mock_pro_client


def test_ai_failure_analysis_basic(mock_engine):
    """AI 실패 분석 기본 기능 테스트"""
    engine, mock_flash_client, mock_pro_client = mock_engine
    
    # 테스트 데이터
    failed_metrics = {
        'correctness': {
            'failure_count': 3,
            'translated_reasons': [
                '코드의 논리적 오류가 발견됨',
                '예상 출력과 실제 출력이 일치하지 않음',
                '함수 동작이 명세와 다름'
            ]
        },
        'clarity': {
            'failure_count': 2,
            'translated_reasons': [
                '코드 주석이 불충분함',
                '변수명이 명확하지 않음'
            ]
        }
    }
    
    # Gemini Pro 응답 모킹
    mock_analysis_response = """
### 1. 핵심 실패 패턴 요약
- 논리적 오류: 코드의 기본 동작 로직에 문제가 있는 경우
- 명세 불일치: 설계 명세와 구현 사이의 불일치 
- 가독성 부족: 코드 이해를 위한 정보 부족

### 2. 메트릭별 분류 및 특성 분석
정확성 메트릭에서 핵심 기능 오류가 집중되어 있으며, 명확성 메트릭에서는 문서화 및 네이밍 이슈가 주를 이룸.

### 3. 근본 원인 분석
시스템적 문제로 판단되며, 코드 리뷰 프로세스 강화가 필요함.

### 4. 구체적 개선 방안
- 단기: 자동화된 정적 분석 도구 도입
- 중기: 코드 리뷰 가이드라인 수립
- 장기: 개발자 교육 프로그램 운영

### 5. 리스크 및 모니터링 포인트
논리적 오류 발생률 지속 모니터링 필요
"""
    
    mock_pro_client.query.return_value = mock_analysis_response
    
    # AI 분석 실행
    result = engine._analyze_metric_failures_with_ai(failed_metrics)
    
    # 결과 검증
    assert result is not None
    assert 'analysis_content' in result
    assert 'analyzed_metrics' in result
    assert 'total_failures_analyzed' in result
    assert 'analysis_timestamp' in result
    
    assert result['total_failures_analyzed'] == 5  # 3 + 2
    assert 'correctness' in result['analyzed_metrics']
    assert 'clarity' in result['analyzed_metrics']
    assert '핵심 실패 패턴 요약' in result['analysis_content']
    
    # Gemini Pro 클라이언트 호출 검증
    mock_pro_client.query.assert_called_once()
    call_args = mock_pro_client.query.call_args
    assert 'messages' in call_args[1]
    assert 'system_instruction' in call_args[1]
    
    # 시스템 인스트럭션 검증
    system_instruction = call_args[1]['system_instruction']
    assert '10년 경력의 시니어 소프트웨어 엔지니어' in system_instruction
    assert '데이터 분석 전문가' in system_instruction
    assert '테크니컬 라이터' in system_instruction


def test_ai_failure_analysis_empty_metrics(mock_engine):
    """빈 메트릭 데이터에 대한 AI 분석 테스트"""
    engine, _, _ = mock_engine
    
    # 빈 데이터로 테스트
    result = engine._analyze_metric_failures_with_ai({})
    assert result is None


def test_ai_failure_analysis_no_gemini_pro_client(mock_engine):
    """Gemini Pro 클라이언트가 없는 경우 테스트"""
    engine, _, _ = mock_engine
    engine.gemini_pro_client = None
    
    failed_metrics = {
        'correctness': {
            'failure_count': 1,
            'translated_reasons': ['테스트 실패']
        }
    }
    
    result = engine._analyze_metric_failures_with_ai(failed_metrics)
    assert result is None


def test_ai_failure_analysis_gemini_error(mock_engine):
    """Gemini API 오류 발생 시 테스트"""
    engine, _, mock_pro_client = mock_engine
    
    # API 오류 모킹
    mock_pro_client.query.side_effect = Exception("API Error")
    
    failed_metrics = {
        'correctness': {
            'failure_count': 1,
            'translated_reasons': ['테스트 실패']
        }
    }
    
    result = engine._analyze_metric_failures_with_ai(failed_metrics)
    assert result is None


def test_markdown_report_includes_ai_analysis(mock_engine):
    """마크다운 보고서에 AI 분석이 포함되는지 테스트"""
    engine, _, mock_pro_client = mock_engine
    
    # AI 분석 응답 모킹
    mock_analysis_response = "AI 분석 결과 내용"
    mock_pro_client.query.return_value = mock_analysis_response
    
    # 테스트 분석 결과 데이터 - 실제 구조에 맞게 수정
    analysis_results = {
        'model_failure_analysis': {
            'model1': {
                'total_tests': 5,
                'total_failures': 2,  # total_failures > 0이어야 상세 분석이 실행됨
                'failure_rate': 0.4,
                'failed_metrics': {
                    'correctness': {
                        'failure_count': 2,
                        'avg_confidence': 0.75,
                        'translated_reasons': ['오류1', '오류2']
                    }
                }
            }
        }
    }
    
    # 마크다운 보고서 생성
    report = engine._generate_markdown_report(analysis_results)
    
    # AI 분석 섹션이 포함되었는지 확인
    assert '🤖 AI 기반 실패 사유 분석' in report
    assert 'AI 분석 결과 내용' in report
    assert '분석 대상' in report
    assert '총 분석 실패 건수' in report


def test_prompt_engineering_quality(mock_engine):
    """프롬프트 엔지니어링 품질 검증"""
    engine, _, mock_pro_client = mock_engine
    
    failed_metrics = {
        'correctness': {
            'failure_count': 1,
            'translated_reasons': ['테스트 오류']
        }
    }
    
    mock_pro_client.query.return_value = "분석 결과"
    
    # AI 분석 실행
    engine._analyze_metric_failures_with_ai(failed_metrics)
    
    # 프롬프트 내용 검증
    call_args = mock_pro_client.query.call_args
    user_message = call_args[1]['messages'][0]['content']
    system_instruction = call_args[1]['system_instruction']
    
    # 개선된 프롬프트 요소들이 포함되었는지 확인
    assert '핵심 실패 패턴 요약' in user_message
    assert '메트릭별 분류 및 특성 분석' in user_message  
    assert '근본 원인 분석 및 실행 계획' in user_message
    assert '실행 가능한 개선 권고사항' in user_message
    assert '1주일 내 실행 가능한 단기 개선책' in user_message
    assert '1개월 내 완료 가능한 중기 개선책' in user_message
    assert '실제 데이터를 구체적으로 인용' in user_message
    
    # 페르소나 검증
    assert '10년 경력의 시니어 소프트웨어 엔지니어' in system_instruction
    assert '데이터 분석 전문가' in system_instruction
    assert '테크니컬 라이터' in system_instruction


def test_ai_failure_summary_generation(mock_engine):
    """AI 실패 분석 종합 요약 생성 테스트"""
    engine, _, mock_pro_client = mock_engine
    
    # 모델별 실패 분석 데이터 준비
    model_failure_analysis = {
        'model1': {
            'total_failures': 5,
            'failed_metrics': {
                'correctness': {
                    'failure_count': 3,
                    'translated_reasons': ['로직 오류 발생', '알고리즘 문제']
                },
                'clarity': {
                    'failure_count': 2,
                    'translated_reasons': ['주석 부족', '가독성 문제']
                }
            }
        },
        'model2': {
            'total_failures': 3,
            'failed_metrics': {
                'actionability': {
                    'failure_count': 2,
                    'translated_reasons': ['실행 방안 부족']
                },
                'json_correctness': {
                    'failure_count': 1,
                    'translated_reasons': ['JSON 형식 오류']
                }
            }
        }
    }
    
    # AI 분석 응답 모킹
    mock_analysis_response = """
    ### 1. 핵심 실패 패턴 요약
    - 로직 오류 (40%): 알고리즘 검증 부족
    - 문서화 부족 (30%): 주석 및 설명 미흡
    - 실행 가능성 (30%): 구체적 방안 제시 부족
    
    ### 2. 메트릭별 분류 및 특성 분석
    정확성 메트릭에서 로직 오류가 집중적으로 발생하고 있음
    """
    
    mock_pro_client.query.return_value = mock_analysis_response
    
    # 종합 요약 생성
    summary = engine._generate_ai_failure_summary(model_failure_analysis)
    
    # 결과 검증
    assert summary is not None
    assert '2개 모델, 총 8건의 실패 사례' in summary
    assert '주요 실패 사유:' in summary
    assert '메트릭별 핵심 이슈:' in summary
    assert '즉시 개선 권고사항:' in summary


def test_failure_patterns_extraction(mock_engine):
    """실패 패턴 추출 로직 테스트"""
    engine, _, _ = mock_engine
    
    # AI 분석 결과 데이터 준비
    ai_analyses = [
        {
            'model_name': 'model1',
            'analysis': {
                'analysis_content': '로직 오류가 발견되었습니다. 알고리즘 검증이 필요합니다.',
                'analyzed_metrics': ['correctness']
            }
        },
        {
            'model_name': 'model2', 
            'analysis': {
                'analysis_content': '주석이 부족하여 가독성이 떨어집니다. 문서화 개선이 필요합니다.',
                'analyzed_metrics': ['clarity']
            }
        }
    ]
    
    # 패턴 추출 실행
    patterns = engine._extract_failure_patterns(ai_analyses)
    
    # 결과 검증
    assert len(patterns) >= 2
    assert any('로직 오류' in pattern[0] for pattern in patterns)
    assert any('문서화 부족' in pattern[0] for pattern in patterns)
    assert all(isinstance(pattern[1], int) for pattern in patterns)  # 비율이 정수인지 확인


def test_metric_issues_extraction(mock_engine):
    """메트릭별 이슈 추출 로직 테스트"""
    engine, _, _ = mock_engine
    
    # AI 분석 결과 데이터 준비
    ai_analyses = [
        {
            'analysis': {
                'analysis_content': '정확한 로직 검증이 필요합니다.',
                'analyzed_metrics': ['correctness']
            }
        },
        {
            'analysis': {
                'analysis_content': '명확한 주석이 부족합니다.',
                'analyzed_metrics': ['clarity']
            }
        }
    ]
    
    # 메트릭 이슈 추출 실행
    metric_issues = engine._extract_metric_issues(ai_analyses)
    
    # 결과 검증
    assert isinstance(metric_issues, dict)
    assert '정확성' in metric_issues or '명확성' in metric_issues
    for metric, issue in metric_issues.items():
        assert len(issue) > 0
        assert '필요' in issue or '요구' in issue


def test_improvement_suggestions_extraction(mock_engine):
    """개선 권고사항 추출 로직 테스트"""
    engine, _, _ = mock_engine
    
    # AI 분석 결과 데이터 준비
    ai_analyses = [
        {
            'analysis': {
                'analysis_content': '단기적으로 즉시 개선이 필요합니다. 자동화 도구 활용을 권장합니다.',
                'analyzed_metrics': ['correctness']
            }
        }
    ]
    
    # 개선 권고사항 추출 실행
    suggestions = engine._extract_improvement_suggestions(ai_analyses)
    
    # 결과 검증
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0
    assert any('단기' in suggestion or '즉시' in suggestion for suggestion in suggestions)


def test_markdown_report_includes_ai_summary(mock_engine):
    """마크다운 보고서에 AI 종합 요약이 포함되는지 테스트"""
    engine, _, mock_pro_client = mock_engine
    
    # AI 분석 응답 모킹
    mock_analysis_response = "종합 분석 결과 내용"
    mock_pro_client.query.return_value = mock_analysis_response
    
    # 테스트 분석 결과 데이터
    analysis_results = {
        'model_failure_analysis': {
            'model1': {
                'total_tests': 5,
                'total_failures': 2,
                'failure_rate': 0.4,
                'failed_metrics': {
                    'correctness': {
                        'failure_count': 2,
                        'avg_confidence': 0.75,
                        'translated_reasons': ['오류1', '오류2']
                    }
                }
            }
        }
    }
    
    # 마크다운 보고서 생성
    report = engine._generate_markdown_report(analysis_results)
    
    # AI 종합 분석 섹션이 포함되었는지 확인
    assert 'AI 기반 종합 실패 분석' in report
    assert '분석 대상' in report
    assert '메트릭별 핵심 이슈' in report or '즉시 개선 권고사항' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])