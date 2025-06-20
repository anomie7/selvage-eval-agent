# CLAUDE.md

이 파일은 저장소에서 작업할 때 Claude Code (claude.ai/code)에게 가이드를 제공합니다.

## 프로젝트 개요

AI 기반 코드 리뷰 도구인 Selvage의 종합적인 자동화 평가 시스템을 구현하는 Selvage 평가 에이전트 프로젝트입니다. 5단계 평가 파이프라인을 따릅니다: meaningful commit 선별, Selvage 리뷰 실행, DeepEval 변환, LLM 평가, 결과 분석.

## 개발 환경

### 사전 요구사항
- Python 3.10+ (타입 힌팅 지원)
- Selvage 바이너리: `/Users/demin_coder/.local/bin/selvage` (버전 0.1.2)
- Selvage 프로젝트 소스: `/Users/demin_coder/Dev/selvage`

### 필수 API 키
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key" 
export GEMINI_API_KEY="your-key"
```

### 핵심 의존성
- `deepeval` - LLM 평가 프레임워크
- `pytest` - 테스트 프레임워크
- subprocess 실행 및 데이터 처리를 위한 표준 라이브러리

## 코딩 스타일 요구사항

- **Python 3.10+** PEP 8 준수
- **타입 힌팅** 모든 함수에 필수
- **Google 스타일 독스트링** 한국어 주석
- 파일 패턴: `**/*.py`, `**/*.md`, `**/*.json`, `**/*.yaml`, `**/*.toml`

## Selvage 통합

### 핵심 명령어 구조
```python
# 주요 평가 명령어
subprocess.run([
    "/Users/demin_coder/.local/bin/selvage", 
    "review", 
    "--target-commit", "<commit_id>",
    "--model", "<model_name>"
], capture_output=True, text=True)
```

### 평가 대상 모델
- **OpenAI**: gpt-4o, gpt-4.1, o4-mini 시리즈
- **Anthropic**: claude-sonnet-4, claude-sonnet-4-thinking  
- **Google**: gemini-2.5-pro, gemini-2.5-flash

### 평가용 주요 Selvage 옵션
- `--target-commit <commit>`: Meaningful commit 평가
- `--model <model>`: 모델 성능 비교
- `--diff-only`: 토큰 효율성 평가
- `--repo-path <path>`: 대상 저장소 지정

## 아키텍처

### 1. Meaningful Commit 선별
적절한 필터링 기준으로 의미있는 커밋을 식별하는 Git 통계 및 커밋 메시지 분석:
- `min_changed_files`: 2-10개 파일
- `min_changed_lines`: 50+ 라인  
- 키워드: "fix", "feature", "refactor"

### 2. Selvage 리뷰 실행
성능 메트릭 수집과 함께 여러 모델에서 자동화된 실행:
- 응답 시간 측정
- 토큰 사용량 추적
- 비용 계산
- 오류 처리 및 재시도 로직

### 3. DeepEval 변환
Selvage 리뷰 결과를 DeepEval 테스트 형식으로 변환:
```python
# 예상 결과 구조
{
    "review_id": "unique_id",
    "commit_id": "abc1234", 
    "model": "gemini-2.5-flash",
    "review_result": {...},
    "performance_metrics": {
        "response_time": 2.5,
        "input_tokens": 1500,
        "output_tokens": 800,
        "cost": 0.025
    }
}
```

### 4. LLM 평가
다음 메트릭으로 DeepEval 기반 정량적 평가:
- 정확도
- 완성도  
- 유용성
- 비용 효율성

### 5. 결과 분석
다음 항목에 대한 비교 분석:
- 모델 성능
- 프롬프트 버전 효과성
- 토큰 사용량 대비 품질 트레이드오프

## 개발 워크플로우

### 테스트 전략
- 현실적인 평가를 위해 실제 Selvage 바이너리 사용
- 실제 저장소의 meaningful commit으로 테스트
- 환경변수에서 API 키 로드
- DeepEval 호환 결과 형식 생성

### 성능 측정 지점
- Git diff 추출 시간
- 프롬프트 생성 시간
- AI API 호출 시간
- 결과 파싱 시간

### 오류 처리
- Selvage 실행 실패
- API 속도 제한
- 모델 가용성 문제
- 잘못된 응답 처리

## 주요 구현 참고사항

- 모든 Selvage 상호작용은 설치된 바이너리에 대한 subprocess 호출을 사용해야 함
- 비교 분석을 위해 결과를 구조화된 형식으로 저장
- 가능한 경우 동시 모델 평가 구현
- 평가 파이프라인 문제 디버깅을 위한 상세 로깅 유지
- 평가 실행 후 적절한 리소스 정리 보장