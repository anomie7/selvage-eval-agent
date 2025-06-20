# Selvage 평가 에이전트 프로젝트

## 개요

이 프로젝트는 AI 기반 코드 리뷰 도구인 **Selvage**의 성능과 품질을 자동화된 방식으로 평가하는 에이전트입니다.

## 프로젝트 목적

### 평가 파이프라인
1. **Meaningful Commit 선별**: Git 통계와 커밋 메시지 분석으로 의미있는 커밋 식별
2. **Selvage 리뷰 실행**: 선별된 커밋에 대해 모델별 코드 리뷰 수행
3. **DeepEval 변환**: 리뷰 결과를 DeepEval 테스트 데이터로 변환
4. **LLM 평가**: DeepEval을 통한 정량적 성능 측정
5. **결과 분석**: 모델별, 프롬프트 버전별 성능 비교

### 평가 목표
- **모델별 성능 비교**: gemini-2.5-flash, claude-sonnet-4 등
- **프롬프트 버전 최적화**: 다양한 프롬프트 전략 효과성 측정
- **비용 효율성 분석**: 토큰 사용량 대비 리뷰 품질 평가

## Selvage 기본 정보

- **프로젝트 위치**: `/Users/demin_coder/Dev/selvage`
- **바이너리**: `/Users/demin_coder/.local/bin/selvage`
- **버전**: 0.1.2
- **언어**: Python 3.10+

### 핵심 명령어
```bash
# 특정 커밋 리뷰 (평가용 핵심 명령어)
selvage review --target-commit <commit_id> --model <model_name>

# 모델별 리뷰 비교
selvage review --target-commit abc1234 --model gemini-2.5-flash
selvage review --target-commit abc1234 --model claude-sonnet-4
```

## 예상 작업 플로우

### 1. Meaningful Commit 추출
```python
# Git 통계 기반 필터링
commits = extract_meaningful_commits(
    repo_path="/path/to/target/repo",
    filters={
        "min_changed_files": 2,
        "min_changed_lines": 50,
        "keywords": ["fix", "feature", "refactor"]
    }
)
```

### 2. Selvage 리뷰 실행
```python
# 모델별 리뷰 수행
for commit in commits:
    for model in ["gemini-2.5-flash", "claude-sonnet-4"]:
        result = run_selvage_review(commit.id, model)
        save_review_result(result, commit.id, model)
```

### 3. DeepEval 변환 및 평가
```python
# 리뷰 결과를 DeepEval 형식으로 변환
test_cases = convert_to_deepeval_format(review_results)

# LLM 평가 실행
evaluation_results = run_deepeval_assessment(test_cases)
```

### 4. 결과 분석
```python
# 모델별 성능 비교 분석
analysis = analyze_evaluation_results(
    evaluation_results,
    metrics=["accuracy", "completeness", "usefulness"]
)
```

## 개발 환경

### 사전 요구사항
- Python 3.10+
- Git 설치
- Selvage 설치 및 설정 완료
- AI API 키 (OpenAI, Anthropic, Google)

### 환경 설정
```bash
# API 키 설정
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key" 
export GEMINI_API_KEY="your-key"

# Selvage 설치 확인
selvage --version  # 0.1.2 확인
```

## 참조

- **Selvage GitHub**: https://github.com/anomie7/selvage
- **Selvage 설치 위치**: `/Users/demin_coder/Dev/selvage`
- **DeepEval 문서**: https://docs.deepeval.ai/