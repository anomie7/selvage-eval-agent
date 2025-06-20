# CLAUDE.md

이 파일은 저장소에서 작업할 때 Claude Code (claude.ai/code)에게 가이드를 제공합니다.

## 프로젝트 개요

AI 기반 코드 리뷰 도구인 Selvage를 평가하는 Selvage 평가 에이전트 프로젝트입니다.

## 상세 문서

- [Selvage 구조 분석 및 평가 컨텍스트](docs/rules/selvage-analysis-context.mdc)
- [구현 가이드](docs/rules/selvage-eval-implementation-guide.md)

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