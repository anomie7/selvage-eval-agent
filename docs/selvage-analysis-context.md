# Selvage 구조 분석 컨텍스트

## Selvage 프로젝트 정보

**Selvage**는 AI 기반 코드 리뷰 도구로, Git diff를 분석하여 코드 품질 향상과 버그 탐지를 지원합니다.

### 기본 정보
- **위치**: `/Users/demin_coder/Dev/selvage`
- **바이너리**: `/Users/demin_coder/.local/bin/selvage`
- **가상환경**: `/Users/demin_coder/Dev/selvage/venv/bin/selvage`
- **언어**: Python 3.10+
- **버전**: 0.1.2
- **라이선스**: MIT

## 평가 대상 핵심 아키텍처

### 1. CLI 인터페이스 (`selvage/cli.py`)

**평가 관련 중요 명령어:**
```bash
selvage review [OPTIONS]         # 코드 리뷰 실행 (핵심 평가 대상)
selvage config [COMMAND]         # 설정 관리  
selvage view [OPTIONS]           # UI 실행
selvage --version               # 버전 정보
```

**평가 대상 주요 옵션:**
- `--repo-path <path>`: Git 저장소 경로 지정
- `--staged`: 스테이징된 변경사항만 리뷰
- `--target-commit <commit>`: 특정 커밋부터 HEAD까지 (meaningful commit 평가용)
- `--target-branch <branch>`: 브랜치 간 비교
- `--model <model>`: AI 모델 선택 (모델별 성능 비교용)
- `--diff-only`: 변경 부분만 분석 (토큰 효율성 평가용)
- `--open-ui`: 리뷰 후 UI 자동 실행

### 2. LLM Gateway 시스템 (`selvage/src/llm_gateway/`)

**팩토리 패턴 기반 모델 게이트웨이:**
```python
class GatewayFactory:
    @staticmethod
    def create(model: str) -> BaseGateway:
        # 모델별 게이트웨이 생성
```

**평가 대상 모델:**
- **OpenAI**: gpt-4o, gpt-4.1, o4-mini 시리즈
- **Anthropic**: claude-sonnet-4, claude-sonnet-4-thinking  
- **Google**: gemini-2.5-pro, gemini-2.5-flash

**게이트웨이 구조:**
```python
class BaseGateway(ABC):
    @abstractmethod
    def review_code(self, prompt: str) -> ReviewResponse:
        """코드 리뷰 요청 (평가 핵심 메서드)"""
        
    @abstractmethod
    def estimate_cost(self, prompt: str) -> EstimatedCost:
        """비용 추정 (성능 평가용)"""
```

### 3. Git 통합 시스템 (`selvage/src/utils/git_utils.py`)

**Meaningful Commit 선별을 위한 Git Diff 모드:**
```python
class GitDiffUtility:
    def get_commit_diff(self, commit_id: str) -> str:
        """특정 커밋과 HEAD 간 diff (평가용 핵심 기능)"""
    
    def get_branch_diff(self, branch_name: str) -> str:
        """브랜치 간 diff"""
        
    def get_staged_diff(self) -> str:
        """스테이징된 변경사항"""
    
    def get_unstaged_diff(self) -> str:
        """워킹 디렉토리 변경사항"""
```

**Git 명령어 실행 방식:**
- **Staged**: `git diff --cached`
- **Unstaged**: `git diff`
- **Commit to Commit**: `git diff <commit1> <commit2>`
- **Branch to Branch**: `git diff <branch1> <branch2>`

### 4. 설정 관리 (`selvage/src/config.py`)

**API 키 관리 (모델별 평가를 위한 필수 요소):**
- **환경변수 우선**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`
- **플랫폼별 설정 파일**: 
  - macOS: `~/Library/Application Support/selvage/config.ini`
  - Linux: `~/.config/selvage/config.ini`

**설정 우선순위:**
1. 환경변수 (최우선)
2. 플랫폼별 설정 파일

### 5. 데이터 모델 (`selvage/src/models/`)

**리뷰 결과 구조 (DeepEval 변환 대상):**
```python
# review_result.py
class ReviewResult:
    """리뷰 결과 데이터 구조 (평가 결과 분석 대상)"""

# model_choice.py  
class ModelChoice(Enum):
    """지원 AI 모델 목록 (모델별 평가 기준)"""

# model_provider.py
class ModelProvider(Enum):
    """AI 제공업체 (모델별 성능 비교용)"""

# review_status.py
class ReviewStatus(Enum):
    """리뷰 상태 관리"""
```

### 6. 프롬프트 시스템 (`selvage/src/utils/prompts/`)

**프롬프트 생성 전략 (버전별 평가 대상):**
```python
class PromptGenerator:
    def generate_review_prompt(
        self, 
        diff_content: str, 
        file_content: Optional[str] = None
    ) -> str:
        """리뷰 프롬프트 생성 (프롬프트 버전별 비교 평가용)"""
```

**프롬프트 구성:**
- **시스템 프롬프트**: AI 역할 정의
- **사용자 프롬프트**: 구체적 리뷰 요청
- **컨텍스트 프롬프트**: 파일 내용 포함 여부

**프롬프트 모드:**
- **diff-only**: 변경사항만 분석 (토큰 절약)
- **with-context**: 전체 파일 컨텍스트 포함 (정확도 향상)

## 평가 워크플로우와 연관된 아키텍처

### Selvage 실행 플로우 (평가 대상 프로세스)

```
1. CLI 명령 파싱 → 2. Git Diff 추출 → 3. 파일 필터링 
→ 4. 프롬프트 생성 → 5. AI 모델 호출 → 6. 응답 파싱 
→ 7. 결과 포맷팅 → 8. 출력/저장
```

**각 단계별 평가 요소:**
1. **CLI 파싱**: 옵션 처리 정확성
2. **Git Diff**: Meaningful commit 처리 능력
3. **파일 필터링**: 지원 파일 형식, 크기 제한
4. **프롬프트 생성**: 버전별 프롬프트 효과성
5. **AI 모델 호출**: 모델별 성능 비교
6. **응답 파싱**: 구조화된 결과 생성
7. **결과 포맷팅**: DeepEval 변환 가능성
8. **출력/저장**: 결과 데이터 품질

### 성능 평가 관련 구조

#### 토큰 관리 (`selvage/src/utils/token/`)
```python
def estimate_tokens(text: str, model: str) -> int:
    """토큰 수 추정 (비용 효율성 평가용)"""

def calculate_cost(
    input_tokens: int, 
    output_tokens: int, 
    model: str
) -> float:
    """API 비용 계산 (성능 메트릭)"""
```

#### 응답 시간 측정 지점
- **Git Diff 추출 시간**
- **프롬프트 생성 시간**  
- **AI API 호출 시간**
- **결과 파싱 시간**

## 평가 에이전트 구현을 위한 핵심 인터페이스

### Selvage 실행 방법
```python
import subprocess

# 기본 실행
result = subprocess.run([
    "/Users/demin_coder/.local/bin/selvage", 
    "review", 
    "--target-commit", "abc1234",
    "--model", "gemini-2.5-flash"
], capture_output=True, text=True)

# 결과 파싱
if result.returncode == 0:
    review_output = result.stdout
else:
    error_output = result.stderr
```

### 모델별 평가 실행
```python
models = [
    "gemini-2.5-flash",
    "claude-sonnet-4", 
    "gpt-4o"
]

for model in models:
    result = run_selvage_review(commit_id, model)
    # DeepEval 변환 및 평가
```

### 결과 데이터 구조 예상
```json
{
    "review_id": "unique_id",
    "commit_id": "abc1234", 
    "model": "gemini-2.5-flash",
    "timestamp": "2024-01-01T00:00:00Z",
    "review_result": {
        "summary": "전반적인 코드 품질 평가",
        "issues": [...],
        "suggestions": [...]
    },
    "performance_metrics": {
        "response_time": 2.5,
        "input_tokens": 1500,
        "output_tokens": 800,
        "cost": 0.025
    }
}
```

## 의존성 (평가 환경 구성용)

```toml
dependencies = [
    "openai==1.68.2",            # OpenAI API
    "anthropic==0.49.0",         # Anthropic Claude API  
    "google-genai==1.13.0",      # Google Gemini API
    "streamlit==1.43.2",         # 웹 UI
    "click==8.1.8",              # CLI 프레임워크
    "pydantic==2.10.6",          # 데이터 검증
    "tiktoken==0.9.0",           # 토큰 계산
    "rich==13.7.1",              # 터미널 출력
    # 평가 관련 추가 필요
    "deepeval",                  # LLM 평가 프레임워크
    "pytest"                     # 테스트 실행
]
```

## 현재 프로젝트 목적과 연관성

이 프로젝트는 Selvage의 **종합적 평가**를 위한 자동화 에이전트 개발을 목적으로 합니다:

1. **Meaningful Commit 선별**: Git 통합 기능 활용
2. **모델별 성능 비교**:
3. **프롬프트 버전 평가**

## 개발 지침

### 코딩 스타일
- **Python 3.10+**, PEP 8 준수
- **타입 힌팅** 필수 적용
- **Google 스타일 독스트링**
- **한국어 주석 및 독스트링**

### 평가 대상 식별
- **핵심 기능**: CLI review 명령어, Git 통합, LLM Gateway
- **성능 메트릭**: 응답 시간, 토큰 사용량, 비용
- **품질 평가**: DeepEval 기반 리뷰 결과 분석
- **비교 기준**: 모델별, 프롬프트 버전별 성능 차이

### 테스트 전략
- 실제 selvage 바이너리 사용
- Meaningful commit 기반 테스트 데이터 고정
- API 키는 환경변수에서 로드
- DeepEval 호환 결과 형식 생성 및 llm eval
- llm eval 결과 리포트 발행


