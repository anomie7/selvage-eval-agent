# Selvage 평가 에이전트 - 아키텍처 및 설정

## 프로젝트 개요
AI 기반 코드 리뷰 도구인 Selvage를 평가하는 자동화 에이전트입니다. 4단계 워크플로우를 통해 모델별 성능과 프롬프트 버전 효과성을 정량적으로 측정합니다.

## 에이전트 아키텍처

### 설계 원칙
- **모듈화**: 각 단계를 독립적인 모듈로 구현
- **재현성**: JSON 기반 데이터 저장으로 테스트 재현 가능
- **확장성**: 새로운 모델 및 평가 지표 추가 용이
- **견고성**: 에러 처리 및 재시도 로직 내장

### 핵심 구현 요구사항
- **Python 3.10+** (타입 힌팅 필수)
- **Google 스타일 독스트링** (한국어 주석)
- **PEP 8 준수**
- **비동기 처리** (다중 모델 병렬 평가)

## Single Agent 아키텍처 패러다임

### ReAct (Reasoning + Acting) 패턴
Selvage 평가 에이전트는 단일 에이전트가 ReAct 패턴으로 4단계 워크플로우를 순차 실행합니다.

```python
class SelvageEvaluationAgent:
    """
    단일 에이전트로 전체 평가 프로세스를 관리하는 Selvage 평가 에이전트
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.tools = self._initialize_tools()
        self.working_memory = WorkingMemory()
        self.session_state = SessionState()
        self.current_phase = None
    
    async def execute_full_evaluation(self) -> EvaluationReport:
        """
        전체 4단계 평가 프로세스 실행
        """
        session_id = self._generate_session_id()
        
        try:
            # Phase 1: Commit Collection - 의미있는 커밋 수집
            commits = await self._execute_phase1_commit_collection()
            
            # Phase 2: Review Execution - Selvage 리뷰 실행
            reviews = await self._execute_phase2_review_execution(commits)
            
            # Phase 3: DeepEval Conversion - 평가 형식 변환 및 실행
            evaluations = await self._execute_phase3_deepeval_conversion(reviews)
            
            # Phase 4: Analysis - 결과 분석 및 인사이트 도출
            analysis = await self._execute_phase4_analysis(evaluations)
            
            return EvaluationReport(
                session_id=session_id,
                commits=commits,
                reviews=reviews,
                evaluations=evaluations,
                analysis=analysis
            )
            
        except Exception as e:
            await self._handle_evaluation_error(e)
            raise
    
    async def _execute_phase1_commit_collection(self) -> List[CommitInfo]:
        """Phase 1: 프롬프트 기반 커밋 수집 및 배점"""
        self.current_phase = "commit_collection"
        
        # 프롬프트로 정의된 전략에 따라 도구 사용
        commits = []
        for repo in self.config.target_repositories:
            # git_log 도구 사용
            raw_commits = await self.tools["git_log"].execute(
                repo_path=repo.path,
                **self.config.commit_filters
            )
            
            # commit_scoring 도구로 배점 계산
            scored_commits = []
            for commit in raw_commits.data:
                score_result = await self.tools["commit_scoring"].execute(
                    commit_hash=commit["hash"],
                    repo_path=repo.path
                )
                if score_result.success and score_result.data["score"].total >= 60:
                    scored_commits.append(score_result.data)
            
            # 상위 점수 커밋 선별
            commits.extend(
                sorted(scored_commits, key=lambda x: x["score"].total, reverse=True)
                [:self.config.commits_per_repo]
            )
        
        return commits
    
    async def _execute_phase2_review_execution(self, commits: List[CommitInfo]) -> List[ReviewResult]:
        """Phase 2: 프롬프트 기반 리뷰 실행"""
        self.current_phase = "review_execution"
        
        reviews = []
        for commit in commits:
            for model in self.config.review_models:
                # selvage_executor 도구 사용
                review_result = await self.tools["selvage_executor"].execute(
                    repo_path=commit["repo_path"],
                    commit_hash=commit["commit_hash"],
                    model=model
                )
                
                if review_result.success:
                    reviews.append({
                        "commit": commit,
                        "model": model,
                        "result": review_result.data
                    })
        
        return reviews
    
    async def _execute_phase3_deepeval_conversion(self, reviews: List[ReviewResult]) -> List[EvaluationResult]:
        """Phase 3: 프롬프트 기반 DeepEval 변환 및 평가"""
        self.current_phase = "deepeval_conversion"
        
        # review_log_scanner 도구로 로그 파일 스캔
        scan_result = await self.tools["review_log_scanner"].execute()
        
        evaluations = []
        for log_entry in scan_result.data:
            # deepeval_converter 도구로 형식 변환
            conversion_result = await self.tools["deepeval_converter"].execute(
                log_file=log_entry["file_path"],
                metadata=log_entry["metadata"]
            )
            
            if conversion_result.success:
                # metric_evaluator 도구로 평가 실행
                eval_result = await self.tools["metric_evaluator"].execute(
                    test_cases=conversion_result.data
                )
                evaluations.append(eval_result.data)
        
        return evaluations
    
    async def _execute_phase4_analysis(self, evaluations: List[EvaluationResult]) -> AnalysisReport:
        """Phase 4: 프롬프트 기반 통계 분석 및 인사이트 도출 (복잡한 추론 필요)"""
        self.current_phase = "analysis"
        
        # 이 단계만 진짜 에이전트 수준의 복잡한 추론 필요
        # statistical_analysis 도구로 기본 통계 계산
        stats_result = await self.tools["statistical_analysis"].execute(
            evaluation_results=evaluations,
            analysis_type="comprehensive"
        )
        
        # AI 추론을 통한 패턴 분석 및 인사이트 도출
        insights = await self._analyze_patterns_with_reasoning(
            stats_result.data,
            evaluations
        )
        
        # 실행 가능한 권장사항 생성
        recommendations = await self._generate_actionable_recommendations(
            insights,
            stats_result.data
        )
        
        return AnalysisReport(
            statistics=stats_result.data,
            insights=insights,
            recommendations=recommendations,
            executive_summary=await self._create_executive_summary(insights, recommendations)
        )
```

## Tool 정의 및 분류

### Tool 분류 및 Interface 정의

도구는 크게 **기본 유틸리티 도구**와 **Phase별 전용 도구**로 분류됩니다:

#### 기본 유틸리티 도구 (모든 Phase에서 공통 사용)
- `execute_terminal`: 터미널 명령어 실행 (git, selvage, 파일 작업 등)
- `read_file`: 파일 내용 읽기 (JSON, 로그, 설정 파일 등)
- `write_file`: 파일 쓰기 (결과 저장, 임시 파일 생성 등)
- `list_directory`: 디렉토리 내용 조회
- `file_exists`: 파일/디렉토리 존재 확인

#### Phase별 전용 도구
- **Phase 1**: `git_log`, `commit_scoring`
- **Phase 2**: `selvage_executor`
- **Phase 3**: `review_log_scanner`, `deepeval_converter`, `metric_evaluator`
- **Phase 4**: `statistical_analysis`

모든 도구는 단일 에이전트가 사용하는 유틸리티로서 표준화된 인터페이스를 구현합니다:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ToolResult:
    """도구 실행 결과"""
    success: bool
    data: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

class Tool(ABC):
    """
    모든 도구의 기본 인터페이스
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """도구 이름"""
        pass
    
    @property 
    @abstractmethod
    def description(self) -> str:
        """도구 설명"""
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """매개변수 스키마 (JSON Schema 형식)"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """도구 실행"""
        pass
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """매개변수 유효성 검증"""
        # JSON Schema 기반 검증 구현
        pass
```

### 기본 유틸리티 도구 구현

**ExecuteTerminalTool** - 터미널 명령어 실행
```python
class ExecuteTerminalTool(Tool):
    """터미널 명령어 실행 도구 (모든 Phase에서 사용)"""
    
    @property
    def name(self) -> str:
        return "execute_terminal"
    
    @property
    def description(self) -> str:
        return "터미널 명령어를 실행하고 결과를 반환합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string", 
                    "description": "실행할 터미널 명령어"
                },
                "cwd": {
                    "type": "string", 
                    "description": "명령어 실행 디렉토리 (선택사항)"
                },
                "timeout": {
                    "type": "integer", 
                    "description": "타임아웃 (초, 기본값: 60)"
                },
                "capture_output": {
                    "type": "boolean", 
                    "description": "출력 캡처 여부 (기본값: true)"
                }
            },
            "required": ["command"]
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        command = kwargs["command"]
        cwd = kwargs.get("cwd", None)
        timeout = kwargs.get("timeout", 60)
        capture_output = kwargs.get("capture_output", True)
        
        try:
            # 보안을 위한 명령어 검증
            if self._is_dangerous_command(command):
                return ToolResult(
                    success=False,
                    error_message=f"Dangerous command blocked: {command}"
                )
            
            # 명령어 실행
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
            
            return ToolResult(
                success=process.returncode == 0,
                data={
                    "returncode": process.returncode,
                    "stdout": stdout.decode() if stdout else "",
                    "stderr": stderr.decode() if stderr else "",
                    "command": command
                },
                error_message=stderr.decode() if process.returncode != 0 and stderr else None
            )
            
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error_message=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to execute command: {str(e)}"
            )
    
    def _is_dangerous_command(self, command: str) -> bool:
        """위험한 명령어 검증"""
        dangerous_commands = ["rm -rf", "format", "del /s", "shutdown", "reboot"]
        return any(danger in command.lower() for danger in dangerous_commands)
```

**ReadFileTool** - 파일 읽기
```python
class ReadFileTool(Tool):
    """파일 내용 읽기 도구 (모든 Phase에서 사용)"""
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "지정된 파일의 내용을 읽어서 반환합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string", 
                    "description": "읽을 파일의 경로"
                },
                "encoding": {
                    "type": "string", 
                    "description": "파일 인코딩 (기본값: utf-8)"
                },
                "max_size_mb": {
                    "type": "integer", 
                    "description": "최대 파일 크기 (MB, 기본값: 10)"
                },
                "as_json": {
                    "type": "boolean", 
                    "description": "JSON으로 파싱 여부 (기본값: false)"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        file_path = kwargs["file_path"]
        encoding = kwargs.get("encoding", "utf-8")
        max_size_mb = kwargs.get("max_size_mb", 10)
        as_json = kwargs.get("as_json", False)
        
        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                return ToolResult(
                    success=False,
                    error_message=f"File not found: {file_path}"
                )
            
            # 파일 크기 확인
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return ToolResult(
                    success=False,
                    error_message=f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB"
                )
            
            # 파일 읽기
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # JSON 파싱 (필요시)
            if as_json:
                try:
                    content = json.loads(content)
                except json.JSONDecodeError as e:
                    return ToolResult(
                        success=False,
                        error_message=f"Invalid JSON format: {str(e)}"
                    )
            
            return ToolResult(
                success=True,
                data={
                    "content": content,
                    "file_path": file_path,
                    "file_size_bytes": os.path.getsize(file_path),
                    "encoding": encoding
                }
            )
            
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                error_message=f"Unable to decode file with encoding: {encoding}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to read file: {str(e)}"
            )
```

**WriteFileTool** - 파일 쓰기
```python
class WriteFileTool(Tool):
    """파일 쓰기 도구 (결과 저장용)"""
    
    @property
    def name(self) -> str:
        return "write_file"
    
    async def execute(self, **kwargs) -> ToolResult:
        file_path = kwargs["file_path"]
        content = kwargs["content"]
        encoding = kwargs.get("encoding", "utf-8")
        create_dirs = kwargs.get("create_dirs", True)
        as_json = kwargs.get("as_json", False)
        
        try:
            # 디렉토리 생성 (필요시)
            if create_dirs:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # JSON 직렬화 (필요시)
            if as_json:
                content = json.dumps(content, indent=2, ensure_ascii=False)
            
            # 파일 쓰기
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                data={
                    "file_path": file_path,
                    "bytes_written": len(content.encode(encoding)),
                    "encoding": encoding
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to write file: {str(e)}"
            )
```

### Phase 1 Tools: Commit Collection

**GitLogTool** - Git 로그 조회
```python
class GitLogTool(Tool):
    """Git 커밋 로그를 조회하는 도구"""
    
    @property
    def name(self) -> str:
        return "git_log"
    
    @property
    def description(self) -> str:
        return "지정된 저장소에서 커밋 로그를 조회합니다"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "저장소 경로"},
                "since": {"type": "string", "description": "시작 날짜 (YYYY-MM-DD)"},
                "until": {"type": "string", "description": "종료 날짜 (YYYY-MM-DD)"},
                "grep": {"type": "string", "description": "커밋 메시지 필터"},
                "max_count": {"type": "integer", "description": "최대 커밋 수"}
            },
            "required": ["repo_path"]
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Git 로그 실행"""
        repo_path = kwargs["repo_path"]
        
        cmd = ["git", "log", "--oneline", "--no-merges"]
        
        if kwargs.get("since"):
            cmd.extend(["--since", kwargs["since"]])
        if kwargs.get("until"):
            cmd.extend(["--until", kwargs["until"]])
        if kwargs.get("grep"):
            cmd.extend(["--grep", kwargs["grep"]])
        if kwargs.get("max_count"):
            cmd.extend(["-n", str(kwargs["max_count"])])
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                commits = self._parse_git_log(stdout.decode())
                return ToolResult(success=True, data=commits)
            else:
                return ToolResult(
                    success=False, 
                    data=None,
                    error_message=stderr.decode()
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                data=None, 
                error_message=str(e)
            )
    
    def _parse_git_log(self, log_output: str) -> List[Dict[str, str]]:
        """Git 로그 출력 파싱"""
        commits = []
        for line in log_output.strip().split('\n'):
            if line:
                parts = line.split(' ', 1)
                commits.append({
                    "hash": parts[0],
                    "message": parts[1] if len(parts) > 1 else ""
                })
        return commits
```

**CommitScoringTool** - 커밋 배점 도구
```python
class CommitScoringTool(Tool):
    """커밋 배점 및 필터링 도구"""
    
    @property
    def name(self) -> str:
        return "commit_scoring"
        
    @property
    def description(self) -> str:
        return "커밋의 평가 가치를 배점하여 의미있는 커밋을 선별합니다"
    
    async def execute(self, **kwargs) -> ToolResult:
        commit_hash = kwargs["commit_hash"]
        repo_path = kwargs["repo_path"]
        scoring_config = kwargs.get("scoring_config", {})
        
        # 커밋 정보 수집
        commit_info = await self._get_commit_info(commit_hash, repo_path)
        
        # 배점 계산
        score = self._calculate_score(commit_info, scoring_config)
        
        return ToolResult(
            success=True,
            data={
                "commit_hash": commit_hash,
                "score": score,
                "breakdown": score.breakdown,
                "commit_info": commit_info
            }
        )
    
    def _calculate_score(self, commit_info: Dict, config: Dict) -> CommitScore:
        """커밋 배점 계산 (기존 로직 구현)"""
        # A. 파일 타입 감점 조정
        file_type_score = self._calculate_file_type_score(commit_info["files"])
        
        # B. 변경 규모 적정성
        scale_score = self._calculate_scale_score(commit_info["stats"])
        
        # C. 커밋 특성
        characteristic_score = self._calculate_characteristic_score(
            commit_info["message"], 
            commit_info["files"]
        )
        
        # D. 시간 가중치
        time_score = self._calculate_time_score(commit_info["date"])
        
        # E. 추가 조정사항
        adjustment_score = self._calculate_adjustments(commit_info)
        
        total_score = (
            file_type_score + scale_score + 
            characteristic_score + time_score + adjustment_score
        )
        
        return CommitScore(
            total=max(0, min(100, total_score)),
            breakdown={
                "file_type": file_type_score,
                "scale": scale_score, 
                "characteristic": characteristic_score,
                "time": time_score,
                "adjustment": adjustment_score
            }
        )
```

### Phase 2 Tools: Review Execution

**SelvageExecutorTool** - Selvage 리뷰 실행
```python
class SelvageExecutorTool(Tool):
    """Selvage 코드 리뷰 실행 도구"""
    
    @property
    def name(self) -> str:
        return "selvage_executor"
    
    async def execute(self, **kwargs) -> ToolResult:
        repo_path = kwargs["repo_path"]
        commit_hash = kwargs["commit_hash"]
        model = kwargs["model"]
        log_dir = kwargs.get("log_dir", "~/Library/selvage-eval-agent/review_logs")
        
        # 1. 커밋 체크아웃
        checkout_result = await self._checkout_commit(repo_path, commit_hash)
        if not checkout_result.success:
            return checkout_result
        
        try:
            # 2. 부모 커밋 ID 획득
            parent_hash = await self._get_parent_commit(repo_path)
            
            # 3. Selvage 실행
            review_result = await self._execute_selvage_review(
                repo_path=repo_path,
                target_commit=parent_hash,
                model=model,
                log_dir=log_dir
            )
            
            return review_result
            
        finally:
            # 4. HEAD로 복원
            await self._checkout_head(repo_path)
    
    async def _execute_selvage_review(self, **params) -> ToolResult:
        """Selvage 리뷰 실제 실행"""
        cmd = [
            "selvage", "review",
            "--target-commit", params["target_commit"],
            "--model", params["model"],
            "--log-dir", params["log_dir"]
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=params["repo_path"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=300  # 5분 타임아웃
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # 리뷰 로그 파일 위치 파악 및 반환
                log_file = await self._find_review_log_file(params["log_dir"])
                return ToolResult(
                    success=True,
                    data={"log_file": log_file, "stdout": stdout.decode()}
                )
            else:
                return ToolResult(
                    success=False,
                    error_message=f"Selvage execution failed: {stderr.decode()}"
                )
                
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error_message="Selvage execution timed out after 5 minutes"
            )
```

### Phase 3 Tools: DeepEval Conversion

**ReviewLogScannerTool** - 리뷰 로그 스캔
```python
class ReviewLogScannerTool(Tool):
    """리뷰 로그 파일 스캔 및 메타데이터 추출"""
    
    async def execute(self, **kwargs) -> ToolResult:
        base_path = kwargs.get("base_path", "~/Library/selvage-eval-agent/review_logs")
        
        review_logs = []
        base_path = Path(base_path).expanduser()
        
        try:
            # 디렉토리 구조 탐색: repo_name/commit_id/model_name/*.json
            for repo_dir in base_path.iterdir():
                if not repo_dir.is_dir():
                    continue
                    
                for commit_dir in repo_dir.iterdir():
                    if not commit_dir.is_dir():
                        continue
                        
                    for model_dir in commit_dir.iterdir():
                        if not model_dir.is_dir():
                            continue
                            
                        for log_file in model_dir.glob("*.json"):
                            metadata = await self._extract_log_metadata(log_file)
                            review_logs.append({
                                "repo_name": repo_dir.name,
                                "commit_id": commit_dir.name,
                                "model_name": model_dir.name,
                                "file_path": log_file,
                                "metadata": metadata
                            })
            
            return ToolResult(success=True, data=review_logs)
            
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to scan review logs: {str(e)}"
            )
```

### Phase 4 Tools: Analysis and Visualization

**StatisticalAnalysisTool** - 통계 분석
```python
class StatisticalAnalysisTool(Tool):
    """DeepEval 결과 통계 분석"""
    
    async def execute(self, **kwargs) -> ToolResult:
        evaluation_results = kwargs["evaluation_results"]
        analysis_type = kwargs.get("analysis_type", "comprehensive")
        
        try:
            if analysis_type == "comprehensive":
                analysis = await self._comprehensive_analysis(evaluation_results)
            elif analysis_type == "model_comparison":
                analysis = await self._model_comparison_analysis(evaluation_results)
            elif analysis_type == "failure_pattern":
                analysis = await self._failure_pattern_analysis(evaluation_results)
            
            return ToolResult(success=True, data=analysis)
            
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Statistical analysis failed: {str(e)}"
            )
    
    async def _comprehensive_analysis(self, results: List[Dict]) -> Dict[str, Any]:
        """종합 통계 분석"""
        metrics_stats = {}
        
        for metric in ["correctness", "clarity", "actionability", "json_correctness"]:
            scores = self._extract_metric_scores(results, metric)
            
            metrics_stats[metric] = {
                "count": len(scores),
                "mean": np.mean(scores),
                "median": np.median(scores), 
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "q25": np.percentile(scores, 25),
                "q75": np.percentile(scores, 75),
                "pass_rate": len([s for s in scores if s >= 0.7]) / len(scores)
            }
        
        return {
            "metrics_statistics": metrics_stats,
            "overall_performance": self._calculate_overall_performance(metrics_stats),
            "recommendations": self._generate_recommendations(metrics_stats)
        }
```

## 단일 에이전트 프롬프트 설계

### Master Agent Prompt
```python
SINGLE_AGENT_PROMPT = """
# ROLE
당신은 Selvage 코드 리뷰 도구를 평가하는 전문 AI 에이전트입니다.
단일 에이전트로서 4단계 워크플로우를 순차적으로 실행하여 체계적이고 정량적인 평가를 수행합니다.

# CAPABILITIES
- 다양한 도구를 사용하여 Git 저장소 분석, 코드 리뷰 실행, 결과 평가 수행
- 통계적 분석을 통한 모델 성능 비교 및 인사이트 도출
- 재현 가능한 평가 환경 구축 및 결과 문서화

# WORKFLOW PHASES
당신은 다음 4단계를 순차적으로 실행합니다:

1. **Phase 1 - Commit Collection**: 
   - 목적: meaningful한 커밋들을 자동 식별 및 배점
   - 사용 도구: git_log, commit_scoring
   - 결과: 평가 가치가 높은 커밋 리스트

2. **Phase 2 - Review Execution**: 
   - 목적: 선별된 커밋에 대해 다중 모델로 Selvage 리뷰 실행
   - 사용 도구: selvage_executor
   - 결과: 모델별 리뷰 결과 로그

3. **Phase 3 - DeepEval Conversion**: 
   - 목적: 리뷰 결과를 DeepEval 형식으로 변환 및 평가
   - 사용 도구: review_log_scanner, deepeval_converter, metric_evaluator
   - 결과: 정량화된 평가 메트릭

4. **Phase 4 - Analysis & Insights**: 
   - 목적: 통계 분석을 통한 actionable insights 도출 (복잡한 추론 필요)
   - 사용 도구: statistical_analysis + AI 추론
   - 결과: 실행 가능한 권장사항 및 인사이트

# PHASE EXECUTION STRATEGY
- Phase 1-3: 주로 도구 호출과 데이터 처리 중심
- Phase 4: AI 추론을 통한 패턴 분석 및 인사이트 도출
- 각 단계의 결과는 다음 단계의 입력으로 사용
- 실패 시 재시도 로직 내장

# DECISION MAKING PRINCIPLES
- **데이터 기반**: 모든 결정은 정량적 데이터에 근거
- **재현성**: 동일 조건에서 동일 결과 보장
- **효율성**: 적절한 도구 선택 및 캐싱 활용
- **신뢰성**: 에러 처리 및 복구 메커니즘 내장

# ERROR HANDLING
- 각 단계에서 실패 시 자동 재시도 (최대 3회)
- 부분 실패 시에도 가능한 결과 수집 및 분석
- 상세한 에러 로깅 및 디버깅 정보 제공

# OUTPUT FORMAT
모든 결과는 JSON 형식으로 구조화하여 제공하며, 
사람이 읽기 쉬운 요약과 함께 제공합니다.

당신의 목표는 Selvage의 성능을 정확하고 공정하게 평가하여 
실제 의사결정에 도움이 되는 인사이트를 제공하는 것입니다.
"""
```

### Phase-Specific Context (프롬프트에 포함될 단계별 컨텍스트)

단일 에이전트가 현재 실행 중인 Phase를 이해할 수 있도록 각 단계별 세부 컨텍스트를 제공합니다:

**Phase 1 Context: Commit Collection**
```python
PHASE1_CONTEXT = """
현재 단계: Phase 1 - Commit Collection

목적: 평가 가치가 높은 의미있는 커밋들을 식별하고 선별

전략:
1. 키워드 기반 1차 필터링 (fix, feature, refactor 포함 / typo, format 제외)
2. 통계 기반 2차 필터링 (파일 수 2-10개, 변경 라인 50+ 기준)
3. 배점 기반 최종 선별 (파일 타입, 변경 규모, 커밋 특성 종합 고려)

사용할 도구: git_log, commit_scoring
예상 결과: commits_per_repo 개수만큼 선별된 고품질 커밋 리스트

실행 단계:
1. 각 저장소별 git_log로 후보 커밋 수집
2. commit_scoring으로 평가 가치 배점
3. 상위 점수 커밋 선별
"""

PHASE2_CONTEXT = """
현재 단계: Phase 2 - Review Execution

목적: 선별된 커밋들에 대해 다중 모델로 Selvage 리뷰 실행

전략:
1. 안전한 커밋 체크아웃 (실행 후 HEAD 복원)
2. 모델별 순차 실행 (동시성 제한)
3. 체계적 결과 저장 (repo/commit/model 구조)

사용할 도구: selvage_executor
예상 결과: 모델별 리뷰 결과 로그 파일들

실행 단계:
1. Phase 1 결과에서 커밋 목록 로드
2. 각 커밋별로 모델별 리뷰 실행
3. 결과 검증 및 구조화된 저장
"""

PHASE3_CONTEXT = """
현재 단계: Phase 3 - DeepEval Conversion

목적: 리뷰 결과를 DeepEval 테스트 케이스로 변환 및 평가

전략:
1. 리뷰 로그 파일 전체 스캔
2. prompt/response 데이터 추출
3. DeepEval 형식 변환
4. 4개 메트릭으로 평가 실행

사용할 도구: review_log_scanner, deepeval_converter, metric_evaluator
평가 메트릭: Correctness, Clarity, Actionability, JsonCorrectness
예상 결과: 정량화된 평가 점수 데이터

실행 단계:
1. 저장된 리뷰 로그 스캔
2. 데이터 추출 및 형식 변환
3. DeepEval 평가 실행
"""

PHASE4_CONTEXT = """
현재 단계: Phase 4 - Analysis & Insights (복잡한 추론 단계)

목적: 평가 결과 종합 분석 및 actionable insights 도출

전략:
1. 통계적 분석으로 기본 패턴 파악
2. AI 추론을 통한 깊이 있는 패턴 분석
3. 실행 가능한 권장사항 생성
4. 의사결정 지원 인사이트 도출

사용할 도구: statistical_analysis + AI 추론 능력
분석 차원: 모델별 성능, 기술스택별 특화, 실패 패턴, 비용 효율성
예상 결과: Executive Summary, 상세 성능 매트릭스, 개선 권장사항

주의: 이 단계는 단순한 도구 호출이 아닌 복잡한 추론과 인사이트 도출이 필요
"""
```

## 단일 에이전트의 Tool 실행 전략

### Phase-Sequential Tool Execution
단일 에이전트가 각 Phase 내에서 도구들을 순차적으로 실행하는 전략:

```python
class SingleAgentToolExecutor:
    """단일 에이전트의 도구 실행 관리"""
    
    def __init__(self, agent: SelvageEvaluationAgent):
        self.agent = agent
        self.retry_count = 3
        self.timeout_seconds = 300
    
    async def execute_phase_tools(self, phase: str, tool_sequence: List[Dict]) -> List[ToolResult]:
        """Phase 내 도구들을 순차 실행"""
        results = []
        
        for tool_config in tool_sequence:
            tool_name = tool_config["name"]
            tool_params = tool_config["params"]
            
            # 재시도 로직 포함 도구 실행
            result = await self._execute_with_retry(
                tool_name=tool_name,
                params=tool_params,
                max_retries=self.retry_count
            )
            
            results.append(result)
            
            # 중요한 도구 실패 시 Phase 중단
            if not result.success and tool_config.get("critical", False):
                raise PhaseExecutionError(f"Critical tool {tool_name} failed in {phase}")
        
        return results
    
    async def _execute_with_retry(self, tool_name: str, params: Dict, max_retries: int) -> ToolResult:
        """재시도 로직이 포함된 도구 실행"""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                tool = self.agent.tools[tool_name]
                result = await asyncio.wait_for(
                    tool.execute(**params),
                    timeout=self.timeout_seconds
                )
                
                if result.success:
                    return result
                    
                last_error = result.error_message
                
            except asyncio.TimeoutError:
                last_error = f"Tool {tool_name} timed out after {self.timeout_seconds}s"
            except Exception as e:
                last_error = str(e)
            
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # 지수 백오프
        
        return ToolResult(
            success=False,
            data=None,
            error_message=f"Failed after {max_retries} retries: {last_error}"
        )
```

### Phase Transition Management
Phase 간 데이터 전달 및 상태 관리:

```python
class PhaseTransitionManager:
    """Phase 간 전환 및 데이터 전달 관리"""
    
    def __init__(self):
        self.phase_results = {}
        self.transition_rules = {
            "commit_collection": "review_execution",
            "review_execution": "deepeval_conversion", 
            "deepeval_conversion": "analysis",
            "analysis": None  # 마지막 단계
        }
    
    def store_phase_result(self, phase: str, result: Any):
        """Phase 결과 저장"""
        self.phase_results[phase] = result
    
    def get_input_for_phase(self, phase: str) -> Dict[str, Any]:
        """다음 Phase의 입력 데이터 준비"""
        if phase == "commit_collection":
            return {}  # 첫 단계는 설정에서 입력
        elif phase == "review_execution":
            return {"commits": self.phase_results["commit_collection"]}
        elif phase == "deepeval_conversion":
            return {"reviews": self.phase_results["review_execution"]}
        elif phase == "analysis":
            return {"evaluations": self.phase_results["deepeval_conversion"]}
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    def get_next_phase(self, current_phase: str) -> Optional[str]:
        """다음 실행할 Phase 반환"""
        return self.transition_rules.get(current_phase)
```

## 상태 관리 및 메모리

### Working Memory
```python
class WorkingMemory:
    """에이전트 작업 메모리"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.memory = {}
        self.access_count = {}
        self.timestamps = {}
    
    def store(self, key: str, value: Any, ttl: Optional[int] = None):
        """메모리에 저장"""
        if len(self.memory) >= self.max_size:
            self._evict_lru()
        
        self.memory[key] = value
        self.access_count[key] = 0
        self.timestamps[key] = time.time()
        
        if ttl:
            asyncio.create_task(self._schedule_cleanup(key, ttl))
    
    def retrieve(self, key: str) -> Optional[Any]:
        """메모리에서 조회"""
        if key in self.memory:
            self.access_count[key] += 1
            return self.memory[key]
        return None
    
    def _evict_lru(self):
        """LRU 정책으로 메모리 정리"""
        if not self.memory:
            return
        
        # 가장 적게 사용된 항목 제거
        lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
        self.remove(lru_key)
    
    async def _schedule_cleanup(self, key: str, ttl: int):
        """TTL 기반 자동 정리"""
        await asyncio.sleep(ttl)
        self.remove(key)
```

### Session State Management
```python
class SessionState:
    """평가 세션 상태 관리"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.current_phase = None
        self.phase_states = {}
        self.global_state = {}
        self.checkpoints = []
    
    def save_checkpoint(self, phase: str, state: Dict[str, Any]):
        """체크포인트 저장"""
        checkpoint = {
            "phase": phase,
            "timestamp": datetime.now(),
            "state": state,
            "checkpoint_id": f"{phase}_{len(self.checkpoints)}"
        }
        self.checkpoints.append(checkpoint)
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """체크포인트 복원"""
        for checkpoint in self.checkpoints:
            if checkpoint["checkpoint_id"] == checkpoint_id:
                return checkpoint["state"]
        return None
    
    def persist_to_disk(self, file_path: str):
        """디스크에 상태 저장"""
        state_data = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "current_phase": self.current_phase,
            "phase_states": self.phase_states,
            "global_state": self.global_state,
            "checkpoints": [
                {
                    **cp,
                    "timestamp": cp["timestamp"].isoformat()
                }
                for cp in self.checkpoints
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(state_data, f, indent=2)
```

## 에이전트 안전성 및 제약

### Resource Management
```python
class ResourceManager:
    """시스템 리소스 관리 및 제한"""
    
    def __init__(self, config: ResourceConfig):
        self.max_memory_mb = config.max_memory_mb
        self.max_cpu_percent = config.max_cpu_percent
        self.max_disk_gb = config.max_disk_gb
        self.max_execution_time = config.max_execution_time
        
        self.current_usage = ResourceUsage()
        self._monitoring_task = None
    
    async def start_monitoring(self):
        """리소스 모니터링 시작"""
        self._monitoring_task = asyncio.create_task(self._monitor_resources())
    
    async def _monitor_resources(self):
        """주기적 리소스 사용량 체크"""
        while True:
            try:
                usage = await self._get_current_usage()
                
                if usage.memory_mb > self.max_memory_mb:
                    await self._handle_memory_limit()
                
                if usage.cpu_percent > self.max_cpu_percent:
                    await self._handle_cpu_limit()
                
                if usage.disk_gb > self.max_disk_gb:
                    await self._handle_disk_limit()
                
                await asyncio.sleep(5)  # 5초마다 체크
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
    
    async def _handle_memory_limit(self):
        """메모리 한계 처리"""
        # 캐시 정리
        await self._clear_caches()
        
        # 가비지 컬렉션 강제 실행
        gc.collect()
        
        # 그래도 한계 초과시 예외 발생
        if await self._get_memory_usage() > self.max_memory_mb:
            raise ResourceLimitExceeded("Memory limit exceeded")
```

### Security Constraints
```python
class SecurityManager:
    """보안 제약 및 접근 제어"""
    
    def __init__(self, config: SecurityConfig):
        self.allowed_paths = config.allowed_paths
        self.forbidden_commands = config.forbidden_commands
        self.audit_log = AuditLog()
    
    def validate_file_access(self, file_path: str, operation: str) -> bool:
        """파일 접근 권한 검증"""
        abs_path = os.path.abspath(file_path)
        
        # 허용된 경로 내부인지 확인
        for allowed_path in self.allowed_paths:
            if abs_path.startswith(os.path.abspath(allowed_path)):
                self.audit_log.log_access(abs_path, operation, "ALLOWED")
                return True
        
        self.audit_log.log_access(abs_path, operation, "DENIED")
        return False
    
    def validate_command(self, command: List[str]) -> bool:
        """명령어 실행 권한 검증"""
        cmd_name = command[0] if command else ""
        
        if cmd_name in self.forbidden_commands:
            self.audit_log.log_command(command, "DENIED")
            return False
        
        # 특별 제약: selvage-deprecated는 읽기 전용
        if "selvage-deprecated" in " ".join(command):
            if any(write_op in " ".join(command) 
                   for write_op in ["commit", "push", "rm", "mv"]):
                self.audit_log.log_command(command, "DENIED - READ_ONLY")
                return False
        
        self.audit_log.log_command(command, "ALLOWED") 
        return True
```

## 사용 모델 전략
- **Primary**: `gemini-2.5-pro` (속도/비용 최적화)

## 대상 repo-path
- cline
    - path: /Users/demin_coder/Dev/cline
    - description: typescript로 구현된 coding assistant
- selvage-deprecated
    - path: /Users/demin_coder/Dev/selvage-deprecated
    - description: selvage가 정식 배포되기 전 commit history를 가지고 있는 repository (주의: 현재 selvage의 이전 작업 폴더이므로 review 대상으로서만 접근할 것)
- ecommerce-microservices
    - path: /Users/demin_coder/Dev/ecommerce-microservices
    - description: java, spring, jpa로 구현된 MSA 서버 애플리케이션
- kotlin-realworld
    - path: /Users/demin_coder/Dev/kotlin-realworld
    - description: java, kotlin, jpa로 구현된 호텔 예약 서버 애플리케이션

# 설정 파일

## CLI 실행 방식
터미널에서 `selvage-eval` 명령어로 바로 실행 가능하도록 설정 파일 기반 구성

### 설정 파일 스키마 (selvage-eval-config.yml)
```yaml
# Selvage 평가 에이전트 설정
agent-model: gemini-2.5-flash

evaluation:
  output_dir: "./selvage-eval-results"
  auto_session_id: true  # 자동 생성: eval_20240120_143022_abc123
  
target_repositories:
  - name: cline
    path: /Users/demin_coder/Dev/cline
    tech_stack: typescript
    description: "typescript로 구현된 coding assistant"
    filter_overrides:
      min_changed_lines: 30  # TS는 더 작은 단위 변경 허용
      file_types: [".ts", ".tsx", ".js", ".jsx"]
      
  - name: selvage-deprecated
    path: /Users/demin_coder/Dev/selvage-deprecated
    tech_stack: mixed
    description: "selvage 이전 버전 commit history"
    access_mode: readonly  # 읽기 전용 접근
    security_constraints:
      - no_write_operations
      - review_target_only
    filter_overrides:
      min_changed_lines: 50
      
  - name: ecommerce-microservices
    path: /Users/demin_coder/Dev/ecommerce-microservices
    tech_stack: java_spring
    description: "java, spring, jpa로 구현된 MSA 서버 애플리케이션"
    filter_overrides:
      min_changed_lines: 100  # Java는 더 큰 단위 변경
      file_types: [".java", ".kt", ".xml"]
      
  - name: kotlin-realworld
    path: /Users/demin_coder/Dev/kotlin-realworld
    tech_stack: kotlin_jpa
    description: "java, kotlin, jpa로 구현된 호텔 예약 서버 애플리케이션"
    filter_overrides:
      min_changed_lines: 80
      file_types: [".kt", ".java"]

review_models:
  - gemini-2.5-pro
  - claude-sonnet-4
  - claude-sonnet-4-thinking

commit_filters:
  keywords:
    include: [fix, feature, refactor, improve, add, update]
    exclude: [typo, format, style, docs, chore]
  stats:
    min_files: 2
    max_files: 10
    min_lines: 50
  merge_handling:
    fast_forward: exclude
    conflict_resolution: include
    squash_merge: include
    feature_branch: conditional  # 변경량 기준
commits_per_repo: 5

workflow:
  skip_existing:
    commit_filtering: true  # 필터링된 commit JSON 존재 시 skip
    review_results: true    # 동일 commit-model 조합 결과 존재 시 skip
  parallel_execution:
    max_concurrent_repos: 2
    max_concurrent_models: 3
  cache_enabled: true
```

### 실행 플래그 옵션
```bash
# 기본 실행
selvage-eval

# 설정 파일 지정
selvage-eval --config custom-config.yml

# 특정 저장소만 실행
selvage-eval --repos cline,ecommerce-microservices

# 특정 모델만 실행
selvage-eval --models gemini-2.5-flash

# 강제 재실행 (캐시 무시)
selvage-eval --force-refresh

# 특정 단계만 실행
selvage-eval --steps filter,review
```

### Skip 로직 상세
- **Meaningful Commit 필터링**: 이미 필터링된 commit 목록 JSON이 존재하면 skip
- **Selvage 리뷰**: 동일한 commit-model 조합의 결과가 존재하면 skip  
- **DeepEval 변환**: 동일한 평가 설정의 결과가 존재하면 skip
- **목적**: 동일한 data source로 재현 가능한 테스트 환경 제공 

## 환경 설정

### 필수 API 키
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key" 
export GEMINI_API_KEY="your-key"
```

### Selvage 통합 설정
- **바이너리 위치**: `/Users/demin_coder/.local/bin/selvage` (v0.1.2)
- **소스 코드**: `/Users/demin_coder/Dev/selvage`
- **통신 방식**: subprocess만 사용 (직접 API 호출 금지)

## 성능 최적화 전략

### 병렬 처리 설계
```python
# 커밋별 병렬 처리
async def process_commits_parallel(commits, models):
    semaphore = asyncio.Semaphore(5)  # 동시 실행 제한
    tasks = [
        process_single_commit(commit, models, semaphore)
        for commit in commits
    ]
    return await asyncio.gather(*tasks)
```

### 캐싱 전략
- **Git 데이터**: 커밋 정보 및 diff 내용 캐싱
- **Selvage 결과**: 동일 커밋/모델 조합 결과 재사용
- **DeepEval 메트릭**: 계산 결과 캐싱

### 성능 측정 지점
1. **Git 작업**: diff 추출, 통계 수집 시간
2. **Selvage 실행**: 프로세스 시작부터 완료까지
3. **API 호출**: 모델별 응답 시간 및 토큰 사용량
4. **데이터 변환**: JSON 파싱 및 변환 시간
5. **평가 실행**: DeepEval 메트릭 계산 시간

### 메타데이터 관리 (자동 생성)
```json
{
  "evaluation_session": {
    "id": "eval_20240620_143022_a1b2c3d",  // 자동 생성: 날짜_시간_git_hash
    "start_time": "2024-06-20T14:30:22Z",
    "end_time": "2024-06-20T16:45:30Z",
    "configuration": {
      "agent_model": "gemini-2.5-flash",
      "review_models": ["gemini-2.5-pro", "claude-sonnet-4", "claude-sonnet-4-thinking"],
      "target_repositories": [
        {"name": "cline", "path": "/Users/demin_coder/Dev/cline"},
        {"name": "ecommerce-microservices", "path": "/Users/demin_coder/Dev/ecommerce-microservices"}
      ],
      "commit_filter_criteria": {...},
      "evaluation_metrics": [...]
    },
    "results_summary": {
      "total_commits_per_repo": {
        "cline": 15,
        "ecommerce-microservices": 10
      },
      "successful_evaluations": 72,  // 25 commits × 3 models - 3 failures
      "failed_evaluations": 3,
      "repository_breakdown": {
        "cline": {"commits": 15, "success_rate": 0.96},
        "ecommerce-microservices": {"commits": 10, "success_rate": 0.94}
      }
    }
  }
}
```