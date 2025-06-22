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

## AI 에이전트 아키텍처 패러다임

### ReAct (Reasoning + Acting) 패턴
Selvage 평가 에이전트는 ReAct 패턴을 기반으로 설계되어 추론과 행동을 반복적으로 수행합니다.

```python
class SelvageEvaluationAgent:
    """
    ReAct 패턴을 구현한 Selvage 평가 에이전트
    """
    
    def __init__(self, model: str, tools: Dict[str, Tool]):
        self.model = model
        self.tools = tools
        self.working_memory = WorkingMemory()
        self.session_state = SessionState()
    
    async def execute_phase(self, phase: str, context: Dict[str, Any]) -> PhaseResult:
        """
        단계별 실행: 추론 → 계획 → 도구 선택 → 실행 → 평가
        """
        # 1. Reasoning: 현재 상황 분석 및 목표 설정
        analysis = await self.analyze_situation(phase, context)
        
        # 2. Planning: 도구 실행 계획 수립
        execution_plan = await self.create_execution_plan(analysis)
        
        # 3. Acting: 계획에 따른 도구 실행
        results = await self.execute_tools(execution_plan)
        
        # 4. Evaluation: 결과 평가 및 다음 단계 결정
        evaluation = await self.evaluate_results(results, analysis.goals)
        
        return PhaseResult(
            phase=phase,
            analysis=analysis,
            plan=execution_plan,
            results=results,
            evaluation=evaluation
        )
```

### Multi-Agent Coordination
복잡한 4단계 워크플로우를 위한 전문화된 서브 에이전트들:

```python
class AgentOrchestrator:
    """
    다중 에이전트 조정 및 관리
    """
    
    def __init__(self):
        self.commit_collector_agent = CommitCollectorAgent()
        self.review_executor_agent = ReviewExecutorAgent() 
        self.evaluation_agent = EvaluationAgent()
        self.analysis_agent = AnalysisAgent()
    
    async def execute_full_evaluation(self, config: EvaluationConfig) -> EvaluationReport:
        """
        전체 평가 프로세스 실행
        """
        # Phase 1: Commit Collection
        commits = await self.commit_collector_agent.collect_meaningful_commits(
            repositories=config.target_repositories,
            filters=config.commit_filters
        )
        
        # Phase 2: Review Execution  
        reviews = await self.review_executor_agent.execute_reviews(
            commits=commits,
            models=config.review_models
        )
        
        # Phase 3: Evaluation
        evaluations = await self.evaluation_agent.convert_and_evaluate(
            reviews=reviews,
            metrics=config.evaluation_metrics
        )
        
        # Phase 4: Analysis
        analysis = await self.analysis_agent.analyze_results(
            evaluations=evaluations,
            generate_insights=True
        )
        
        return EvaluationReport(
            session_id=generate_session_id(),
            commits=commits,
            reviews=reviews, 
            evaluations=evaluations,
            analysis=analysis
        )
```

## Tool 정의 및 분류

### Tool Interface 정의
모든 도구는 표준화된 인터페이스를 구현합니다:

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

## 에이전트 프롬프트 설계

### Master Orchestrator Prompt
```python
MASTER_ORCHESTRATOR_PROMPT = """
# ROLE
당신은 Selvage 코드 리뷰 도구를 평가하는 전문 AI 에이전트입니다.
4단계 워크플로우를 통해 체계적이고 정량적인 평가를 수행합니다.

# CAPABILITIES
- 다양한 도구를 사용하여 Git 저장소 분석, 코드 리뷰 실행, 결과 평가 수행
- 통계적 분석을 통한 모델 성능 비교 및 인사이트 도출
- 재현 가능한 평가 환경 구축 및 결과 문서화

# WORKFLOW
1. **Commit Collection**: meaningful한 커밋들을 자동 식별 및 배점
2. **Review Execution**: 다중 모델로 Selvage 리뷰 병렬 실행
3. **DeepEval Conversion**: 리뷰 결과를 DeepEval 형식으로 변환 및 평가
4. **Analysis & Insights**: 통계 분석을 통한 actionable insights 도출

# DECISION MAKING PRINCIPLES
- **데이터 기반**: 모든 결정은 정량적 데이터에 근거
- **재현성**: 동일 조건에서 동일 결과 보장
- **효율성**: 병렬 처리 및 캐싱을 통한 성능 최적화
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

### Phase-Specific Prompts

**Phase 1: Commit Collection Prompt**
```python
COMMIT_COLLECTION_PROMPT = """
# MISSION
지정된 저장소들에서 평가 가치가 높은 의미있는 커밋들을 식별하고 선별하세요.

# STRATEGY
1. **키워드 기반 1차 필터링**: fix, feature, refactor 등 포함, typo, format 등 제외
2. **통계 기반 2차 필터링**: 파일 수 2-10개, 변경 라인 50+ 기준
3. **배점 기반 최종 선별**: 파일 타입, 변경 규모, 커밋 특성, 시간 등 종합 고려

# TOOLS AVAILABLE
- git_log: 커밋 로그 조회
- git_show: 커밋 상세 정보
- commit_scoring: 커밋 배점 계산
- json_writer: 결과 저장

# EXECUTION STEPS
1. 각 저장소별로 git_log 실행하여 후보 커밋 수집
2. commit_scoring으로 각 커밋의 평가 가치 배점
3. commits_per_repo 설정에 따라 상위 점수 커밋 선별
4. meaningful_commits.json 형식으로 결과 저장

# SUCCESS CRITERIA
- 기술스택별로 적절한 커밋 다양성 확보
- 평가에 적합한 코드 변경 규모와 복잡도
- 재현 가능한 결과 저장 형식
"""
```

**Phase 2: Review Execution Prompt**
```python
REVIEW_EXECUTION_PROMPT = """
# MISSION  
선별된 커밋들에 대해 다중 모델로 Selvage 코드 리뷰를 병렬 실행하세요.

# STRATEGY
1. **안전한 커밋 체크아웃**: 각 커밋으로 이동 후 리뷰 실행, 완료 후 HEAD 복원
2. **모델별 병렬 실행**: review_models 설정에 따른 동시 실행
3. **결과 체계적 저장**: repo/commit/model 구조로 로그 분리 저장

# TOOLS AVAILABLE
- git_checkout: 커밋 체크아웃
- selvage_executor: Selvage 리뷰 실행
- process_monitor: 실행 상태 모니터링
- log_parser: 결과 파싱

# EXECUTION STEPS
1. meaningful_commits.json에서 커밋 목록 로드
2. 각 커밋별로 모델별 병렬 리뷰 실행
3. 실행 결과 검증 및 에러 처리
4. 구조화된 디렉토리에 결과 저장

# ERROR HANDLING
- Selvage 실행 실패 시 최대 3회 재시도
- 타임아웃 설정 및 무한 대기 방지
- 부분 실패 시에도 성공한 결과는 보존
"""
```

**Phase 3: DeepEval Conversion Prompt**
```python
DEEPEVAL_CONVERSION_PROMPT = """
# MISSION
Selvage 리뷰 결과를 DeepEval 테스트 케이스로 변환하고 4개 메트릭으로 평가하세요.

# STRATEGY
1. **리뷰 로그 스캔**: 저장된 모든 리뷰 결과 파일 탐색
2. **데이터 추출**: prompt와 review_response 필드 추출
3. **형식 변환**: DeepEval TestCase 형식으로 변환
4. **메트릭 평가**: Correctness, Clarity, Actionability, JsonCorrectness

# TOOLS AVAILABLE
- review_log_scanner: 리뷰 로그 파일 스캔
- data_extractor: prompt/response 추출
- deepeval_converter: 형식 변환
- metric_evaluator: 평가 실행

# EVALUATION METRICS
- **Correctness (0.7)**: 이슈 탐지 정확성 및 적절성
- **Clarity (0.7)**: 설명의 명확성 및 이해도
- **Actionability (0.7)**: 제안의 실행 가능성
- **JsonCorrectness**: 스키마 준수도

# OUTPUT FORMAT
repo_name과 model_name별로 그룹화하여 테스트 케이스 저장
"""
```

**Phase 4: Analysis Prompt** 
```python
ANALYSIS_PROMPT = """
# MISSION
DeepEval 평가 결과를 종합 분석하여 실제 의사결정에 도움이 되는 인사이트를 도출하세요.

# STRATEGY
1. **통계적 분석**: 메트릭별 성능 분포 및 유의성 검정
2. **모델 비교**: 기술스택별 최적 모델 조합 식별
3. **실패 패턴 분석**: 개선 방향 도출
4. **비용 최적화**: 성능 대비 비용 효율성 분석

# TOOLS AVAILABLE
- statistical_analysis: 통계 분석
- visualization: 차트 생성
- report_generator: 보고서 작성
- insight_extractor: 인사이트 도출

# ANALYSIS DIMENSIONS
- 모델별 종합 성능 비교
- 기술스택별 특화 성능 분석
- 실패 사유별 개선 우선순위
- 프롬프트 버전 효과성 A/B 테스트

# DELIVERABLES
1. Executive Summary (핵심 발견사항 및 권장사항)
2. Detailed Performance Matrix (모델별 상세 성능)
3. Actionable Insights (구체적 개선 방향)
4. Cost Optimization Recommendations (비용 효율화 방안)
"""
```

## Tool 실행 전략

### Sequential vs Parallel Execution
```python
class ToolExecutionStrategy:
    """도구 실행 전략 관리"""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_sequential(self, tools: List[Tool], contexts: List[Dict]) -> List[ToolResult]:
        """순차 실행 (의존성이 있는 경우)"""
        results = []
        for tool, context in zip(tools, contexts):
            result = await tool.execute(**context)
            results.append(result)
            
            # 실패 시 중단 여부 결정
            if not result.success and context.get("stop_on_failure", True):
                break
                
        return results
    
    async def execute_parallel(self, tools: List[Tool], contexts: List[Dict]) -> List[ToolResult]:
        """병렬 실행 (독립적인 경우)"""
        async def execute_with_semaphore(tool: Tool, context: Dict) -> ToolResult:
            async with self.semaphore:
                return await tool.execute(**context)
        
        tasks = [
            execute_with_semaphore(tool, context)
            for tool, context in zip(tools, contexts)
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def execute_pipeline(self, pipeline: ToolPipeline) -> PipelineResult:
        """파이프라인 실행 (출력이 다음 입력으로 연결)"""
        current_data = pipeline.initial_data
        results = []
        
        for stage in pipeline.stages:
            # 이전 단계 결과를 현재 단계 입력으로 변환
            stage_input = pipeline.transform_data(current_data, stage.input_mapping)
            
            # 단계 실행
            if stage.parallel:
                stage_results = await self.execute_parallel(stage.tools, stage_input)
            else:
                stage_results = await self.execute_sequential(stage.tools, stage_input)
            
            results.append(stage_results)
            current_data = pipeline.extract_output(stage_results)
        
        return PipelineResult(
            stages=results,
            final_output=current_data,
            success=all(r.success for stage in results for r in stage if hasattr(r, 'success'))
        )
```

### Tool Dependency Management
```python
class ToolDependencyGraph:
    """도구 간 의존성 관리"""
    
    def __init__(self):
        self.dependencies = {}  # tool_name -> [prerequisite_tools]
        self.results_cache = {}
    
    def add_dependency(self, tool: str, prerequisites: List[str]):
        """의존성 추가"""
        self.dependencies[tool] = prerequisites
    
    def get_execution_order(self) -> List[List[str]]:
        """위상 정렬을 통한 실행 순서 결정"""
        # Kahn's algorithm 구현
        in_degree = {tool: 0 for tool in self.dependencies}
        
        for tool, deps in self.dependencies.items():
            for dep in deps:
                in_degree[tool] += 1
        
        # 실행 레벨별 그룹화
        execution_levels = []
        remaining = set(self.dependencies.keys())
        
        while remaining:
            # 의존성이 없는 도구들 (현재 레벨)
            current_level = [
                tool for tool in remaining 
                if in_degree[tool] == 0
            ]
            
            if not current_level:
                raise ValueError("Circular dependency detected")
            
            execution_levels.append(current_level)
            
            # 다음 레벨 준비
            for tool in current_level:
                remaining.remove(tool)
                for dependent in self.dependencies:
                    if tool in self.dependencies[dependent]:
                        in_degree[dependent] -= 1
        
        return execution_levels
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