import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from .commit_stats import CommitStats
from .commit_score import CommitScore
from .commit_data import CommitData
from .repository_metadata import RepositoryMetadata
from .repository_result import RepositoryResult
from .meaningful_commits_data import MeaningfulCommitsData
from selvage_eval.config.settings import EvaluationConfig, TargetRepository, load_commit_diversity_config
from selvage_eval.tools.tool_executor import ToolExecutor
from selvage_eval.tools.tool_result import ToolResult
from selvage_eval.commit_collection.commit_size_category import CommitSizeCategory
from selvage_eval.commit_collection.diversity_selector import DiversityBasedSelector

class CommitCollector:
    """의미있는 커밋 수집 및 필터링 클래스"""
    
    # 파일 타입별 점수 조정
    NON_CODE_FILES = {
        '.txt', '.rst', '.adoc', '.doc', '.docx', '.pdf',
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', 
        '.mp4', '.mp3', '.wav', '.zip', '.tar', '.gz', 
        '.exe', '.dll', '.so', '.dylib', '.bin',
        'package-lock.json', '.lock', '.cache', '.gitignore',
        '.md', '.yml', '.json', '.yaml'
    }
    
    MINOR_PENALTY_FILES = {
        '.toml', '.ini', '.env', 
        '.mdc', 'Dockerfile', 'Makefile', 'requirements.txt'
    }
    
    # 커밋에서 완전히 제외할 파일 타입 (파일 확장자)
    EXCLUDE_FILE_TYPES = {
        '.log',  # 로그 파일
        '.gif',  # 이미지 파일 (데모)
        '.cast', # asciinema 녹화 파일
        '.tmp',  # 임시 파일
    }
    
    # 키워드 정의
    POSITIVE_KEYWORDS = {
        'fix', 'feat', 'refactor', 'add', 'update', 'implement',
        'migrate', 'generate', 'create', 'support', 'integrate',
        'improve', 'optimize', 'enhance'  # 유지하되 가중치 낮춤
    }
    
    NEGATIVE_KEYWORDS = {
        'merge', 'revert', 'chore', 'docs', 'release', 'version',
        'dependency', 'readme', 'format', 'style', 'lint',
        'typo', 'whitespace', 'backup'
    }
    
    # 경로 패턴 가중치
    CORE_PATH_PATTERNS = [
        r'src/',                    # 범용적 소스 디렉토리 (모든 프로젝트)
        r'core/',                   # 모듈 기반 프로젝트 (kotlin-realworld)
        r'api/',                    # API 계층 (kotlin-realworld)
        r'core-impl/',              # 구현 모듈 (kotlin-realworld)
        r'[a-z-]+-service/',        # 마이크로서비스 패턴 (ecommerce-microservices)
        r'common-lib/',             # 공통 라이브러리 (ecommerce-microservices)
        r'selvage/',                # Python 패키지 (selvage-deprecated)
        r'webview-ui/',             # UI 모듈 (cline)
    ]
    UTILITY_PATH_PATTERNS = [
        r'utils/',                  # 유틸리티 함수
        r'helpers/',                # 헬퍼 함수
        r'scripts/',                # 스크립트 (모든 프로젝트에서 빈번)
        r'tools/',                  # 개발 도구
        r'proto/',                  # Protocol Buffers (cline)
    ]
    CONFIG_PATH_PATTERNS = [
        r'config/',                 # 설정 파일
        r'build/',                  # 빌드 관련
        r'\.github/',               # GitHub Actions/설정
        r'gradle/',                 # Gradle 설정
        r'tests/',                  # 테스트 디렉토리
        r'test/',                   # 테스트 디렉토리 (단수형)
        r'__tests__/',              # Jest 테스트 (JS/TS)
        r'e2e/',                    # End-to-end 테스트
        r'docs/',                   # 문서
        r'assets/',                 # 정적 파일
    ]
    
    def __init__(self, config: EvaluationConfig, tool_executor: ToolExecutor):
        """
        Args:
            config: 설정 정보 (target_repositories, commit_filters 포함)
            tool_executor: ExecuteSafeCommandTool을 포함한 도구 실행기
        """
        self.repo_configs = config.target_repositories
        self.commit_filters = config.commit_filters
        self.commits_per_repo = config.commits_per_repo
        self.tool_executor = tool_executor
        self.logger = logging.getLogger(__name__)
        
        # 커밋 다양성 설정 로드 및 초기화
        if config.commit_diversity is None:
            config.commit_diversity = load_commit_diversity_config()
        
        self.diversity_config = config.commit_diversity
        self.diversity_selector = DiversityBasedSelector(
            self.diversity_config
        ) if self.diversity_config.enabled else None
    
    def collect_commits(self) -> MeaningfulCommitsData:
        """
        모든 대상 저장소에서 의미있는 커밋을 수집하고 필터링
        
        Returns:
            MeaningfulCommitsData: 전체 수집 결과
        """
        repository_results: List[RepositoryResult] = []
        
        for repo_config in self.repo_configs:
            self.logger.info(f"저장소 처리 시작: {repo_config.name}")
            start_time = time.time()
            
            try:
                # 저장소별 커밋 수집 및 처리
                commits = self._collect_repo_commits(repo_config)
                filtered_commits = self._filter_commits(commits)
                scored_commits = [self._score_commits(commit) for commit in filtered_commits]
                selected_commits = self._select_diverse_commits(
                    scored_commits, 
                    self.commits_per_repo
                )
                
                # 메타데이터 생성
                processing_time = time.time() - start_time
                metadata = RepositoryMetadata(
                    total_commits=len(commits),
                    filtered_commits=len(filtered_commits),
                    selected_commits=len(selected_commits),
                    filter_timestamp=datetime.now(),
                    processing_time_seconds=processing_time
                )
                
                # 저장소 결과 생성
                repo_result = RepositoryResult(
                    repo_name=repo_config.name,
                    repo_path=repo_config.path,
                    commits=selected_commits,
                    metadata=metadata
                )
                
                repository_results.append(repo_result)
                self.logger.info(
                    f"저장소 처리 완료: {repo_config.name} "
                    f"({len(selected_commits)}/{len(commits)} 커밋 선별, "
                    f"{processing_time:.2f}초)"
                )
                
            except Exception as e:
                self.logger.error(f"저장소 처리 실패: {repo_config.name} - {e}")
                # 실패한 저장소는 건너뛰고 계속 진행
                continue
        
        return MeaningfulCommitsData(repositories=repository_results)
    
    def _collect_repo_commits(self, repo_config: TargetRepository) -> List[CommitData]:
        """단일 저장소에서 커밋 수집"""
        commits: List[CommitData] = []
        
        # 최근 커밋 목록 가져오기 (최대 1000개)
        log_result = self._execute_git_command(
            "git log --oneline --max-count=1000 --pretty=format:'%H|%s|%ae|%ai'",
            repo_config.path
        )
        
        if not log_result.success:
            self.logger.warning(f"Git log 실행 실패: {repo_config.path}")
            return commits
        
        for line in log_result.data['stdout'].strip().split('\n'):
            if not line.strip():
                continue
                
            try:
                parts = line.split('|', 3)
                if len(parts) != 4:
                    continue
                    
                commit_id, message, author, date_str = parts
                
                # 커밋 상세 정보 수집
                commit_data = self._get_commit_details(commit_id, message, author, date_str, repo_config.path)
                if commit_data:
                    commits.append(commit_data)
                    
            except Exception as e:
                self.logger.warning(f"커밋 파싱 실패: {line} - {e}")
                continue
        
        self.logger.info(f"수집된 커밋 수: {len(commits)}")
        return commits
    
    def _get_commit_details(self, commit_id: str, message: str, author: str, 
                          date_str: str, repo_path: str) -> Optional[CommitData]:
        """개별 커밋의 상세 정보 수집"""
        try:
            # 날짜 파싱
            commit_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            # 통계 정보 수집
            stats_result = self._execute_git_command(
                f"git show --stat --format='' {commit_id}",
                repo_path
            )
            
            if not stats_result.success:
                return None
            
            # 파일 목록 수집
            files_result = self._execute_git_command(
                f"git show --name-only --format='' {commit_id}",
                repo_path
            )
            
            if not files_result.success:
                return None
            
            # 통계 파싱
            stats = self._parse_commit_stats(stats_result.data['stdout'])
            file_paths = [f.strip() for f in files_result.data['stdout'].strip().split('\n') if f.strip()]
            
            # 기본 점수로 초기화 (나중에 _score_commits에서 계산)
            initial_score = CommitScore(
                total_score=100,
                file_type_penalty=0,
                scale_appropriateness_score=0,
                commit_characteristics_score=0,
                time_weight_score=0,
                additional_adjustments=0
            )
            
            return CommitData(
                id=commit_id,
                message=message,
                author=author,
                date=commit_date,
                stats=stats,
                score=initial_score,
                file_paths=file_paths
            )
            
        except Exception as e:
            self.logger.warning(f"커밋 상세 정보 수집 실패: {commit_id} - {e}")
            return None
    
    def _parse_commit_stats(self, stats_output: str) -> CommitStats:
        """git show --stat 출력 파싱"""
        lines = stats_output.strip().split('\n')
        files_changed = 0
        lines_added = 0
        lines_deleted = 0
        
        for line in lines:
            if 'file changed' in line or 'files changed' in line:
                # "2 files changed, 45 insertions(+), 12 deletions(-)" 형태 파싱
                parts = line.split(',')
                
                # 파일 수
                if 'file' in parts[0]:
                    match = re.search(r'(\d+)', parts[0])
                    if match:
                        files_changed = int(match.group(1))
                
                # 추가 라인 수
                for part in parts:
                    if 'insertion' in part:
                        match = re.search(r'(\d+)', part)
                        if match:
                            lines_added = int(match.group(1))
                    elif 'deletion' in part:
                        match = re.search(r'(\d+)', part)
                        if match:
                            lines_deleted = int(match.group(1))
        
        return CommitStats(
            files_changed=files_changed,
            lines_added=lines_added,
            lines_deleted=lines_deleted
        )
    
    def _filter_commits(self, commits: List[CommitData]) -> List[CommitData]:
        """키워드 및 통계 기준으로 커밋 필터링"""
        filtered_commits: List[CommitData] = []
        
        for commit in commits:
            # 1. 통계 기준 필터링
            if not self._passes_stats_filter(commit):
                continue
            
            # 2. 키워드 기준 필터링
            if not self._passes_keyword_filter(commit):
                continue
            
            # 3. 머지 커밋 특별 처리
            if not self._passes_merge_filter(commit):
                continue
            
            # 4. EXCLUDE_FILE_TYPES 포함 여부 필터링
            if any(Path(path).suffix.lower() in self.EXCLUDE_FILE_TYPES for path in commit.file_paths):
                continue
            
            filtered_commits.append(commit)
        
        self.logger.info(f"필터링 후 커밋 수: {len(filtered_commits)}/{len(commits)}")
        return filtered_commits
    
    def _passes_stats_filter(self, commit: CommitData) -> bool:
        """통계 기준 필터링"""
        stats = commit.stats
        
        # 파일 수 범위 확인
        if not (self.commit_filters.stats.min_files <= stats.files_changed <= self.commit_filters.stats.max_files):
            return False
        
        # 최소 변경 라인 수 확인
        if stats.total_lines_changed < self.commit_filters.stats.min_lines:
            return False
        
        return True
    
    def _passes_keyword_filter(self, commit: CommitData) -> bool:
        """키워드 기준 필터링"""
        message_lower = commit.message.lower()
        
        # 제외 키워드 확인
        for keyword in self.NEGATIVE_KEYWORDS:
            if keyword in message_lower:
                return False
        
        # 포함 키워드 확인 (하나라도 있으면 통과)
        for keyword in self.POSITIVE_KEYWORDS:
            if keyword in message_lower:
                return True
        
        # 포함 키워드가 없으면 제외
        return False
    
    def _passes_merge_filter(self, commit: CommitData) -> bool:
        """머지 커밋 특별 처리"""
        message_lower = commit.message.lower()
        
        # 머지 커밋 여부 확인
        if not ('merge' in message_lower or commit.message.startswith('Merge')):
            return True  # 일반 커밋은 통과
        
        # Fast-forward 머지는 제외 (변경사항이 거의 없음)
        if commit.stats.files_changed == 0 or commit.stats.total_lines_changed < 10:
            return False
        
        # 충돌 해결이나 스쿼시 머지는 포함
        if 'conflict' in message_lower or 'squash' in message_lower:
            return True
        
        # 기타 머지는 변경량 기준으로 판단
        return commit.stats.files_changed >= 2 and commit.stats.total_lines_changed >= 50

    def _score_commits(self, commit: CommitData) -> CommitData:
        """커밋 배점 계산 (0-100점)"""
        base_score = 100
        
        # A. 파일 타입 감점 조정
        file_type_penalty = self._calculate_file_type_penalty(commit.file_paths)
        
        # B. 변경 규모 적정성 (25점)
        scale_score = self._calculate_scale_appropriateness_score(commit.stats)
        
        # C. 커밋 특성 (25점)
        characteristics_score = self._calculate_commit_characteristics_score(commit)
        
        # D. 시간 가중치 (20점)
        time_score = self._calculate_time_weight_score(commit.date)
        
        # E. 추가 조정 사항
        additional_adjustments = self._calculate_additional_adjustments(commit)
        
        # 최종 점수 계산
        total_score = (base_score + file_type_penalty + scale_score + 
                      characteristics_score + time_score + additional_adjustments)
        
        # 0-100 범위로 정규화
        total_score = max(0, min(100, total_score))
        
        # 새로운 점수 객체 생성
        new_score = CommitScore(
            total_score=total_score,
            file_type_penalty=file_type_penalty,
            scale_appropriateness_score=scale_score,
            commit_characteristics_score=characteristics_score,
            time_weight_score=time_score,
            additional_adjustments=additional_adjustments
        )
        
        # 새로운 CommitData 객체 반환 (불변성 유지)
        return CommitData(
            id=commit.id,
            message=commit.message,
            author=commit.author,
            date=commit.date,
            stats=commit.stats,
            score=new_score,
            file_paths=commit.file_paths
        )
    
    def _calculate_file_type_penalty(self, file_paths: List[str]) -> int:
        """파일 타입별 감점 계산"""
        penalty = 0
        
        for file_path in file_paths:
            file_lower = file_path.lower()
            file_ext = Path(file_path).suffix.lower()
            file_name = Path(file_path).name.lower()
            
            # 비-코드 파일 감점 (-5점)
            if (file_ext in self.NON_CODE_FILES or 
                file_name in self.NON_CODE_FILES or
                any(pattern in file_name for pattern in ['.lock', '.cache'])):
                penalty -= 5
            
            # 경미한 감점 파일 (-2점)
            elif (file_ext in self.MINOR_PENALTY_FILES or 
                  file_name in self.MINOR_PENALTY_FILES):
                penalty -= 2
        
        return penalty
    
    def _calculate_scale_appropriateness_score(self, stats: CommitStats) -> int:
        """변경 규모 적정성 점수 계산 (25점)"""
        score = 0
        
        # 파일 수 적정성 (10점)
        if 2 <= stats.files_changed <= 4:
            score += 10
        elif 5 <= stats.files_changed <= 7:
            score += 7
        elif 8 <= stats.files_changed <= 10:
            score += 4
        
        # 변경 라인 수 밸런스 (15점)
        total_lines = stats.total_lines_changed
        if 50 <= total_lines <= 200:
            score += 15
        elif 201 <= total_lines <= 400:
            score += 10
        elif 401 <= total_lines <= 600:
            score += 5
        
        # 추가/삭제 비율 극단적인 경우 감점
        if total_lines > 0:
            addition_ratio = stats.addition_ratio
            if addition_ratio < 0.1 or addition_ratio > 0.9:  # 90% 이상 추가 또는 삭제
                score -= 5
        
        return score
    
    def _calculate_commit_characteristics_score(self, commit: CommitData) -> int:
        """커밋 특성 점수 계산 (25점)"""
        score = 0
        message_lower = commit.message.lower()
        
        # 긍정 키워드 (최대 15점)
        keyword_score = 0
        for keyword in self.POSITIVE_KEYWORDS:
            if keyword in message_lower:
                keyword_score += 5
                if keyword_score >= 15:  # 최대 15점
                    break
        score += min(keyword_score, 15)
        
        # 부정 키워드 감점
        for keyword in self.NEGATIVE_KEYWORDS:
            if keyword in message_lower:
                score -= 3
        
        # 경로 패턴 가중치 (10점)
        path_score = self._calculate_path_pattern_score(commit.file_paths)
        score += path_score
        
        return score
    
    def _calculate_path_pattern_score(self, file_paths: List[str]) -> int:
        """경로 패턴 가중치 계산"""
        core_count = 0
        utility_count = 0
        config_count = 0
        
        for file_path in file_paths:
            path_lower = file_path.lower()
            
            # 핵심 로직 경로
            if any(re.search(pattern, path_lower) for pattern in self.CORE_PATH_PATTERNS):
                core_count += 1
            # 유틸리티 경로
            elif any(re.search(pattern, path_lower) for pattern in self.UTILITY_PATH_PATTERNS):
                utility_count += 1
            # 설정/빌드 경로
            elif any(re.search(pattern, path_lower) for pattern in self.CONFIG_PATH_PATTERNS):
                config_count += 1
        
        # 주요 경로에 따른 점수 계산
        if core_count > 0:
            return 10
        elif utility_count > 0:
            return 7
        elif config_count > 0:
            return -5
        else:
            return 0  # 기타 경로
    
    def _calculate_time_weight_score(self, commit_date: datetime) -> int:
        """시간 가중치 점수 계산 (20점)"""
        now = datetime.now(commit_date.tzinfo)
        age = now - commit_date
        
        if age <= timedelta(days=30):
            return 20
        elif age <= timedelta(days=90):
            return 15
        elif age <= timedelta(days=180):
            return 10
        elif age <= timedelta(days=365):
            return 5
        else:
            return 2
    
    def _calculate_additional_adjustments(self, commit: CommitData) -> int:
        """추가 조정 사항 계산"""
        adjustment = 0
        message_lower = commit.message.lower()
        
        # 머지 커밋 처리
        if 'merge' in message_lower or commit.message.startswith('Merge'):
            if 'conflict' in message_lower:  # 충돌 해결 머지
                adjustment += 5
            elif 'squash' in message_lower:  # 스쿼시 머지
                adjustment += 3
        
        return adjustment
    
    def _select_top_commits(self, commits: List[CommitData], count: int) -> List[CommitData]:
        """점수 기준 상위 커밋 선별"""
        # 점수 기준 내림차순 정렬
        sorted_commits = sorted(commits, key=lambda c: c.score.total_score, reverse=True)
        
        # 상위 N개 선별
        selected = sorted_commits[:count]
        
        self.logger.info(f"선별된 커밋 수: {len(selected)}/{len(commits)}")
        if selected:
            scores = [c.score.total_score for c in selected]
            self.logger.info(f"점수 범위: {min(scores):.1f} ~ {max(scores):.1f}")
        
        return selected
    
    def _categorize_commit(self, commit: CommitData) -> CommitSizeCategory:
        """라인 수 우선 + 파일 수 보정 방식으로 커밋 분류
        
        Args:
            commit: 분류할 커밋 데이터
            
        Returns:
            해당하는 커밋 크기 카테고리
        """
        if not self.diversity_config:
            # 다양성 설정이 없으면 기본 분류
            return CommitSizeCategory.categorize_by_lines(
                commit.stats.total_lines_changed,
                commit.stats.files_changed
            )
        
        stats = commit.stats
        total_lines = stats.total_lines_changed
        files_changed = stats.files_changed
        
        # 설정에서 임계값 가져오기
        thresholds = self.diversity_config.size_thresholds
        
        # 1차: 라인 수 기준 분류
        if total_lines <= thresholds.extra_small_max:
            base_category = CommitSizeCategory.EXTRA_SMALL
        elif total_lines <= thresholds.small_max:
            base_category = CommitSizeCategory.SMALL
        elif total_lines <= thresholds.medium_max:
            base_category = CommitSizeCategory.MEDIUM
        elif total_lines <= thresholds.large_max:
            base_category = CommitSizeCategory.LARGE
        else:
            base_category = CommitSizeCategory.EXTRA_LARGE
        
        # 2차: 파일 수 보정
        correction = self.diversity_config.file_correction
        
        if files_changed >= correction.large_file_threshold:
            # 파일 수가 많으면 최소 LARGE 카테고리로 상향 조정
            categories = list(CommitSizeCategory)
            base_index = categories.index(base_category)
            large_index = categories.index(CommitSizeCategory.LARGE)
            return categories[max(base_index, large_index)]
        
        elif files_changed == 1 and total_lines > correction.single_file_large_lines:
            # 단일 파일에 많은 변경은 최대 LARGE로 제한
            categories = list(CommitSizeCategory)
            base_index = categories.index(base_category)
            large_index = categories.index(CommitSizeCategory.LARGE)
            return categories[min(base_index, large_index)]
        
        return base_category

    def _select_diverse_commits(
        self, 
        commits: List[CommitData], 
        count: int
    ) -> List[CommitData]:
        """다양성을 고려한 커밋 선택 (기존 _select_top_commits 대체)
        
        Args:
            commits: 선택 대상 커밋 목록
            count: 선택할 커밋 개수
            
        Returns:
            선택된 커밋 목록
        """
        if not self.diversity_selector or not self.diversity_config.enabled:
            self.logger.info("다양성 선택기가 비활성화됨. 기존 방식으로 선택합니다.")
            return self._select_top_commits(commits, count)
        
        self.logger.info(f"다양성 기반 커밋 선택 시작: {len(commits)}개 중 {count}개 선택")
        
        try:
            selected = self.diversity_selector.select_diverse_commits(commits, count)
            self.logger.info(f"다양성 기반 선택 완료: {len(selected)}개 선택됨")
            
            # 선택 결과 보고서 생성 (디버그 모드)
            if self.diversity_config.debug.export_selection_report:
                report = self.diversity_selector.generate_selection_report(commits, selected)
                self._log_selection_report(report)
            
            return selected
            
        except Exception as e:
            self.logger.error(f"다양성 기반 선택 실패: {e}. 기존 방식으로 fallback합니다.")
            return self._select_top_commits(commits, count)

    def _calculate_diversity_adjusted_score(
        self, 
        commit: CommitData, 
        category: CommitSizeCategory
    ) -> int:
        """다양성을 고려한 점수 조정
        
        Args:
            commit: 점수를 조정할 커밋
            category: 커밋의 크기 카테고리
            
        Returns:
            조정된 점수
        """
        base_score = commit.score.total_score
        
        if not self.diversity_config:
            return base_score
        
        category_config = self.diversity_config.categories.get(category.value)
        if not category_config:
            return base_score
        
        score_boost = category_config.score_boost
        return base_score + score_boost

    def _log_selection_report(self, report):
        """선택 결과 보고서 로깅"""
        self.logger.info("=== 커밋 선택 보고서 ===")
        self.logger.info(f"총 커밋: {report.total_available}개 → 선택: {report.total_selected}개")
        
        for category_name, distribution in report.category_distributions.items():
            original = distribution["original"]
            selected = distribution["selected"]
            ratio = selected / original if original > 0 else 0
            self.logger.info(f"{category_name}: {selected}/{original}개 ({ratio:.1%})")
    
    def _execute_git_command(self, command: str, cwd: str) -> ToolResult:
        """Git 명령어 실행"""
        try:
            return self.tool_executor.execute_tool_call("execute_safe_command", {
                "command": command,
                "cwd": cwd,
                "timeout": 60
            })
        except Exception as e:
            self.logger.error(f"Git 명령어 실행 실패: {command} - {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=str(e)
            )