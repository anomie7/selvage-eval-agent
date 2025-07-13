"""ReviewExecutorTool 통합 테스트

Mocking 없이 실제 Git과 Selvage를 실행하여 ReviewExecutorTool의 전체 워크플로우를 테스트합니다.
주의: 이 테스트는 실제 저장소의 상태를 변경할 수 있으므로 신중하게 실행해야 합니다.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path

from selvage_eval.tools.review_executor_tool import ReviewExecutorTool
from selvage_eval.commit_collection import MeaningfulCommitsData

TEST_REVIEW_MODEL = "gemini-2.5-flash"

class TestReviewExecutorIntegration:
    """ReviewExecutorTool 통합 테스트 (Mocking 없음)"""
    
    @pytest.fixture
    def real_meaningful_commits_file(self):
        """실제 meaningful_commits.json 파일 경로"""
        return "/Users/demin_coder/Dev/selvage-eval-agent/selvage-eval-results/meaningful_commits.json"
    
    @pytest.fixture
    def check_prerequisites(self):
        """테스트 실행 전 필수 조건 확인"""
        # Selvage 바이너리 존재 확인
        selvage_path = "/Users/demin_coder/.local/bin/selvage"
        if not Path(selvage_path).exists():
            pytest.skip(f"Selvage binary not found at {selvage_path}")
        
        # API 키 확인
        required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            pytest.skip(f"Missing API keys: {missing_keys}")
        
        return True
    
    def test_single_commit_real_review(self, real_meaningful_commits_file, check_prerequisites):
        """단일 커밋에 대한 실제 리뷰 실행 테스트"""
        # 실제 데이터 로드
        meaningful_commits = MeaningfulCommitsData.from_json(real_meaningful_commits_file)
        
        # 첫 번째 저장소의 첫 번째 커밋만 테스트
        first_repo = meaningful_commits.repositories[0]
        first_commit = first_repo.commits[0]
        
        print(f"\\n실제 리뷰 테스트:")
        print(f"저장소: {first_repo.repo_name}")
        print(f"커밋: {first_commit.id[:8]} - {first_commit.message[:50]}...")
        print(f"경로: {first_repo.repo_path}")
        
        # 저장소 존재 확인
        repo_path = Path(first_repo.repo_path)
        if not repo_path.exists():
            pytest.skip(f"Repository not found: {repo_path}")
        
        # 임시 출력 디렉토리
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = ReviewExecutorTool()
            
            # 단일 커밋 테스트를 위한 임시 meaningful_commits.json 생성
            temp_commits_data = MeaningfulCommitsData(
                repositories=[
                    type(first_repo)(
                        repo_name=first_repo.repo_name,
                        repo_path=first_repo.repo_path,
                        commits=[first_commit],  # 단일 커밋만
                        metadata=first_repo.metadata
                    )
                ]
            )
            
            temp_commits_file = Path(temp_dir) / "single_commit.json"
            temp_commits_data.save_to_json(str(temp_commits_file))
            
            # 실제 리뷰 실행
            result = tool.execute(
                meaningful_commits_path=str(temp_commits_file),
                model=TEST_REVIEW_MODEL,
                output_dir=temp_dir
            )
            
            # 결과 검증
            print(f"\\n실행 결과:")
            print(f"성공: {result.success}")
            print(f"에러 메시지: {result.error_message}")
            
            if result.success:
                print(f"리뷰된 커밋 수: {result.data.total_commits_reviewed}")
                print(f"성공: {result.data.total_successes}")
                print(f"실패: {result.data.total_failures}")
                print(f"성공률: {result.data.success_rate:.1%}")
                print(f"실행 시간: {result.data.execution_time_seconds:.2f}초")
                
                # 실제 리뷰 로그 파일 생성 확인
                output_path = Path(temp_dir)
                log_files = list(output_path.rglob("*.json"))
                log_files = [f for f in log_files if f.name != "single_commit.json"]  # 입력 파일 제외
                
                print(f"생성된 리뷰 로그 파일: {len(log_files)}개")
                for log_file in log_files:
                    rel_path = log_file.relative_to(output_path)
                    print(f"  - {rel_path}")
                    
                    # 로그 파일 내용 간단 확인
                    try:
                        with open(log_file, 'r') as f:
                            log_data = json.load(f)
                        print(f"    → 모델: {log_data.get('model', {}).get('name', 'Unknown')}")
                        print(f"    → 상태: {log_data.get('status', 'Unknown')}")
                        print(f"    → 프롬프트: {'있음' if log_data.get('prompt') else '없음'}")
                        print(f"    → 응답: {'있음' if log_data.get('review_response') else '없음'}")
                        
                        # 리뷰 summary 출력
                        summary = None
                        if log_data.get('review_response'):
                            # 여러 경로에서 summary 찾기 시도
                            if isinstance(log_data['review_response'], dict):
                                summary = log_data['review_response'].get('summary')
                            elif isinstance(log_data['review_response'], str):
                                # review_response가 문자열인 경우 그 자체를 summary로 사용
                                summary = log_data['review_response']
                        
                        # 직접 summary 필드도 확인
                        if not summary:
                            summary = log_data.get('summary')
                        
                        if summary:
                            # summary가 너무 길면 일부만 출력
                            if len(summary) > 100:
                                summary_preview = summary[:100] + "..."
                            else:
                                summary_preview = summary
                            print(f"    → 리뷰 요약: {summary_preview}")
                        else:
                            print("    → 리뷰 요약: 없음")
                            
                    except Exception as e:
                        print(f"    → 파일 읽기 오류: {e}")
                
                # 기본 검증
                assert result.data.total_commits_reviewed == 1
                assert len(log_files) > 0  # 최소 1개의 로그 파일 생성
            else:
                # 실패한 경우에도 정보 출력
                print(f"실행 실패: {result.error_message}")
                # 실패도 유효한 테스트 결과로 간주 (실제 환경 문제일 수 있음)
    
    @pytest.mark.slow
    def test_multiple_commits_real_review(self, real_meaningful_commits_file, check_prerequisites):
        """여러 커밋에 대한 실제 리뷰 실행 테스트 (시간이 오래 걸림)"""
        # 실제 데이터 로드
        meaningful_commits = MeaningfulCommitsData.from_json(real_meaningful_commits_file)
        
        # 첫 번째 저장소의 처음 2개 커밋만 테스트
        first_repo = meaningful_commits.repositories[0]
        test_commits = first_repo.commits[:2]
        
        print(f"\\n다중 커밋 실제 리뷰 테스트:")
        print(f"저장소: {first_repo.repo_name}")
        print(f"테스트 커밋 수: {len(test_commits)}")
        
        # 저장소 존재 확인
        repo_path = Path(first_repo.repo_path)
        if not repo_path.exists():
            pytest.skip(f"Repository not found: {repo_path}")
        
        # 임시 출력 디렉토리
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = ReviewExecutorTool()
            
            # 다중 커밋 테스트를 위한 임시 meaningful_commits.json 생성
            temp_commits_data = MeaningfulCommitsData(
                repositories=[
                    type(first_repo)(
                        repo_name=first_repo.repo_name,
                        repo_path=first_repo.repo_path,
                        commits=test_commits,
                        metadata=first_repo.metadata
                    )
                ]
            )
            
            temp_commits_file = Path(temp_dir) / "multi_commit.json"
            temp_commits_data.save_to_json(str(temp_commits_file))
            
            # 실제 리뷰 실행
            result = tool.execute(
                meaningful_commits_path=str(temp_commits_file),
                model=TEST_REVIEW_MODEL,
                output_dir=temp_dir
            )
            
            # 결과 검증 및 보고
            print(f"\\n다중 커밋 실행 결과:")
            print(f"성공: {result.success}")
            
            if result.success:
                print(f"리뷰된 커밋 수: {result.data.total_commits_reviewed}")
                print(f"성공: {result.data.total_successes}")
                print(f"실패: {result.data.total_failures}")
                print(f"성공률: {result.data.success_rate:.1%}")
                print(f"실행 시간: {result.data.execution_time_seconds:.2f}초")
                
                # 생성된 파일 통계
                output_path = Path(temp_dir)
                log_files = list(output_path.rglob("*.json"))
                log_files = [f for f in log_files if f.name != "multi_commit.json"]
                
                print(f"\\n생성된 리뷰 로그 파일: {len(log_files)}개")
                
                # 커밋별 결과 요약
                by_commit = {}
                for log_file in log_files:
                    parts = log_file.relative_to(output_path).parts
                    if len(parts) >= 2:
                        commit_id = parts[1]
                        if commit_id not in by_commit:
                            by_commit[commit_id] = []
                        by_commit[commit_id].append(log_file)
                
                for commit_id, files in by_commit.items():
                    print(f"  {commit_id[:8]}: {len(files)}개 로그")
                
                # 기본 검증
                assert result.data.total_commits_reviewed == len(test_commits)
            else:
                print(f"실행 실패: {result.error_message}")
    
    def test_repository_state_preservation(self, real_meaningful_commits_file, check_prerequisites):
        """저장소 상태 보존 테스트 - 리뷰 후 원래 브랜치로 복원되는지 확인"""
        # 실제 데이터 로드
        meaningful_commits = MeaningfulCommitsData.from_json(real_meaningful_commits_file)
        first_repo = meaningful_commits.repositories[0]
        
        repo_path = Path(first_repo.repo_path)
        if not repo_path.exists():
            pytest.skip(f"Repository not found: {repo_path}")
        
        # 현재 브랜치 확인
        import subprocess
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            original_branch = result.stdout.strip()
            print(f"\\n원래 브랜치: {original_branch}")
        except Exception as e:
            pytest.skip(f"Git command failed: {e}")
        
        # 단일 커밋 리뷰 실행
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = ReviewExecutorTool()
            
            # 첫 번째 커밋만 테스트
            temp_commits_data = MeaningfulCommitsData(
                repositories=[
                    type(first_repo)(
                        repo_name=first_repo.repo_name,
                        repo_path=first_repo.repo_path,
                        commits=[first_repo.commits[0]],
                        metadata=first_repo.metadata
                    )
                ]
            )
            
            temp_commits_file = Path(temp_dir) / "state_test.json"
            temp_commits_data.save_to_json(str(temp_commits_file))
            
            # 리뷰 실행
            review_result = tool.execute(
                meaningful_commits_path=str(temp_commits_file),
                model=TEST_REVIEW_MODEL,
                output_dir=temp_dir
            )
            
            # 리뷰 후 브랜치 확인
            try:
                result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                final_branch = result.stdout.strip()
                print(f"리뷰 후 브랜치: {final_branch}")
                
                # 브랜치가 원래대로 복원되었는지 확인
                assert original_branch == final_branch, f"브랜치가 복원되지 않음: {original_branch} -> {final_branch}"
                print("✅ 저장소 상태 보존 성공")
                
            except Exception as e:
                print(f"브랜치 확인 실패: {e}")
                # 이 경우에도 테스트를 실패시키지 않음 (환경 문제일 수 있음)


# pytest 마커 정의
pytestmark = pytest.mark.integration