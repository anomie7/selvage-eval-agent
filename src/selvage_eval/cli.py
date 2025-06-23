"""CLI 진입점

명령행 인터페이스를 통한 에이전트 실행을 제공합니다.
"""

import argparse
import sys
import logging
from pathlib import Path

from .config.settings import load_config, get_default_config_path
from .agent.core_agent import SelvageEvaluationAgent


def setup_logging(level: str = "INFO") -> None:
    """로깅 설정
    
    Args:
        level: 로그 레벨
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def interactive_mode(agent: SelvageEvaluationAgent) -> None:
    """대화형 모드 실행
    
    Args:
        agent: 에이전트 인스턴스
    """
    print("[INTERACTIVE] Selvage 평가 에이전트 (대화형 모드)")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("-" * 50)
    
    # 세션 정보 표시 (이미 생성자에서 초기화됨)
    session_id = agent.session_state.session_id
    print(f"[SESSION] 세션이 준비되었습니다: {session_id}")
    print()
    
    while True:
        try:
            user_input = input("질문: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("[EXIT] 세션을 종료합니다.")
                break
                
            if not user_input:
                continue
            
            print("[PROCESSING] 처리 중...")
            response = agent.handle_user_message(user_input)
            print(f"[RESPONSE] 답변: {response}")
            print()
            
        except KeyboardInterrupt:
            print("\n[EXIT] 세션을 종료합니다.")
            break
        except Exception as e:
            print(f"[ERROR] 오류가 발생했습니다: {e}")


def automatic_mode(agent: SelvageEvaluationAgent) -> None:
    """자동 실행 모드
    
    Args:
        agent: 에이전트 인스턴스
    """
    print("[AUTO] Selvage 평가 에이전트 (자동 실행 모드)")
    print("-" * 50)
    
    try:
        result = agent.execute_evaluation()
        print("[SUCCESS] 평가가 완료되었습니다.")
        print(f"[RESULT] 결과: {result}")
        
    except Exception as e:
        print(f"[ERROR] 평가 실행 중 오류가 발생했습니다: {e}")
        sys.exit(1)


def main() -> None:
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Selvage 평가 에이전트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  selvage-eval                          # 대화형 모드
  selvage-eval --auto                   # 자동 실행 모드
  selvage-eval --config custom.yml     # 사용자 설정 파일
  selvage-eval --repos cline,selvage   # 특정 저장소만
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="설정 파일 경로 (기본값: configs/selvage-eval-config.yml)"
    )
    
    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        help="자동 실행 모드 (대화형 모드 대신 전체 평가 자동 실행)"
    )
    
    parser.add_argument(
        "--repos", "-r",
        type=str,
        help="실행할 저장소 이름들 (쉼표로 구분, 예: cline,selvage)"
    )
    
    parser.add_argument(
        "--models", "-m",
        type=str,
        help="사용할 모델들 (쉼표로 구분, 예: gemini-2.5-pro,claude-sonnet-4)"
    )
    
    parser.add_argument(
        "--force-refresh", "-f",
        action="store_true",
        help="기존 결과 무시하고 강제 재실행"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="로그 레벨 (기본값: INFO)"
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.log_level)
    
    try:
        # 설정 파일 로드
        config_path = args.config
        if not config_path:
            try:
                config_path = get_default_config_path()
            except FileNotFoundError:
                print("[ERROR] 기본 설정 파일을 찾을 수 없습니다.")
                print("--config 옵션으로 설정 파일을 지정하거나")
                print("configs/selvage-eval-config.yml 파일을 생성하세요.")
                sys.exit(1)
        
        config = load_config(config_path)
        print(f"[CONFIG] 설정을 로드했습니다: {config_path}")
        
        # 저장소 필터링
        if args.repos:
            repo_names = [name.strip() for name in args.repos.split(",")]
            config.target_repositories = [
                repo for repo in config.target_repositories 
                if repo.name in repo_names
            ]
            print(f"[TARGET] 선택된 저장소: {repo_names}")
        
        # 모델 필터링
        if args.models:
            model_names = [name.strip() for name in args.models.split(",")]
            config.review_models = [
                model for model in config.review_models 
                if model in model_names
            ]
            print(f"[MODEL] 선택된 모델: {model_names}")
        
        # 강제 재실행 설정
        if args.force_refresh:
            config.workflow.skip_existing.commit_filtering = False
            config.workflow.skip_existing.review_results = False
            print("[FORCE] 강제 재실행 모드가 활성화되었습니다.")
        
        # 에이전트 생성
        agent = SelvageEvaluationAgent(config)
        
        # 실행 모드 선택
        if args.auto:
            automatic_mode(agent)
        else:
            interactive_mode(agent)
            
    except Exception as e:
        print(f"[ERROR] 실행 중 오류가 발생했습니다: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()