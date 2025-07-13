import pytest
from selvage_eval.review_execution_summary import ReviewExecutionSummary


class TestReviewExecutionSummary:
    """ReviewExecutionSummary 클래스 테스트"""
    
    @pytest.fixture
    def sample_execution_summary(self):
        """샘플 실행 요약"""
        return ReviewExecutionSummary(
            total_commits_reviewed=10,
            total_reviews_executed=10,
            total_successes=8,
            total_failures=2,
            execution_time_seconds=120.5,
            output_directory="/tmp/test_output",
            success_rate=0.8
        )
    
    def test_creation(self, sample_execution_summary):
        """정상적인 객체 생성 테스트"""
        assert sample_execution_summary.total_commits_reviewed == 10
        assert sample_execution_summary.total_reviews_executed == 10
        assert sample_execution_summary.total_successes == 8
        assert sample_execution_summary.total_failures == 2
        assert sample_execution_summary.execution_time_seconds == 120.5
        assert sample_execution_summary.output_directory == "/tmp/test_output"
        assert sample_execution_summary.success_rate == 0.8
    
    def test_to_dict_conversion(self, sample_execution_summary):
        """딕셔너리 변환 테스트"""
        data_dict = sample_execution_summary.to_dict()
        
        assert data_dict['total_commits_reviewed'] == 10
        assert data_dict['total_reviews_executed'] == 10
        assert data_dict['total_successes'] == 8
        assert data_dict['total_failures'] == 2
        assert data_dict['execution_time_seconds'] == 120.5
        assert data_dict['output_directory'] == "/tmp/test_output"
        assert data_dict['success_rate'] == 0.8
    
    def test_from_dict_conversion(self, sample_execution_summary):
        """딕셔너리에서 생성 테스트"""
        data_dict = sample_execution_summary.to_dict()
        recreated = ReviewExecutionSummary.from_dict(data_dict)
        
        assert recreated.total_commits_reviewed == sample_execution_summary.total_commits_reviewed
        assert recreated.total_reviews_executed == sample_execution_summary.total_reviews_executed
        assert recreated.total_successes == sample_execution_summary.total_successes
        assert recreated.total_failures == sample_execution_summary.total_failures
        assert recreated.execution_time_seconds == sample_execution_summary.execution_time_seconds
        assert recreated.output_directory == sample_execution_summary.output_directory
        assert recreated.success_rate == sample_execution_summary.success_rate
    
    def test_summary_message(self, sample_execution_summary):
        """요약 메시지 테스트"""
        message = sample_execution_summary.summary_message
        
        assert "10개 커밋" in message
        assert "10개 리뷰" in message
        assert "80.0% 성공" in message
    
    def test_perfect_success_rate(self):
        """100% 성공률 테스트"""
        summary = ReviewExecutionSummary(
            total_commits_reviewed=5,
            total_reviews_executed=5,
            total_successes=5,
            total_failures=0,
            execution_time_seconds=60.0,
            output_directory="/tmp/output",
            success_rate=1.0
        )
        
        assert summary.success_rate == 1.0
        assert "100.0% 성공" in summary.summary_message
    
    def test_zero_success_rate(self):
        """0% 성공률 테스트"""
        summary = ReviewExecutionSummary(
            total_commits_reviewed=3,
            total_reviews_executed=3,
            total_successes=0,
            total_failures=3,
            execution_time_seconds=30.0,
            output_directory="/tmp/output",
            success_rate=0.0
        )
        
        assert summary.success_rate == 0.0
        assert "0.0% 성공" in summary.summary_message