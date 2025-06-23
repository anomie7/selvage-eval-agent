"""Selvage 평가 에이전트 단위 테스트를 위한 pytest 설정"""

import tempfile
import os
import shutil
from pathlib import Path
from typing import Generator
import pytest


def pytest_configure(config):
    """pytest 설정을 구성합니다. 단위/통합 테스트용 마커들을 등록합니다."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """테스트용 임시 디렉토리 생성"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file() -> Generator[Path, None, None]:
    """테스트용 임시 파일 생성"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
        tmp.write("테스트 파일 내용\n")
        tmp_path = Path(tmp.name)
    
    try:
        yield tmp_path
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@pytest.fixture
def temp_json_file() -> Generator[Path, None, None]:
    """테스트용 임시 JSON 파일 생성"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
        import json
        test_data = {"test": "data", "number": 42, "list": [1, 2, 3]}
        json.dump(test_data, tmp, ensure_ascii=False, indent=2)
        tmp_path = Path(tmp.name)
    
    try:
        yield tmp_path
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@pytest.fixture
def sample_file_content() -> str:
    """샘플 파일 내용"""
    return "안녕하세요\n이것은 테스트 파일입니다\n한글 내용이 포함되어 있습니다"


@pytest.fixture
def sample_json_data() -> dict:
    """샘플 JSON 데이터"""
    return {
        "name": "테스트 프로젝트",
        "version": "1.0.0", 
        "config": {
            "debug": True,
            "items": ["item1", "item2", "item3"]
        }
    }


@pytest.fixture
def mock_allowed_paths():
    """안전한 테스트 경로 목록"""
    return [
        "./test-results/",
        "/tmp/selvage-test/",
        "/Users/demin_coder/Dev/test-repo"
    ]