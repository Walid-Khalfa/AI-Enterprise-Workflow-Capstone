import pytest

@pytest.fixture
def temporary_dir():
    return tmp_path / "logs"

@pytest.fixture
def temp_log_dir(tmp_path):
    return tmp_path / "logs"