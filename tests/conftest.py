# Configuration des répertoires temporaires
@pytest.fixture
def temp_log_dir(tmp_path):
    return tmp_path / "logs"