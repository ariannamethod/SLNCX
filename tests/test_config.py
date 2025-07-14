import pathlib
import sys
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

try:
    from config import default_config
    from model import LanguageModelConfig
except Exception as exc:  # pragma: no cover - env may lack JAX
    pytest.skip(f"JAX unavailable: {exc}", allow_module_level=True)

def test_default_config_type():
    cfg = default_config()
    assert isinstance(cfg, LanguageModelConfig)
