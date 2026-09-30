import subprocess
import sys


def test_research_imports_without_torch_or_production_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import experiments; import sys; assert 'torch' not in sys.modules"],
        check=True,
        capture_output=True,
    )
    assert result.returncode == 0
