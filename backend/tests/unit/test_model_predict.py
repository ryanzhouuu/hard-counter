"""Protect prediction dependency direction and legacy application imports."""

import subprocess
import sys


def test_prediction_import_does_not_load_training_modules() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

from clash_sos.application.model_predict import predict_matchup

assert callable(predict_matchup)
assert "clash_sos.application.model_train" not in sys.modules
assert "clash_sos.application.model_train_lgbm" not in sys.modules
assert "clash_sos.application.model_train_pair" not in sys.modules
""",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_model_train_preserves_legacy_prediction_exports() -> None:
    from clash_sos.application.model_errors import KaggleV6ModelTrainError
    from clash_sos.application.model_predict import predict_matchup
    from clash_sos.application.model_train import (
        KaggleV6ModelTrainError as LegacyError,
    )
    from clash_sos.application.model_train import (
        predict_matchup as legacy_predict_matchup,
    )

    assert LegacyError is KaggleV6ModelTrainError
    assert legacy_predict_matchup is predict_matchup
