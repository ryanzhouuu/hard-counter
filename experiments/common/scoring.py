from pathlib import Path

import torch

from experiments.common.calibration import TemperatureFit, fit_temperature
from experiments.common.checkpoints import predictions
from experiments.common.components import Components
from experiments.common.data_access import RoleAccess
from experiments.common.fit import FitResult, predict_logits
from experiments.common.predictions import Prediction, write_predictions
from experiments.player_adjustment.model import PlayerModel


def score_fit(
    stage: Path,
    access: RoleAccess,
    fitted: FitResult,
    recipe: Components,
) -> tuple[TemperatureFit, TemperatureFit, tuple[Prediction, ...]]:
    refit = access.read("refit", "fit")
    calibration = access.read("calibration", "calibrate")
    development = access.read("development", "compare")
    cal_x = recipe.builder(refit, calibration) / fitted.scales
    dev_x = recipe.builder(refit, development) / fitted.scales
    cal_z = predict_logits(fitted.model, calibration, cal_x)
    dev_z = predict_logits(fitted.model, development, dev_x)
    actual_temperature = fit_temperature(
        tuple(float(z) for z in cal_z), tuple(r.label for r in calibration)
    )
    matchup_temperature = actual_temperature
    if isinstance(fitted.model, PlayerModel):
        with torch.inference_mode():
            device = next(fitted.model.parameters()).device
            cal_match = (
                fitted.model.matchup_logits(
                    torch.tensor([r.tokens for r in calibration], dtype=torch.long, device=device)
                )
                .cpu()
                .numpy()
            )
            dev_match = (
                fitted.model.matchup_logits(
                    torch.tensor([r.tokens for r in development], dtype=torch.long, device=device)
                )
                .cpu()
                .numpy()
            )
        matchup_temperature = fit_temperature(
            tuple(float(z) for z in cal_match), tuple(r.label for r in calibration)
        )
        write_predictions(
            stage / "actual-development.json",
            predictions(development, dev_z, actual_temperature.temperature),
        )
        write_predictions(
            stage / "actual-calibration.json",
            predictions(calibration, cal_z, actual_temperature.temperature),
        )
        cal_z, dev_z = cal_match, dev_match
    cal_predictions = predictions(calibration, cal_z, matchup_temperature.temperature)
    dev_predictions = predictions(development, dev_z, matchup_temperature.temperature)
    write_predictions(stage / "calibration.json", cal_predictions)
    write_predictions(stage / "development.json", dev_predictions)
    return matchup_temperature, actual_temperature, dev_predictions
