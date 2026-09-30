from io import BytesIO

import torch
from experiments.matchup_features.model import ResearchModel

from clash_sos.domain.attention_interactions import ExplicitInteractions


def test_zero_features_match_explicit_and_checkpoint_round_trip() -> None:
    tokens = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7, 13], [1, 2, 3, 4, 5, 6, 7, 10, 14]]])
    model = ResearchModel(17, 2)
    baseline = ExplicitInteractions(17, input_size=9)
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.normal_()
    model.explicit.load_state_dict(baseline.state_dict())
    x = torch.tensor([[2.0, -3.0]])
    assert torch.equal(model(tokens, x), baseline(tokens[:, 0], tokens[:, 1]))
    with torch.no_grad():
        model.feature_weights.copy_(torch.tensor([0.3, -0.7]))
    assert torch.allclose(model(tokens, x), -model(tokens.flip(1), -x), atol=1e-5)
    out = BytesIO()
    torch.save(model.state_dict(), out)
    out.seek(0)
    restored = ResearchModel(17, 2)
    restored.load_state_dict(torch.load(out, weights_only=True))
    assert torch.equal(model(tokens, x), restored(tokens, x))
    assert model.parameter_groups(0.001)[1]["weight_decay"] == 0
    assert torch.allclose(model.penalty(0.1), torch.tensor(0.029))
