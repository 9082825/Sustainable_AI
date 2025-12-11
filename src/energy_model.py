import joblib
import numpy as np

MODEL_PATH = "model/energy_predictor/energy_predictor.pkl"

def load_energy_model(path: str = MODEL_PATH):
    return joblib.load(path)

def predict_energy(
    num_layers: int,
    training_hours: float,
    flops_per_hour: float,
    prompt_tokens: int,
    gpu_power_watts: float = 250.0,
    batch_size: int = 32,
    model=None,
) -> float:
    """
    Predict energy_kwh for the given config using your trained regression model.
    Make sure the feature order matches how you trained it.
    """
    if model is None:
        model = load_energy_model(path=MODEL_PATH)

    X = np.array(
        [[
            num_layers,
            training_hours,
            flops_per_hour,
            prompt_tokens,
            gpu_power_watts,
            batch_size,
        ]],
        dtype=float,
    )
    return float(model.predict(X)[0])
