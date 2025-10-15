import os, json, joblib, numpy as np, pandas as pd

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'boxing_winner_v1')

def load_model(model_dir: str = DEFAULT_MODEL_DIR):
    model_path = os.path.join(model_dir, 'logreg_delta.pkl')
    cfg_path   = os.path.join(model_dir, 'config.json')
    pipe = joblib.load(model_path)
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    return pipe, cfg

def _delta_vector(opp1: dict, opp2: dict, order: list) -> pd.DataFrame:
    mapping = {
        'round_ko_percentage': 'delta_round_ko_percentage',
        'estimated_punch_resistance': 'delta_estimated_punch_resistance',
        'estimated_ability_to_take_punch': 'delta_estimated_ability_to_take_punch',
        'avg_weight': 'delta_avg_weight',
        'estimated_punch_power': 'delta_estimated_punch_power',
        'rounds_boxed': 'delta_rounds_boxed',
        'has_been_ko_percentage': 'delta_has_been_ko_percentage',
    }
    deltas = {}
    for k, out_name in mapping.items():
        v1 = float(opp1.get(k, np.nan))
        v2 = float(opp2.get(k, np.nan))
        deltas[out_name] = v1 - v2
    return pd.DataFrame([[deltas[c] for c in order]], columns=order)

def predict_winner(opp1: dict, opp2: dict, threshold: float = None, model_dir: str = DEFAULT_MODEL_DIR):
    pipe, cfg = load_model(model_dir)
    order = cfg['delta_feature_order']
    thr   = cfg.get('threshold', 0.55) if threshold is None else threshold
    X = _delta_vector(opp1, opp2, order)
    prob = float(pipe.predict_proba(X)[0, 1])
    return {
        "prob_opp1_win": round(prob, 4),
        "decision_opp1_wins": bool(prob >= thr),
        "threshold": thr,
        "version": cfg.get('version', 'v1.0'),
        "deltas": dict(zip(X.columns, X.iloc[0].tolist()))
    }
