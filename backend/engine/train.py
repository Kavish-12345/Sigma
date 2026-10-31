"""
Retrain the suitability model with corrected labels.

The labels mostly came from one description per row ("Structural steel"), so a material
capable of several things usually carries one class, and the model learned that it is
not the others. fixed_labels() adds a class where the row's own data supports it. It never
removes a label, and Data_labelled.csv is not changed.

Run from the project root:
    python -m backend.engine.train            compare only, nothing is saved
    python -m backend.engine.train --save     also replace the model in backend/models/
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, hamming_loss
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from .fit import CAST_GROUPS, HARDENED, HARDENING
from .preprocess import build_X, clean
from .pricing import load_prices, price_category
from .recommend import DATA_DIR, MODEL_DIR, _proba

SEED, N_SPLITS = 42, 5
CHANGED = ("Corrosion-resistant", "Castings", "Heat-resistant", "Heat-treatable", "Structural")  # classes fixed_labels adds to

CONFIGS = {
    "current settings": dict(max_iter=300, learning_rate=0.05),
    "regularised": dict(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=20, l2_regularization=1.0),
    "strongly regularised": dict(max_iter=80, learning_rate=0.05, max_depth=2, min_samples_leaf=30, l2_regularization=5.0),
}


def make_model(params: dict) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", OneVsRestClassifier(HistGradientBoostingClassifier(
                         class_weight="balanced", random_state=SEED, **params)))])


def fixed_labels(df: pd.DataFrame, categories: dict) -> tuple[pd.Series, dict]:
    """Original labels plus the classes each row's own data supports; also returns how many were added."""
    group = df["price_category"]
    text = df["Heat treatment"].fillna("").str.lower()
    hardened = text.str.contains(HARDENED) & ~text.str.contains("anneal")
    supported = {
        "Heat-treatable": hardened & (group.map(HARDENING).fillna(0) >= 0.5),
        "Structural": group.isin(("carbon_steel", "alloy_steel")) & (df["Sy"] >= 235) & (df["A5"] >= 10) & ~df["A5_imputed"],
        # not heat-resistant steel: its score is one value for a broad group
        "Corrosion-resistant": (group.map(lambda g: categories[g]["corrosion_score"]) >= 4) & (group != "heat_resistant_steel"),
        "Heat-resistant": group == "heat_resistant_steel",
        "Castings": group.isin(CAST_GROUPS),
    }
    labels, added = df["labels"].map(list), {}
    for name, mask in supported.items():
        missing = mask & ~labels.map(lambda v: name in v)
        added[name] = int(missing.sum())
        labels = labels.where(~missing, labels.map(lambda v: v + [name]))
    return labels, added


def out_of_fold(params: dict, X: pd.DataFrame, Y: np.ndarray, groups: pd.Series) -> np.ndarray:
    """Every row is predicted by a model that never saw its alloy."""
    P = np.zeros(Y.shape, dtype=float)
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, Y, groups):
        P[te] = _proba(make_model(params).fit(X.iloc[tr], Y[tr]), X.iloc[te])
    return P


def scores(Y: np.ndarray, P: np.ndarray) -> dict:
    pred = (P >= 0.5).astype(int)
    return {"micro_F1": round(f1_score(Y, pred, average="micro", zero_division=0), 3),
            "macro_F1": round(f1_score(Y, pred, average="macro", zero_division=0), 3),
            "exact_match": round(accuracy_score(Y, pred), 3),
            "hamming_loss": round(hamming_loss(Y, pred), 3)}


def extreme_share(P: np.ndarray) -> float:
    return round(float(((P < 0.05) | (P > 0.95)).mean()), 3)


def main(save: bool = False) -> None:
    raw = pd.read_csv(DATA_DIR / "Data_labelled.csv")
    df, report = clean(raw)
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
    train = df[df["status"] == "train"].reset_index(drop=True)
    categories = load_prices()["categories"]
    train["price_category"] = train.apply(price_category, axis=1)
    labels, added = fixed_labels(train, categories)
    print(f"{len(train)} rows, labels added: {added} (total {sum(added.values())})")

    X, groups = build_X(train, meta["features"]), train["group"]
    mlb = MultiLabelBinarizer().fit(train["labels"])
    classes = list(mlb.classes_)
    Y_old, Y_new = mlb.transform(train["labels"]), mlb.transform(labels)
    same = [i for i, c in enumerate(classes) if c not in CHANGED]

    print("\nsaved model, old inputs, old labels:", meta["cv"]["scores"]["All"])
    P_inputs = out_of_fold(CONFIGS["current settings"], X, Y_old, groups)
    print("new inputs, old labels:             ", scores(Y_old, P_inputs))
    best, best_key, results = None, None, {}
    for name, params in CONFIGS.items():
        P = out_of_fold(params, X, Y_new, groups)
        results[name] = P
        unchanged = round(f1_score(Y_old[:, same], (P[:, same] >= 0.5).astype(int), average="macro", zero_division=0), 3)
        brier = round(float(np.mean([brier_score_loss(Y_new[:, i], P[:, i]) for i in range(len(classes))])), 4)
        print(f"\n{name}: new labels {scores(Y_new, P)}")
        print(f"  macro F1 on classes with no added labels: {unchanged} | extreme scores {extreme_share(P):.1%} | Brier {brier}")
        kept = round(float(((P >= 0.5) & (Y_old == 1)).sum() / Y_old.sum()), 3)
        print(f"  original labels still found: {kept:.1%}")
        if best_key is None or brier < best_key:
            best, best_key = name, brier
    P_base = out_of_fold(CONFIGS["current settings"], X, Y_old, groups)
    print(f"\nbaseline on the same classes: macro F1 {round(f1_score(Y_old[:, same], (P_base[:, same] >= 0.5).astype(int), average='macro'), 3)}"
          f" | extreme scores {extreme_share(P_base):.1%}")
    print("per class F1 (best model, new labels):",
          {c: round(f1_score(Y_new[:, i], (results[best][:, i] >= 0.5).astype(int), zero_division=0), 2) for i, c in enumerate(classes)})
    print("best by Brier:", best)

    if save:
        model = make_model(CONFIGS[best]).fit(X, Y_new)
        joblib.dump(model, MODEL_DIR / "suitability_model.joblib")
        joblib.dump(mlb, MODEL_DIR / "label_binarizer.joblib")
        meta.update(params=CONFIGS[best], labels_added=added, trained_on=datetime.date.today().isoformat(),
                    sklearn_version=sklearn.__version__, a5_medians=report["a5_medians"],
                    cv={"method": f"GroupKFold({N_SPLITS}), groups = alloy name without temper",
                        "scores": {"All": scores(Y_new, results[best])}})
        (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
        print("saved to", MODEL_DIR)


if __name__ == "__main__":
    main("--save" in sys.argv)
