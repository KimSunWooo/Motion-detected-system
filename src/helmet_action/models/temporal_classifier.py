from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from helmet_action.config import load_config, repo_root
from helmet_action.features.temporal_features import extract_feature_vector, feature_names
from helmet_action.models.labels import ActionClass, ML_CLASSES
from helmet_action.pose.confidence import prepare_sequence
from helmet_action.pose.normalizer import normalize_keypoints


class TemporalActionModel(Protocol):
    """Swap-in interface for a future LSTM / TCN / ST-GCN / Transformer."""

    classes_: list[str]

    def predict(self, keypoints: np.ndarray, confidence: np.ndarray | None = None) -> str:
        ...

    def predict_proba(self, keypoints: np.ndarray, confidence: np.ndarray | None = None) -> dict[str, float]:
        ...

    def save(self, path: str | Path) -> Path:
        ...


def _estimator(name: str, cfg):
    seed = int(cfg.get("ml.random_state", 17))
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(cfg.get("ml.n_estimators", 250)),
            max_depth=int(cfg.get("ml.max_depth", 12)),
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=int(cfg.get("ml.max_iter", 250)),
        max_depth=int(cfg.get("ml.max_depth", 6)),
        learning_rate=float(cfg.get("ml.learning_rate", 0.08)),
        l2_regularization=0.1,
        random_state=seed,
    )


@dataclass
class SklearnActionClassifier:
    estimator: object
    encoder: LabelEncoder
    feature_names: list[str]
    classes_: list[str]

    def vectorize(self, keypoints: np.ndarray, confidence: np.ndarray | None = None) -> np.ndarray:
        seq, _ = prepare_sequence(keypoints, confidence)
        seq_norm, _ = normalize_keypoints(seq)
        return extract_feature_vector(seq_norm)

    def predict_proba(self, keypoints: np.ndarray, confidence: np.ndarray | None = None) -> dict[str, float]:
        x = self.vectorize(keypoints, confidence).reshape(1, -1)
        proba = self.estimator.predict_proba(x)[0]
        out = {c: 0.0 for c in self.classes_}
        for idx, p in zip(self.estimator.classes_, proba):
            label = self.encoder.inverse_transform([idx])[0]
            out[str(label)] = float(p)
        return out

    def predict(self, keypoints: np.ndarray, confidence: np.ndarray | None = None) -> str:
        proba = self.predict_proba(keypoints, confidence)
        return max(proba.items(), key=lambda kv: kv[1])[0]

    def save(self, path: str | Path | None = None) -> Path:
        cfg = load_config()
        dest = Path(path) if path else repo_root() / cfg.get("ml.model_path", "models/action_classifier.joblib")
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "encoder": self.encoder,
                "feature_names": self.feature_names,
                "classes": self.classes_,
                "kind": "sklearn_hgb_v1",
            },
            dest,
        )
        return dest

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SklearnActionClassifier":
        cfg = load_config()
        dest = Path(path) if path else repo_root() / cfg.get("ml.model_path", "models/action_classifier.joblib")
        blob = joblib.load(dest)
        return cls(
            estimator=blob["estimator"],
            encoder=blob["encoder"],
            feature_names=list(blob["feature_names"]),
            classes_=list(blob.get("classes", [c.value for c in ML_CLASSES])),
        )

    @classmethod
    def try_load(cls, path: str | Path | None = None) -> "SklearnActionClassifier | None":
        try:
            return cls.load(path)
        except FileNotFoundError:
            return None
        except Exception:
            return None


def train_sklearn_classifier(
    X: np.ndarray,
    y: np.ndarray,
    estimator_name: str | None = None,
) -> SklearnActionClassifier:
    cfg = load_config()
    name = estimator_name or str(cfg.get("ml.estimator", "hist_gradient_boosting"))
    enc = LabelEncoder()
    y_idx = enc.fit_transform(y)
    est = _estimator(name, cfg)
    est.fit(X, y_idx)
    classes = [str(c) for c in enc.classes_]
    return SklearnActionClassifier(
        estimator=est,
        encoder=enc,
        feature_names=feature_names(),
        classes_=classes,
    )
