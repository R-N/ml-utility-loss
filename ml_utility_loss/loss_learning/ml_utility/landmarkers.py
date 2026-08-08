"""Cheap utility proxy from fast landmarker models.

Big win (2026-08-08): promoted from ``scripts/measure_landmarker_rf_baseline.py``
(CLAUDE.md "Random-forest-on-landmarkers baseline") into reusable code, per
the literature review's "cheap labels from landmarkers" item -- more proxy
training data provably raises the reward-model-overoptimization peak (Gao
et al., ICML 2023), so cheap relabeling matters twice over.

A suite of five deliberately weak, fast models (1-NN, decision stump,
Naive Bayes/linear, ZeroR) scores a table in 1-10 seconds; fitting a
``RandomForestRegressor`` from those scores to a handful of cached
``eval_ml_utility()`` labels reached 0.78-0.97 held-out Spearman across all
four project datasets (measured 2026-08-08, see CLAUDE.md) -- orders of
magnitude cheaper than a real CatBoost fit, at the accuracy cost documented
there.

Not a drop-in replacement for ``eval_ml_utility()``: ``LandmarkerProxy``
needs labeled examples to fit on (typically the existing
``aug_train/*/all/info.csv`` cache) and only sees the synthetic table, not
the paired real train table, so use it to pre-filter/rerank many cheap
candidates before a real CatBoost fit on the survivors -- not as the final
reported number.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

CLASSIFICATION_LANDMARKERS = {
    "1nn": lambda: KNeighborsClassifier(n_neighbors=1),
    "decision_stump": lambda: DecisionTreeClassifier(max_depth=1, random_state=0),
    "naive_bayes": lambda: GaussianNB(),
    "linear": lambda: LogisticRegression(max_iter=200),
    "zeror": lambda: DummyClassifier(strategy="most_frequent"),
}
REGRESSION_LANDMARKERS = {
    "1nn": lambda: KNeighborsRegressor(n_neighbors=1),
    "decision_stump": lambda: DecisionTreeRegressor(max_depth=1, random_state=0),
    "linear": lambda: LinearRegression(),
    "zeror": lambda: DummyRegressor(strategy="mean"),
}


def encode(df):
    """Label-encode every string/object column. Landmarkers need numeric input."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def landmark_features(df, task, target, cv_folds=3, seed=0):
    """Cross-validated score of each landmarker in the task's suite.

    Returns ``None`` if the table is too small for ``cv_folds``-fold CV.
    """
    df = encode(df)
    y = df[target].to_numpy()
    X = df.drop(columns=[target]).to_numpy(dtype=float)
    n = len(df)
    cv = min(cv_folds, n) if n >= 2 else 0
    if cv < 2:
        return None

    suite = CLASSIFICATION_LANDMARKERS if task != "regression" else REGRESSION_LANDMARKERS
    scoring = "accuracy" if task != "regression" else "r2"
    kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)

    scores = {}
    for name, make_model in suite.items():
        try:
            s = cross_val_score(make_model(), X, y, cv=kfold, scoring=scoring)
            scores[name] = float(np.mean(s))
        except Exception:
            scores[name] = np.nan
    return scores


class LandmarkerProxy:
    """Fits a ``RandomForestRegressor`` from landmarker features to utility labels.

    Cheap substitute for a real ``eval_ml_utility()`` fit once trained: call
    ``fit()`` with a handful of already-labeled synthetic tables (e.g. from
    ``aug_train/<dataset>/all/info.csv``), then ``predict()`` on new
    candidate tables in ~1-10s each instead of a full CatBoost fit.
    """

    def __init__(self, task, target, n_estimators=200, cv_folds=3, seed=0):
        self.task = task
        self.target = target
        self.cv_folds = cv_folds
        self.seed = seed
        self._model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
        self._feature_names = None

    def _features(self, df):
        scores = landmark_features(df, self.task, self.target, cv_folds=self.cv_folds, seed=self.seed)
        if scores is None:
            return None
        if self._feature_names is None:
            self._feature_names = sorted(scores)
        return np.array([scores.get(name, np.nan) for name in self._feature_names])

    def fit(self, dfs, labels):
        """``dfs``: iterable of synthetic tables. ``labels``: matching cached
        ``eval_ml_utility()`` scores. Tables too small for CV are skipped."""
        rows, ys = [], []
        for df, y in zip(dfs, labels):
            feats = self._features(df)
            if feats is not None:
                rows.append(feats)
                ys.append(y)
        if len(rows) < 2:
            raise ValueError("need at least 2 labeled tables with valid landmarker features")
        self._model.fit(np.stack(rows), np.array(ys))
        return self

    def predict(self, df):
        feats = self._features(df)
        if feats is None:
            raise ValueError("landmarker features unavailable for this table (too few rows for CV)")
        return float(self._model.predict(feats[None, :])[0])
