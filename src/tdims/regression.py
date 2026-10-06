"""Nested cross-validation for a TDiMS regression, without descriptor leakage."""

from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold


def run_tdims_regression_cv(
    sm_list,
    y,
    *,
    radius=1,
    func_dis=-2,
    func_merge=sum,
    fragment_set=True,
    atom_set=True,
    fingerprint_set=True,
    cv=3,
    n_repeats=2,
    inner_cv=3,
    random_state=0,
    model_name="lasso",
    mode="quick",
    use_feature_selection=True,
    fs_alpha=0.0001,
    fs_threshold="mean",
    return_fold_details=True,
    fit_final_estimator=True,
    n_jobs=1,
    verbose=0,
):
    """Fit a regressor on TDiMS features with nested cross-validation.

    Descriptors are built from SMILES only. Feature selection and the
    hyperparameter search see the outer-training split, and the outer-test
    split is used only for the reported R².
    """
    from sklearn.feature_selection import SelectFromModel

    from tdims.tdims_ext import get_representation

    sm_list = np.asarray(sm_list)
    y = np.asarray(y)
    if sm_list.shape[0] != y.shape[0]:
        raise ValueError("sm_list and y must have the same length.")

    mode = str(mode).lower().strip()
    if mode not in {"full", "quick"}:
        raise ValueError("mode must be either 'full' or 'quick'.")

    x_all, key_all = get_representation(
        sm_list,
        model="tdims",
        radius=radius,
        func_dis=func_dis,
        func_merge=func_merge,
        fragment_set=fragment_set,
        atom_set=atom_set,
        fingerprint_set=fingerprint_set,
    )
    x_all = np.asarray(x_all)
    key_all = np.asarray(key_all)
    print(f"\nFull descriptor matrix shape: {x_all.shape}")

    if mode == "quick":
        alpha_lasso = np.array([1e-4, 1e-2, 1.0])
        alpha_en = np.array([1e-2, 1e-1, 1.0])
        alpha_ridge = np.array([1e-2, 1e0, 1e2])
        rf_min_samples_split = [2]
        en_l1_ratio = [0.2, 0.8]
    else:
        alpha_lasso = np.logspace(-6, 1, 8)
        alpha_en = np.logspace(-2, 2, 5)
        alpha_ridge = np.logspace(-2, 3, 6)
        rf_min_samples_split = [2, 3]
        en_l1_ratio = [0.2, 0.4, 0.6, 0.8]

    model_name = str(model_name).lower()
    if model_name == "lasso":
        base_model = Lasso(random_state=random_state, max_iter=100000, tol=1e-4)
        param_grid = {"alpha": alpha_lasso}
    elif model_name == "ridge":
        base_model = Ridge(random_state=random_state)
        param_grid = {"alpha": alpha_ridge}
    elif model_name == "elasticnet":
        base_model = ElasticNet(
            random_state=random_state, max_iter=100000, tol=1e-4
        )
        param_grid = {"alpha": alpha_en, "l1_ratio": en_l1_ratio}
    elif model_name in {"rf", "randomforest", "random_forest"}:
        base_model = RandomForestRegressor(
            n_estimators=200, random_state=random_state
        )
        param_grid = {"min_samples_split": rf_min_samples_split}
    else:
        raise ValueError(
            "model_name must be one of: 'lasso', 'ridge', 'elasticnet', 'rf'"
        )

    outer_cv = RepeatedKFold(
        n_splits=cv, n_repeats=n_repeats, random_state=random_state
    )
    fold_r2_scores = []
    fold_details = []
    estimators = []
    selectors = []
    best_params_per_fold = []
    start = time.time()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x_all, y)):
            x_train = x_all[train_idx]
            x_test = x_all[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            selected_keys = key_all.copy()
            selector_fitted = None

            if use_feature_selection:
                selector_fitted = SelectFromModel(
                    estimator=Lasso(
                        random_state=0,
                        alpha=fs_alpha,
                        max_iter=100000,
                        tol=1e-4,
                    ),
                    threshold=fs_threshold,
                )
                selector_fitted.fit(x_train, y_train)
                x_train_used = selector_fitted.transform(x_train)
                x_test_used = selector_fitted.transform(x_test)
                selected_keys = key_all[selector_fitted.get_support()]
            else:
                x_train_used = x_train
                x_test_used = x_test

            search = GridSearchCV(
                estimator=clone(base_model),
                param_grid=param_grid,
                scoring="r2",
                cv=KFold(
                    n_splits=inner_cv, shuffle=True, random_state=random_state
                ),
                n_jobs=n_jobs,
                refit=True,
                verbose=verbose,
            )
            search.fit(x_train_used, y_train)
            best_estimator = search.best_estimator_
            best_params = dict(search.best_params_)
            fold_r2 = best_estimator.score(x_test_used, y_test)
            fold_r2_scores.append(fold_r2)
            estimators.append(best_estimator)
            selectors.append(selector_fitted)
            best_params_per_fold.append(best_params)
            if return_fold_details:
                fold_details.append(
                    {
                        "fold": fold_idx,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "r2": float(fold_r2),
                        "n_features_before_fs": int(x_train.shape[1]),
                        "n_features_after_fs": int(x_train_used.shape[1]),
                        "selected_keys": list(selected_keys),
                        "best_params": best_params,
                        "best_inner_cv_score": float(search.best_score_),
                    }
                )

        elapsed = time.time() - start
        fold_r2_scores = np.asarray(fold_r2_scores, dtype=float)
        final_selector = None
        final_estimator = None
        final_selected_keys = list(key_all)
        final_best_params = None
        final_x_selected = x_all

        if fit_final_estimator:
            if use_feature_selection:
                final_selector = SelectFromModel(
                    estimator=Lasso(
                        random_state=0,
                        alpha=fs_alpha,
                        max_iter=100000,
                        tol=1e-4,
                    ),
                    threshold=fs_threshold,
                )
                final_selector.fit(x_all, y)
                x_all_used = final_selector.transform(x_all)
                final_selected_keys = list(key_all[final_selector.get_support()])
                final_x_selected = x_all_used
            else:
                x_all_used = x_all
            final_search = GridSearchCV(
                estimator=clone(base_model),
                param_grid=param_grid,
                scoring="r2",
                cv=KFold(
                    n_splits=inner_cv, shuffle=True, random_state=random_state
                ),
                n_jobs=n_jobs,
                refit=True,
                verbose=verbose,
            )
            final_search.fit(x_all_used, y)
            final_estimator = final_search.best_estimator_
            final_best_params = dict(final_search.best_params_)

    print("\n===== TDiMS regression CV result =====")
    print(f"Model family        : {model_name}")
    print(f"Mode                : {mode}")
    print(f"Descriptor shape    : {x_all.shape}")
    print(f"Use FS              : {use_feature_selection}")
    print(f"Outer CV            : RepeatedKFold(n_splits={cv}, n_repeats={n_repeats})")
    print(f"Inner CV            : KFold(n_splits={inner_cv}, shuffle=True)")
    print(f"R2 scores           : {np.round(fold_r2_scores, 4)}")
    std = float(np.std(fold_r2_scores, ddof=1)) if len(fold_r2_scores) > 1 else 0.0
    print(f"Mean R2 +/- std     : {np.mean(fold_r2_scores):.4f} +/- {std:.4f}")
    print(f"Execution time      : {elapsed:.6f} sec")

    result = {
        "r2_scores": fold_r2_scores,
        "r2_mean": float(np.mean(fold_r2_scores)),
        "r2_std": std,
        "X_shape": x_all.shape,
        "model_name": model_name,
        "mode": mode,
        "cv": cv,
        "n_repeats": n_repeats,
        "inner_cv": inner_cv,
        "use_feature_selection": use_feature_selection,
        "radius": radius,
        "func_dis": func_dis,
        "func_merge": getattr(func_merge, "__name__", str(func_merge)),
        "fragment_set": fragment_set,
        "elapsed_time_sec": float(elapsed),
        "estimators": estimators,
        "selectors": selectors,
        "best_params_per_fold": best_params_per_fold,
        "feature_names_all": list(key_all),
        "final_estimator": final_estimator,
        "final_selector": final_selector,
        "final_selected_keys": final_selected_keys,
        "final_best_params": final_best_params,
        "final_X_selected": final_x_selected,
        "final_y": np.asarray(y),
        "search_space": {
            "alpha_lasso": alpha_lasso.tolist(),
            "alpha_en": alpha_en.tolist(),
            "alpha_ridge": alpha_ridge.tolist(),
            "rf_min_samples_split": rf_min_samples_split,
            "en_l1_ratio": en_l1_ratio,
        },
    }
    if return_fold_details:
        result["fold_details"] = fold_details
    return result
