import argparse

import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    average_precision_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from utils import find_file, create_csv_file

NON_FEATS = ["source", "target", "edge", "label"]

LOGISTIC = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42)
)

RAND_FOREST = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

def load_data(feats_group):
    match feats_group:
        case "baseline" | "markov" | "kernel":
            TRAIN = f"{feats_group}_train.csv"
            TEST = f"{feats_group}_test.csv"
            VAL = f"{feats_group}_validation.csv"
        case _:
            raise ValueError("Invalid feature group passed as argument")

    train = pd.read_csv(find_file(TRAIN))
    test = pd.read_csv(find_file(TEST))
    validation = pd.read_csv(find_file(VAL))

    return train, test, validation

def feature_split(df):
    feats = df.drop(columns=NON_FEATS)
    labels = df["label"]

    return feats, labels

def correlation_analysis(train_feats):
    return train_feats.corr()

def sort_arrays(aucs, feats):
    for i in range(1, len(aucs)):
        for j in range(i, 0, -1):
            if aucs[j] >= aucs[j - 1]:
                aucs[j], aucs[j - 1] = aucs[j - 1], aucs[j]
                feats[j], feats[j - 1] = feats[j - 1], feats[j]

def singular_feature_auc(validation_feats, validation_labels):
    aucs = []
    feats_vec = []

    for feat in validation_feats.columns:
        auc = roc_auc_score(validation_labels, validation_feats[feat])
        aucs.append(auc)
        feats_vec.append(feat)

    sort_arrays(aucs, feats_vec)

    return pd.Series(data=aucs, index=feats_vec)

def train_model(model, x, y):
    return clone(model).fit(x, y)

def get_model_preds(fitted, x_val):
    y_prob = fitted.predict_proba(x_val)[:, 1]
    y_pred = fitted.predict(x_val)

    return y_pred, y_prob

def get_model_metrics(y_val, y_pred, y_prob):
    return {
        "roc auc": roc_auc_score(y_val, y_prob),
        "average precision": average_precision_score(y_val, y_prob),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1-score": f1_score(y_val, y_pred)
    }

def singular_feature_models(train_feats, train_labels, validation_feats, validation_labels):
    log_dfs = []
    forest_dfs = []

    for feat in train_feats.columns:
        print(f"Evaluating single feature: {feat}")
        logistic = train_model(LOGISTIC, train_feats[[feat]], train_labels)
        log_class, log_prob = get_model_preds(logistic, validation_feats[[feat]])
        data = get_model_metrics(validation_labels, log_class, log_prob)
        log_dfs.append(pd.DataFrame(data=data, index=[feat]))

        rand_forest = train_model(RAND_FOREST, train_feats[[feat]], train_labels)
        forest_class, forest_prob = get_model_preds(rand_forest, validation_feats[[feat]])
        data = get_model_metrics(validation_labels, forest_class, forest_prob)
        forest_dfs.append(pd.DataFrame(data=data, index=[feat]))

    logistic_summary = pd.concat(log_dfs, axis=0)
    forest_summary = pd.concat(forest_dfs, axis=0)

    return logistic_summary, forest_summary

def importance_summary(result, features):
    return pd.DataFrame({
        "feature": features.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)

def model_permutation_importance(model, train_feats, train_labels,
                                 validation_feats, validation_labels):
    fitted = train_model(model, train_feats, train_labels)
    result = permutation_importance(
        fitted,
        validation_feats,
        validation_labels,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42
    )

    return importance_summary(result, validation_feats)

def multiple_feature_analysis(train_feats, train_labels, validation_feats, validation_labels):
    print("Permutation importance: Logistic Regression", flush=True)
    logistic_summary = model_permutation_importance(
        LOGISTIC, train_feats, train_labels, validation_feats, validation_labels
    )

    print("Permutation importance: Random Forest", flush=True)
    forest_summary = model_permutation_importance(
        RAND_FOREST, train_feats, train_labels, validation_feats, validation_labels
    )

    return logistic_summary, forest_summary

def summarize_results(result, feats_group, suffix):
    DIR_NAME = "results"
    FILE_NAME = f"{feats_group}_{suffix}.csv"

    create_csv_file(result, FILE_NAME, feats_group, dir_name=DIR_NAME)

def eval_features(feats_group, analysis="all"):
    train_df, _, validation_df = load_data(feats_group)
    train_feats, train_labels = feature_split(train_df)
    validation_feats, validation_labels = feature_split(validation_df)
    validation_feats = validation_feats[train_feats.columns]

    if analysis in ("all", "correlation"):
        train_corr = correlation_analysis(train_feats)
        summarize_results(train_corr, feats_group, "pearson_corr")

    if analysis in ("all", "auc"):
        aucs_series = singular_feature_auc(validation_feats, validation_labels)
        summarize_results(aucs_series, feats_group, "auc_scores")

    if analysis in ("all", "models"):
        logistic_summary, forest_summary = singular_feature_models(
            train_feats, train_labels, validation_feats, validation_labels
        )
        summarize_results(logistic_summary, feats_group, "logistic_single_feat")
        summarize_results(forest_summary, feats_group, "rand_forest_single_feat")

    if analysis in ("all", "permutation"):
        logistic_summary, forest_summary = multiple_feature_analysis(
            train_feats, train_labels, validation_feats, validation_labels
        )
        summarize_results(logistic_summary, feats_group, "logistic_permutation")
        summarize_results(forest_summary, feats_group, "rand_forest_permutation")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["baseline", "markov", "kernel"],
        required=True,
        help="Select a dataset by features group (baseline: Topology, markov: Markov Chain, kernel: Kernels)"
    )
    parser.add_argument(
        "--analysis",
        choices=["all", "correlation", "auc", "models", "permutation"],
        default="all",
        help="Select an analysis to run (default: all)."
    )
    args = parser.parse_args()
    eval_features(args.dataset, args.analysis)

if __name__ == "__main__":
    main()
