import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import find_file, create_csv_file

import argparse

IDX = [] 
THRESHOLDS = []

TEST_AUC = []
TEST_AVG_PRECISION = []
TEST_PRECISION = []
TEST_RECALL = []
TEST_F1_SCORE = []
TEST_F1_TUNED = []

VALIDATION_AUC = []
VALIDATION_AVG_PRECISION = []
VALIDATION_PRECISION = []
VALIDATION_RECALL = []
VALIDATION_F1_SCORE = []
VALIDATION_F1_TUNED = []

LOGISTIC = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42)
        )

RAND_FOREST = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

def load_data(feat_group): 
    match feat_group:
        case "baseline":
            TRAIN = "baseline_train.csv"
            TEST = "baseline_test.csv"
            VAL = "baseline_validation.csv"
        case "markov":
            TRAIN = "markov_train.csv"
            TEST = "markov_test.csv"
            VAL = "markov_validation.csv"
        case "kernel":
            TRAIN = "kernel_train.csv"
            TEST = "kernel_test.csv"
            VAL = "kernel_validation.csv"
        case _:
            raise ValueError("Invalid feature group passed as argument")

    train_path = find_file(TRAIN)
    test_path = find_file(TEST)
    validation_path = find_file(VAL)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    validation = pd.read_csv(validation_path)

    return train, test, validation

def topology_split(df):
    feats = [
        "min_degree",
        "max_degree",
        "common_neighbors_score",
        "jaccard_coefficient",
        "adamic_adar",
        "resource_allocation",
        "shortest_path_len"
    ]

    x = df[feats]
    y = df[df.columns[len(df.columns) - 1]]

    return x, y

def markov_split(df, feats_group):
    if feats_group == "kernel" or feats_group == "markov":
        feats = [
            "two_step_transition",
            "three_step_transition",
            "four_step_transition",
            "five_step_transition"
        ]
    else:
        feats = [
            "two_step_transition",
            "three_step_transition",
        ]

    x = df[feats]
    y = df[df.columns[len(df.columns) - 1]]

    return x, y

def kernels_split(df, feats_group):
    if feats_group == "kernel":
        feats = [
            "inv_cos_score",
            "diffusion_score",
            "two_step_random_walk",
            "three_step_random_walk",
            "four_step_random_walk",
            "five_step_random_walk"
        ]
    else:
        feats = [
            "inv_cos_score",
            "diffusion_score",
            "two_step_random_walk",
            "three_step_random_walk",
        ]

    x = df[feats]
    y = df[df.columns[len(df.columns) - 1]]

    return x, y

def agg_split(x1, x2):
    X = pd.concat([x1, x2], axis=1)

    return X

def final_split(df):
    x = df[df.columns[3:len(df.columns) - 1]]
    y = df[df.columns[len(df.columns) - 1]]

    return x, y

def opt_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    i = int(np.argmax(f1))

    return float(thresholds[i]), float(f1[i])

def train_model(model, x, y):
    return model.fit(x, y)

def get_model_preds(fitted, x_test, x_val):
    y_pred = fitted.predict(x_test)
    y_prob = fitted.predict_proba(x_test)[:, 1]

    y_pred_val = fitted.predict(x_val)
    y_prob_val = fitted.predict_proba(x_val)[:, 1]

    return y_pred, y_prob, y_pred_val, y_prob_val

def get_model_metrics(y_test, y_val, y_pred, y_prob, y_pred_val, y_prob_val):
    auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    TEST_AUC.append(auc)
    TEST_AVG_PRECISION.append(avg_precision)
    TEST_PRECISION.append(precision)
    TEST_RECALL.append(recall)
    TEST_F1_SCORE.append(f1)

    auc = roc_auc_score(y_val, y_prob_val)
    avg_precision = average_precision_score(y_val, y_prob_val)
    precision = precision_score(y_val, y_pred_val)
    recall = recall_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val)

    VALIDATION_AUC.append(auc)
    VALIDATION_AVG_PRECISION.append(avg_precision)
    VALIDATION_PRECISION.append(precision)
    VALIDATION_RECALL.append(recall)
    VALIDATION_F1_SCORE.append(f1)

    threshold, f1_tuned = opt_threshold(y_val, y_prob_val)
    THRESHOLDS.append(threshold)
    VALIDATION_F1_TUNED.append(f1_tuned)

    test_pred = (y_prob >= threshold).astype(int)
    f1_tuned = f1_score(y_test, test_pred)
    TEST_F1_TUNED.append(f1_tuned)

def summarize_results(feats_group):
    test_results = pd.DataFrame(data={
            "ROC-AUC": TEST_AUC,
            "Average Precision": TEST_AVG_PRECISION,
            "Precision": TEST_PRECISION,
            "Recall": TEST_RECALL,
            "F1-Score": TEST_F1_SCORE,
            "Optimal Threshold": THRESHOLDS,
            "Tuned F1-Score": TEST_F1_TUNED
        }, index=IDX)

    val_results = pd.DataFrame(data={
            "ROC-AUC": VALIDATION_AUC,
            "Average Precision": VALIDATION_AVG_PRECISION,
            "Precision": VALIDATION_PRECISION,
            "Recall": VALIDATION_RECALL,
            "F1-Score": VALIDATION_F1_SCORE,
            "Optimal Threshold": THRESHOLDS,
            "Tuned F1-Score": VALIDATION_F1_TUNED
        }, index=IDX)

    DIR_NAME = "results"

    match feats_group:
        case "baseline":
            TEST_FILE_NAME = "baseline_models_test.csv"
            VALIDATION_FILE_NAME = "baseline_models_validation.csv"
            create_csv_file(test_results, TEST_FILE_NAME, feats_group, dir_name=DIR_NAME)
            create_csv_file(val_results, VALIDATION_FILE_NAME, feats_group, dir_name=DIR_NAME)
        case "markov":
            TEST_FILE_NAME = "markov_models_test.csv"
            VALIDATION_FILE_NAME = "markov_models_validation.csv"
            create_csv_file(test_results, TEST_FILE_NAME, feats_group, dir_name=DIR_NAME)
            create_csv_file(val_results, VALIDATION_FILE_NAME, feats_group, dir_name=DIR_NAME)
        case "kernel":
            TEST_FILE_NAME = "kernel_models_test.csv"
            VALIDATION_FILE_NAME = "kernel_models_validation.csv"
            create_csv_file(test_results, TEST_FILE_NAME, feats_group, dir_name=DIR_NAME)
            create_csv_file(val_results, VALIDATION_FILE_NAME, feats_group, dir_name=DIR_NAME)
        case _:
            raise ValueError("Invalid argument passed to the function")

def eval_baseline():
    train_df, test_df, validation_df = load_data("baseline")

    top_x_train, top_y_train = topology_split(train_df)
    top_x_test, top_y_test = topology_split(test_df)
    top_x_validation, top_y_validation = topology_split(validation_df)

    IDX.append("LR TOPOLOGY")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, top_x_train, top_y_train),
                top_x_test,
                top_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF TOPOLOGY")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, top_x_train, top_y_train),
                top_x_test,
                top_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    rw_x_train, rw_y_train = markov_split(train_df, "baseline")
    rw_x_test, rw_y_test = markov_split(test_df, "baseline")
    rw_x_validation, rw_y_validation = markov_split(validation_df, "baseline")
    
    IDX.append("LR MARKOV")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, rw_x_train, rw_y_train),
                rw_x_test,
                rw_x_validation
            )
    get_model_metrics(
            rw_y_test,
            rw_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF MARKOV")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, rw_x_train, rw_y_train),
                rw_x_test,
                rw_x_validation
            )
    get_model_metrics(
            rw_y_test,
            rw_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    kernel_x_train, kernel_y_train = kernels_split(train_df, "baseline")
    kernel_x_test, kernel_y_test = kernels_split(test_df, "baseline")
    kernel_x_validation, kernel_y_validation = kernels_split(validation_df, "baseline")

    IDX.append("LR KERNEL")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, kernel_x_train, kernel_y_train),
                kernel_x_test,
                kernel_x_validation
            )
    get_model_metrics(
            kernel_y_test,
            kernel_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF KERNEL")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, kernel_x_train, kernel_y_train),
                kernel_x_test,
                kernel_x_validation
            )
    get_model_metrics(
            kernel_y_test,
            kernel_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    top_rw_x_train = agg_split(top_x_train, rw_x_train)
    top_rw_x_test = agg_split(top_x_test, rw_x_test)
    top_rw_x_validation = agg_split(top_x_validation, rw_x_validation)
    
    IDX.append("LR TOPOLOGY + MARKOV")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, top_rw_x_train, top_y_train),
                top_rw_x_test,
                top_rw_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF TOPOLOGY + MARKOV")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, top_rw_x_train, top_y_train),
                top_rw_x_test,
                top_rw_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    top_kernel_x_train = agg_split(top_x_train, kernel_x_train)
    top_kernel_x_test = agg_split(top_x_test, kernel_x_test)
    top_kernel_x_validation = agg_split(top_x_validation, kernel_x_validation)

    IDX.append("LR TOPOLOGY + KERNEL")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, top_kernel_x_train, top_y_train),
                top_kernel_x_test,
                top_kernel_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF TOPOLOGY + KERNEL")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, top_kernel_x_train, top_y_train),
                top_kernel_x_test,
                top_kernel_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    rw_kernel_x_train = agg_split(rw_x_train, kernel_x_train)
    rw_kernel_x_test = agg_split(rw_x_test, kernel_x_test)
    rw_kernel_x_validation = agg_split(rw_x_validation, kernel_x_validation)

    IDX.append("LR MARKOV + KERNEL")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, rw_kernel_x_train, top_y_train),
                rw_kernel_x_test,
                rw_kernel_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF MARKOV + KERNEL")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, rw_kernel_x_train, top_y_train),
                rw_kernel_x_test,
                rw_kernel_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    final_x_train, final_y_train = final_split(train_df)
    final_x_test, final_y_test = final_split(test_df)
    final_x_validation, final_y_validation = final_split(validation_df)

    IDX.append("LR ALL")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, final_x_train, final_y_train),
                final_x_test,
                final_x_validation
            )
    get_model_metrics(
            final_y_test,
            final_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF ALL")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, final_x_train, final_y_train),
                final_x_test,
                final_x_validation
            )
    get_model_metrics(
            final_y_test,
            final_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    summarize_results("baseline")    

def eval_markov():
    train_df, test_df, validation_df = load_data("markov")

    top_x_train, top_y_train = topology_split(train_df)
    top_x_test, top_y_test = topology_split(test_df)
    top_x_validation, top_y_validation = topology_split(validation_df)

    IDX.append("LR TOPOLOGY")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, top_x_train, top_y_train),
                top_x_test,
                top_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF TOPOLOGY")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, top_x_train, top_y_train),
                top_x_test,
                top_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    rw_x_train, rw_y_train = markov_split(train_df, "markov")
    rw_x_test, rw_y_test = markov_split(test_df, "markov")
    rw_x_validation, rw_y_validation = markov_split(validation_df, "markov")
    
    IDX.append("LR MARKOV")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, rw_x_train, rw_y_train),
                rw_x_test,
                rw_x_validation
            )
    get_model_metrics(
            rw_y_test,
            rw_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF MARKOV")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, rw_x_train, rw_y_train),
                rw_x_test,
                rw_x_validation
            )
    get_model_metrics(
            rw_y_test,
            rw_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    kernel_x_train, kernel_y_train = kernels_split(train_df, "markov")
    kernel_x_test, kernel_y_test = kernels_split(test_df, "markov")
    kernel_x_validation, kernel_y_validation = kernels_split(validation_df, "markov")

    IDX.append("LR KERNEL")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, kernel_x_train, kernel_y_train),
                kernel_x_test,
                kernel_x_validation
            )
    get_model_metrics(
            kernel_y_test,
            kernel_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF KERNEL")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, kernel_x_train, kernel_y_train),
                kernel_x_test,
                kernel_x_validation
            )
    get_model_metrics(
            kernel_y_test,
            kernel_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    top_rw_x_train = agg_split(top_x_train, rw_x_train)
    top_rw_x_test = agg_split(top_x_test, rw_x_test)
    top_rw_x_validation = agg_split(top_x_validation, rw_x_validation)
    
    IDX.append("LR TOPOLOGY + MARKOV")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, top_rw_x_train, top_y_train),
                top_rw_x_test,
                top_rw_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF TOPOLOGY + MARKOV")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, top_rw_x_train, top_y_train),
                top_rw_x_test,
                top_rw_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    top_kernel_x_train = agg_split(top_x_train, kernel_x_train)
    top_kernel_x_test = agg_split(top_x_test, kernel_x_test)
    top_kernel_x_validation = agg_split(top_x_validation, kernel_x_validation)

    IDX.append("LR TOPOLOGY + KERNEL")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, top_kernel_x_train, top_y_train),
                top_kernel_x_test,
                top_kernel_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF TOPOLOGY + KERNEL")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, top_kernel_x_train, top_y_train),
                top_kernel_x_test,
                top_kernel_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    rw_kernel_x_train = agg_split(rw_x_train, kernel_x_train)
    rw_kernel_x_test = agg_split(rw_x_test, kernel_x_test)
    rw_kernel_x_validation = agg_split(rw_x_validation, kernel_x_validation)

    IDX.append("LR MARKOV + KERNEL")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, rw_kernel_x_train, top_y_train),
                rw_kernel_x_test,
                rw_kernel_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF MARKOV + KERNEL")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, rw_kernel_x_train, top_y_train),
                rw_kernel_x_test,
                rw_kernel_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    final_x_train, final_y_train = final_split(train_df)
    final_x_test, final_y_test = final_split(test_df)
    final_x_validation, final_y_validation = final_split(validation_df)

    IDX.append("LR ALL")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, final_x_train, final_y_train),
                final_x_test,
                final_x_validation
            )
    get_model_metrics(
            final_y_test,
            final_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF ALL")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, final_x_train, final_y_train),
                final_x_test,
                final_x_validation
            )
    get_model_metrics(
            final_y_test,
            final_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    summarize_results("markov")    

def eval_kernel():
    train_df, test_df, validation_df = load_data("kernel")

    top_x_train, top_y_train = topology_split(train_df)
    top_x_test, top_y_test = topology_split(test_df)
    top_x_validation, top_y_validation = topology_split(validation_df)

    IDX.append("LR TOPOLOGY")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, top_x_train, top_y_train),
                top_x_test,
                top_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF TOPOLOGY")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, top_x_train, top_y_train),
                top_x_test,
                top_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    rw_x_train, rw_y_train = markov_split(train_df, "kernel")
    rw_x_test, rw_y_test = markov_split(test_df, "kernel")
    rw_x_validation, rw_y_validation = markov_split(validation_df, "kernel")
    
    IDX.append("LR MARKOV")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, rw_x_train, rw_y_train),
                rw_x_test,
                rw_x_validation
            )
    get_model_metrics(
            rw_y_test,
            rw_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF MARKOV")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, rw_x_train, rw_y_train),
                rw_x_test,
                rw_x_validation
            )
    get_model_metrics(
            rw_y_test,
            rw_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    kernel_x_train, kernel_y_train = kernels_split(train_df, "kernel")
    kernel_x_test, kernel_y_test = kernels_split(test_df, "kernel")
    kernel_x_validation, kernel_y_validation = kernels_split(validation_df, "kernel")

    IDX.append("LR KERNEL")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, kernel_x_train, kernel_y_train),
                kernel_x_test,
                kernel_x_validation
            )
    get_model_metrics(
            kernel_y_test,
            kernel_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF KERNEL")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, kernel_x_train, kernel_y_train),
                kernel_x_test,
                kernel_x_validation
            )
    get_model_metrics(
            kernel_y_test,
            kernel_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    top_rw_x_train = agg_split(top_x_train, rw_x_train)
    top_rw_x_test = agg_split(top_x_test, rw_x_test)
    top_rw_x_validation = agg_split(top_x_validation, rw_x_validation)
    
    IDX.append("LR TOPOLOGY + MARKOV")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, top_rw_x_train, top_y_train),
                top_rw_x_test,
                top_rw_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF TOPOLOGY + MARKOV")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, top_rw_x_train, top_y_train),
                top_rw_x_test,
                top_rw_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    top_kernel_x_train = agg_split(top_x_train, kernel_x_train)
    top_kernel_x_test = agg_split(top_x_test, kernel_x_test)
    top_kernel_x_validation = agg_split(top_x_validation, kernel_x_validation)

    IDX.append("LR TOPOLOGY + KERNEL")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, top_kernel_x_train, top_y_train),
                top_kernel_x_test,
                top_kernel_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF TOPOLOGY + KERNEL")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, top_kernel_x_train, top_y_train),
                top_kernel_x_test,
                top_kernel_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    rw_kernel_x_train = agg_split(rw_x_train, kernel_x_train)
    rw_kernel_x_test = agg_split(rw_x_test, kernel_x_test)
    rw_kernel_x_validation = agg_split(rw_x_validation, kernel_x_validation)

    IDX.append("LR MARKOV + KERNEL")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, rw_kernel_x_train, top_y_train),
                rw_kernel_x_test,
                rw_kernel_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF MARKOV + KERNEL")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, rw_kernel_x_train, top_y_train),
                rw_kernel_x_test,
                rw_kernel_x_validation
            )
    get_model_metrics(
            top_y_test,
            top_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    final_x_train, final_y_train = final_split(train_df)
    final_x_test, final_y_test = final_split(test_df)
    final_x_validation, final_y_validation = final_split(validation_df)

    IDX.append("LR ALL")
    logistic_pred_test, logistic_prob_test, logistic_pred_val, logistic_prob_val = get_model_preds(
                train_model(LOGISTIC, final_x_train, final_y_train),
                final_x_test,
                final_x_validation
            )
    get_model_metrics(
            final_y_test,
            final_y_validation,
            logistic_pred_test,
            logistic_prob_test,
            logistic_pred_val,
            logistic_prob_val
        )

    IDX.append("RF ALL")
    rand_forest_pred_test, rand_forest_prob_test, rand_forest_pred_val, rand_forest_prob_val = get_model_preds(
                train_model(RAND_FOREST, final_x_train, final_y_train),
                final_x_test,
                final_x_validation
            )
    get_model_metrics(
            final_y_test,
            final_y_validation,
            rand_forest_pred_test,
            rand_forest_prob_test,
            rand_forest_pred_val,
            rand_forest_prob_val
        )

    summarize_results("kernel")    

def main():
    parser = argparse.ArgumentParser()    

    parser.add_argument(
        "--dataset",
        choices=["baseline", "markov", "kernel"],
        required=True,
        help="Select a dataset by features group (baseline: Topology, markov: Markov Chain, kernel: Kernels)"
    )

    args = parser.parse_args()
    
    match args.dataset:
        case "baseline":
            eval_baseline()
        case "markov":
            eval_markov()
        case "kernel":
            eval_kernel()
        case _:
            print("Invalid argument")

if __name__ == "__main__":
    main()
