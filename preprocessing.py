import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import expm
import pandas as pd

from utils import find_file, create_csv_file
from feature_engineer import *

import argparse

def load_data():
    TARGET = "ChCh-Miner_durgbank-chem-chem.tsv"

    data_path = find_file(TARGET)
    return pd.read_csv(data_path, sep="\t", header=None)

def build_full_graph(df):
    g = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1])

    return g

def get_negatives_from_full_graph(g):
    comp = nx.complement(g)
    edges = comp.edges()

    src = []
    target = []

    for u, v in edges:
        src.append(u)
        target.append(v)

    return pd.DataFrame(data={
        "source": src,
        "target": target
    }, index=range(len(src)))

def build_connected_ref_graph(g, df, seed=42):
    rng = np.random.default_rng(seed)
    src, target = df.columns[:2]

    edges = list(g.edges())
    rng.shuffle(edges)

    ccs = nx.utils.UnionFind(g.nodes)
    protected = set()

    for u, v in edges:
        if ccs[u] != ccs[v]:
            ccs.union(u, v)
            protected.add(frozenset((u, v)))

    keys = [frozenset((u, v)) for u, v in zip(df[src], df[target])]

    ref_mask = np.array([edge in protected for edge in keys], dtype=bool)

    n_ref = g.number_of_edges() // 2

    if ref_mask.sum() > n_ref:
        raise ValueError("Not enough edges to preserve connectivity")

    candidates = np.flatnonzero(~ref_mask)

    extra = rng.choice(candidates, size=n_ref - int(ref_mask.sum()), replace=False)

    ref_mask[extra] = True

    ref_edges = df.iloc[np.flatnonzero(ref_mask)]
    remaining = df.iloc[np.flatnonzero(~ref_mask)]

    train_edges = remaining.sample(frac=0.5, random_state=seed)
    test_edges = remaining.drop(train_edges.index)

    validation_edges = test_edges.sample(n=int(len(df.index) * 0.1), random_state=seed)
    test_edges = test_edges.drop(validation_edges.index)

    ref_graph = nx.from_pandas_edgelist(ref_edges, source=src, target=target)

    ref_graph.add_nodes_from(g.nodes)

    return (
            ref_graph, 
            train_edges.reset_index(drop=True),
            test_edges.reset_index(drop=True),
            validation_edges.reset_index(drop=True)
        )

def join_cols(row):
    to_merge = [row["source"], row["target"]]

    return ",".join(to_merge)

def create_baseline_set(g, curr_set):
    cp_set = curr_set.copy().reset_index(drop=True)
    if 0 in cp_set.columns:
        cp_set.rename(columns={0: "source", 1: "target"}, inplace=True)

    cp_set["edge"] = cp_set.apply(join_cols, axis=1)

    cp_set["min_degree"], cp_set["max_degree"] = compute_min_max_degrees_from_nodes(g, cp_set["edge"])
    cp_set["common_neighbors_score"] = compute_common_neighbors_from_nodes(g, cp_set["edge"])
    cp_set["jaccard_coefficient"] = compute_jaccard_from_nodes(g, cp_set["edge"])
    cp_set["adamic_adar"] = compute_adamic_adar_from_nodes(g, cp_set["edge"])
    cp_set["resource_allocation"] = compute_resource_allocation_from_nodes(g, cp_set["edge"])
    cp_set["shortest_path_len"] = compute_shortest_path_from_nodes(g, cp_set["edge"])
    cp_set["two_step_transition"] = get_markov_probs_from_nodes(g, cp_set["edge"], steps=2)
    cp_set["three_step_transition"] = get_markov_probs_from_nodes(g, cp_set["edge"], steps=3)
    cp_set["inv_cos_score"] = get_inv_cos_scores(g, cp_set["edge"])
    cp_set["diffusion_score"] = get_diffusion_scores(g, cp_set["edge"], sig=1)
    cp_set["two_step_random_walk"] = get_p_steps_rw_scores(g, cp_set["edge"], a=2, p=2)
    cp_set["three_step_random_walk"] = get_p_steps_rw_scores(g, cp_set["edge"], a=2, p=3)

    return cp_set

def create_markov_set(g, curr_set):
    cp_set = curr_set.copy().reset_index(drop=True)
    if 0 in cp_set.columns:
        cp_set.rename(columns={0: "source", 1: "target"}, inplace=True)

    cp_set["edge"] = cp_set.apply(join_cols, axis=1)

    cp_set["min_degree"], cp_set["max_degree"] = compute_min_max_degrees_from_nodes(g, cp_set["edge"])
    cp_set["common_neighbors_score"] = compute_common_neighbors_from_nodes(g, cp_set["edge"])
    cp_set["jaccard_coefficient"] = compute_jaccard_from_nodes(g, cp_set["edge"])
    cp_set["adamic_adar"] = compute_adamic_adar_from_nodes(g, cp_set["edge"])
    cp_set["resource_allocation"] = compute_resource_allocation_from_nodes(g, cp_set["edge"])
    cp_set["shortest_path_len"] = compute_shortest_path_from_nodes(g, cp_set["edge"])
    cp_set["two_step_transition"] = get_markov_probs_from_nodes(g, cp_set["edge"], steps=2)
    cp_set["three_step_transition"] = get_markov_probs_from_nodes(g, cp_set["edge"], steps=3)
    cp_set["four_step_transition"] = get_markov_probs_from_nodes(g, cp_set["edge"], steps=4)
    cp_set["five_step_transition"] = get_markov_probs_from_nodes(g, cp_set["edge"], steps=5)
    cp_set["inv_cos_score"] = get_inv_cos_scores(g, cp_set["edge"])
    cp_set["diffusion_score"] = get_diffusion_scores(g, cp_set["edge"], sig=1)
    cp_set["two_step_random_walk"] = get_p_steps_rw_scores(g, cp_set["edge"], a=2, p=2)
    cp_set["three_step_random_walk"] = get_p_steps_rw_scores(g, cp_set["edge"], a=2, p=3)

    return cp_set

def create_kernel_set(g, curr_set):
    cp_set = curr_set.copy().reset_index(drop=True)
    if 0 in cp_set.columns:
        cp_set.rename(columns={0: "source", 1: "target"}, inplace=True)

    cp_set["edge"] = cp_set.apply(join_cols, axis=1)

    cp_set["min_degree"], cp_set["max_degree"] = compute_min_max_degrees_from_nodes(g, cp_set["edge"])
    cp_set["common_neighbors_score"] = compute_common_neighbors_from_nodes(g, cp_set["edge"])
    cp_set["jaccard_coefficient"] = compute_jaccard_from_nodes(g, cp_set["edge"])
    cp_set["adamic_adar"] = compute_adamic_adar_from_nodes(g, cp_set["edge"])
    cp_set["resource_allocation"] = compute_resource_allocation_from_nodes(g, cp_set["edge"])
    cp_set["shortest_path_len"] = compute_shortest_path_from_nodes(g, cp_set["edge"])
    cp_set["two_step_transition"] = get_markov_probs_from_nodes(g, cp_set["edge"], steps=2)
    cp_set["three_step_transition"] = get_markov_probs_from_nodes(g, cp_set["edge"], steps=3)
    cp_set["four_step_transition"] = get_markov_probs_from_nodes(g, cp_set["edge"], steps=4)
    cp_set["five_step_transition"] = get_markov_probs_from_nodes(g, cp_set["edge"], steps=5)
    cp_set["inv_cos_score"] = get_inv_cos_scores(g, cp_set["edge"])
    cp_set["diffusion_score"] = get_diffusion_scores(g, cp_set["edge"], sig=1)
    cp_set["two_step_random_walk"] = get_p_steps_rw_scores(g, cp_set["edge"], a=2, p=2)
    cp_set["three_step_random_walk"] = get_p_steps_rw_scores(g, cp_set["edge"], a=2, p=3)
    cp_set["four_step_random_walk"] = get_p_steps_rw_scores(g, cp_set["edge"], a=2, p=4)
    cp_set["five_step_random_walk"] = get_p_steps_rw_scores(g, cp_set["edge"], a=2, p=5)

    return cp_set

def create_files(training, test, validation, negatives, n=3, m=10, feats_group="baseline"):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")

    if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)):
        raise ValueError("m must be an integer")

    training["label"] = 1
    test["label"] = 1
    validation["label"] = 1
    negatives["label"] = 0

    training_negatives = n * len(training.index)

    negatives_sample = negatives.sample(n=training_negatives, random_state=42)
    training_df = pd.concat([training, negatives_sample], axis=0).reset_index(drop=True)

    dropped_negatives = negatives.drop(index=negatives_sample.index).reset_index(drop=True)

    test_negatives = m * len(test.index)

    negatives_sample = dropped_negatives.sample(n=test_negatives, random_state=42)
    test_df = pd.concat([test, negatives_sample], axis=0).reset_index(drop=True)

    remaining_negatives = dropped_negatives.drop(index=negatives_sample.index)

    min_cuts = np.unique(
        validation["min_degree"].quantile([1/3, 2/3]).to_numpy()
    )
    max_cuts = np.unique(
        validation["max_degree"].quantile([1/3, 2/3]).to_numpy()
    )

    def assign_bins(df):
        df = df.copy()
        df["min_bin"] = np.searchsorted(min_cuts, df["min_degree"], side="left")
        df["max_bin"] = np.searchsorted(max_cuts, df["max_degree"], side="left")

        return df

    pos_binned = assign_bins(validation)
    neg_binned = assign_bins(remaining_negatives)

    sampled_groups = []

    for (min_bin, max_bin), group in pos_binned.groupby(["min_bin", "max_bin"]):
        candidates = neg_binned[
            (neg_binned["min_bin"] == min_bin)
            & (neg_binned["max_bin"] == max_bin)
        ]

        n_samples = len(group) * m

        if len(candidates) < n_samples:
            raise ValueError(
                f"Not enough negatives in bin ({min_bin}, {max_bin}): "
                f"requested {n_samples}, available {len(candidates)}"
            )

        sampled_groups.append(candidates.sample(n=n_samples, random_state=42))

    validation_negatives = pd.concat(
        sampled_groups, ignore_index=True
    ).drop(columns=["min_bin", "max_bin"])

    validation_df = pd.concat(
        [validation, validation_negatives],
        ignore_index=True
    )

    rng = np.random.default_rng(seed=42)

    train_idx = training_df.index.to_list()
    rng.shuffle(train_idx)
    training_df = training_df.loc[train_idx].reset_index(drop=True)

    test_idx = test_df.index.to_list()
    rng.shuffle(test_idx)
    test_df = test_df.loc[test_idx].reset_index(drop=True)

    validation_idx = validation_df.index.to_list()
    rng.shuffle(validation_idx)
    validation_df = validation_df.loc[validation_idx].reset_index(drop=True)

    match feats_group:
        case "baseline":
            create_csv_file(training_df, "baseline_train.csv", feats_group, dir_name="data")
            create_csv_file(test_df, "baseline_test.csv", feats_group, dir_name="data")
            create_csv_file(validation_df, "baseline_validation.csv", feats_group, dir_name="data")
        case "markov":
            create_csv_file(training_df, "markov_train.csv", feats_group, dir_name="data")
            create_csv_file(test_df, "markov_test.csv", feats_group, dir_name="data")
            create_csv_file(validation_df, "markov_validation.csv", feats_group, dir_name="data")
        case "kernel":
            create_csv_file(training_df, "kernel_train.csv", feats_group, dir_name="data")
            create_csv_file(test_df, "kernel_test.csv", feats_group, dir_name="data")
            create_csv_file(validation_df, "kernel_validation.csv", feats_group, dir_name="data")

def process_baseline_data():
    df = load_data()

    g = build_full_graph(df)
    negatives = get_negatives_from_full_graph(g)

    ref_g, train_set, test_set, validation_set = build_connected_ref_graph(g, df, seed=42)

    train_positives = create_baseline_set(ref_g, train_set)
    test_positives = create_baseline_set(ref_g, test_set)
    validation_positives = create_baseline_set(ref_g, validation_set)
    proc_negatives = create_baseline_set(ref_g, negatives)   

    create_files(train_positives, test_positives, validation_positives, proc_negatives, n=5, m=10)

def process_markov_data():
    df = load_data()

    g = build_full_graph(df)
    negatives = get_negatives_from_full_graph(g)

    ref_g, train_set, test_set, validation_set = build_connected_ref_graph(g, df, seed=42)

    train_positives = create_markov_set(ref_g, train_set)
    test_positives = create_markov_set(ref_g, test_set)
    validation_positives = create_markov_set(ref_g, validation_set)
    proc_negatives = create_markov_set(ref_g, negatives)   

    create_files(train_positives, test_positives, validation_positives, proc_negatives, n=5, m=10, feats_group="markov")

def process_kernel_data():
    df = load_data()

    g = build_full_graph(df)
    negatives = get_negatives_from_full_graph(g)

    ref_g, train_set, test_set, validation_set = build_connected_ref_graph(g, df, seed=42)

    train_positives = create_kernel_set(ref_g, train_set)
    test_positives = create_kernel_set(ref_g, test_set)
    validation_positives = create_kernel_set(ref_g, validation_set)
    proc_negatives = create_kernel_set(ref_g, negatives)   

    create_files(train_positives, test_positives, validation_positives, proc_negatives, n=5, m=10, feats_group="kernel")

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
            process_baseline_data()
        case "markov":
            process_markov_data()
        case "kernel":
            process_kernel_data()
        case _:
            print("Invalid argument")

if __name__ == "__main__":
    main()
