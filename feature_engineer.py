import networkx as nx
import numpy as np
from scipy.sparse.linalg import expm
from scipy.sparse import csr_matrix, eye
import pandas as pd

class MarkovChain:
    def __init__(self, g):
        self.G = g
        self.prob_mat = csr_matrix(self.compute_prob_mat())

    def get_prob_mat(self):
        return self.prob_mat

    def compute_prob_mat(self):
        A = nx.adjacency_matrix(self.G)
        D_inv = np.diag([
            1 / self.G.degree(n) if self.G.degree(n) > 0 else 0 for n in self.G.nodes
        ])
        D_inv = csr_matrix(D_inv)

        return D_inv @ A

    def p_step_transition(self, p):
        if isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer)):
            raise ValueError("p parameter must be an integer")
        if p <= 0:
            raise ValueError("Cannot have number of steps bellow 1")

        if p == 1:
            return self.prob_mat
        else:
            temp = eye(m=self.prob_mat.shape[0], format="csr")
        for i in range(p):
            temp = self.prob_mat @ temp

        return temp

class InvCosineKernel:
    def __init__(self, g):
        self.G = g
        self.norm_L = self.compute_norm_laplacian()
        self.alpha = np.pi / 4

    def compute_norm_laplacian(self):
        return nx.normalized_laplacian_matrix(self.G)

    def compute_kernel(self):
        kernel = expm(1j * self.alpha * self.norm_L)

        return kernel.real

class PStepRandomWalkKernel:
    def __init__(self, g, a, p):
        self.G = g
        self.norm_L = self.compute_norm_laplacian()
        if (isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer))):
            raise ValueError("p must be an integer")
        if p <= 0:
            raise ValueError("p must be a positive number")

        self.p = p

        if a < 2:
            raise ValueError("a must be a positive number equal or greater than 2")

        self.a = a

    def compute_norm_laplacian(self):
        return nx.normalized_laplacian_matrix(self.G)

    def compute_kernel(self):
        base_mat = self.a * eye(m=len(self.G.nodes), format="csr") - self.norm_L
        kernel = eye(m=len(self.G.nodes), format="csr")

        for i in range(self.p):
            kernel = kernel @ base_mat

        return kernel

class DiffusionKernel:
    def __init__(self, g, sig):
        self.G = g
        self.sig_sqrd = sig ** 2
        self.norm_L = self.compute_norm_laplacian()

    def compute_norm_laplacian(self):
        return nx.normalized_laplacian_matrix(self.G)

    def compute_kernel(self):
        base_mat = -(self.sig_sqrd / 2) * self.norm_L
        kernel = expm(base_mat)

        return kernel

def compute_min_max_degrees_from_nodes(g, node_pairs):
    min_degs = []
    max_degs = []

    for idx in node_pairs.index:
        nodes = node_pairs.loc[idx].split(",")

        min_deg = np.min([g.degree(nodes[0]), g.degree(nodes[1])])
        max_deg = np.max([g.degree(nodes[0]), g.degree(nodes[1])])

        min_degs.append(min_deg)
        max_degs.append(max_deg)

    return min_degs, max_degs

def compute_common_neighbors_from_nodes(g, node_pairs):
    scores = []

    for edge in node_pairs:
        nodes = edge.split(",")

        common_neighbors_score = len(nx.common_neighbors(g, nodes[0], nodes[1]))
        scores.append(common_neighbors_score)

    return scores

def compute_jaccard_from_nodes(g, node_pairs):
    scores = []

    for edge in node_pairs:
        nodes = edge.split(",")
        nodes_tpl = (nodes[0], nodes[1])

        jaccard_coeff = nx.jaccard_coefficient(g, ebunch=[nodes_tpl])
        scores.append(list(jaccard_coeff)[0][2])

    return scores

def compute_adamic_adar_from_nodes(g, node_pairs):
    scores = []

    for edge in node_pairs:
        nodes = edge.split(",")
        nodes_tpl = (nodes[0], nodes[1])

        adamic_adar_idx = nx.adamic_adar_index(g, ebunch=[nodes_tpl])
        scores.append(list(adamic_adar_idx)[0][2])

    return scores

def compute_resource_allocation_from_nodes(g, node_pairs):
    scores = []

    for edge in node_pairs:
        nodes = edge.split(",")
        nodes_tpl = (nodes[0], nodes[1])

        resource_alloc_idx = nx.resource_allocation_index(g, ebunch=[nodes_tpl])
        scores.append(list(resource_alloc_idx)[0][2])

    return scores

def compute_shortest_path_from_nodes(g, node_pairs):
    paths = []

    for edge in node_pairs:
        nodes = edge.split(",")

        try:
            short_path = nx.shortest_path_length(g, source=nodes[0], target=nodes[1])
            paths.append(short_path)
        except nx.NetworkXNoPath:
            paths.append(-1)

    return paths

def get_markov_probs_from_nodes(g, node_pairs, steps=2):
    edges = []

    for nodes in node_pairs:
        edges.append(nodes.split(","))

    mc = MarkovChain(g)
    transitions_mat = mc.p_step_transition(steps).toarray()

    markov_df = pd.DataFrame(data=transitions_mat, columns=g.nodes, index=g.nodes)
    markov_probs = []

    for u, v in edges:
        p_uv = markov_df.loc[u, v]
        p_vu = markov_df.loc[v, u]

        markov_probs.append((p_uv + p_vu) / 2)

    return markov_probs

def get_inv_cos_scores(g, node_pairs):
    edges = []

    for nodes in node_pairs:
        edges.append(nodes.split(","))

    inv_cos_ker = InvCosineKernel(g)
    kernel_mat = inv_cos_ker.compute_kernel().toarray()

    kernel_df = pd.DataFrame(data=kernel_mat, columns=g.nodes, index=g.nodes)
    scores = []

    for e in edges:
        scores.append(kernel_df.loc[e[0], e[1]])

    return scores

def get_p_steps_rw_scores(g, node_pairs, a, p):
    edges = []

    for nodes in node_pairs:
        edges.append(nodes.split(","))

    p_step_ker = PStepRandomWalkKernel(g, a=a, p=p)
    kernel_mat = p_step_ker.compute_kernel().toarray()

    kernel_df = pd.DataFrame(data=kernel_mat, columns=g.nodes, index=g.nodes)
    scores = []

    for e in edges:
        scores.append(kernel_df.loc[e[0], e[1]])

    return scores

def get_diffusion_scores(g, node_pairs, sig):
    edges = []

    for nodes in node_pairs:
        edges.append(nodes.split(","))

    diff_ker = DiffusionKernel(g, sig=sig)
    kernel_mat = diff_ker.compute_kernel().toarray()

    kernel_df = pd.DataFrame(data=kernel_mat, columns=g.nodes, index=g.nodes)
    scores = []

    for e in edges:
        scores.append(kernel_df.loc[e[0], e[1]])

    return scores
