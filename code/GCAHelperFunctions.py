def profile_function(adata, func, N_values, suppress_output=False, *args, **kwargs):
    import time, tracemalloc, matplotlib.pyplot as plt, numpy as np
    import contextlib, io
    N_sorted = sorted(N_values, reverse=True)
    times = {}
    mems = {}
    largest = N_sorted[0]
    idx = np.random.choice(adata.shape[0], largest, replace=False)
    current_subset = adata[idx,:]
    for N in N_sorted:
        if current_subset.shape[0] > N:
            current_subset = current_subset[np.random.choice(current_subset.shape[0], N, replace=False),:]
        tracemalloc.start()
        start = time.perf_counter()
        if suppress_output:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                func(current_subset, *args, **kwargs)
        else:
            func(current_subset, *args, **kwargs)
        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times[N] = elapsed
        mems[N] = peak / 1024
    return times, mems

def plot_profile(time, mems): 
    sorted_N = sorted(times.keys())
    time_vals = [times[n] for n in sorted_N]
    mem_vals = [mems[n] for n in sorted_N]
    fig1 = plt.figure()
    plt.plot(sorted_N, time_vals)
    plt.xlabel('N')
    plt.ylabel('Time (s)')
    plt.title('Time vs N')
    fig2 = plt.figure()
    plt.plot(sorted_N, mem_vals)
    plt.xlabel('N')
    plt.ylabel('Peak Memory (KB)')
    plt.title('Memory vs N')
    plt.show()

import obonet
import requests
from io import StringIO
import pandas as pd

# Download the latest CL ontology (only once needed)
url = 'http://purl.obolibrary.org/obo/cl/cl-basic.obo'
response = requests.get(url)
graph = obonet.read_obo(StringIO(response.text))

# Make a mapping from CL ID to name
cl_to_name = {
    node_id: data.get('name')
    for node_id, data in graph.nodes(data=True)
    if node_id.startswith('CL:')
}


class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = {}
        self.metadata = {}

    def add_child(self, node_name):
        if node_name not in self.children:
            self.children[node_name] = TreeNode(node_name)
        return self.children[node_name]

def build_tree_from_taxonomy(
    csv_path: str,
    resolution_cols: list[str],
    annotation_cols: list[str] | None = None
) -> TreeNode:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    # Download the latest CL ontology (only once needed)
    url = 'http://purl.obolibrary.org/obo/cl/cl-basic.obo'
    response = requests.get(url)
    graph = obonet.read_obo(StringIO(response.text))

    # Make a mapping from CL ID to name
    cl_to_name = {
        node_id: data.get('name')
        for node_id, data in graph.nodes(data=True)
        if node_id.startswith('CL:')
    }
    df['harmonized_author_ontology_labels'] = (
        df['cell_type_ontology_term_id']
          .map(cl_to_name)
          .fillna(df['cell_type_ontology_term_id'])
    )
    if annotation_cols is None:
        annotation_cols = [c for c in df.columns if c not in resolution_cols]
    root = TreeNode("root")
    for _, row in df.iterrows():
        node = root
        for col in resolution_cols:
            val = row[col]
            if not val:
                break
            node = node.add_child(val)
        for col in annotation_cols:
            node.metadata[col] = row[col]
    return root

def flatten(
    node: TreeNode,
    order: list[str] | None = None,
    depths: list[int] | None = None,
    depth: int = 0,
    name_key: str | None = None
) -> tuple[list[str], list[int]]:
    if order is None:
        order, depths = [], []
    # choose metadata name if requested
    label = node.metadata.get(name_key, node.name) if name_key else node.name
    order.append(label)
    depths.append(depth)
    for child in node.children.values():
        flatten(child, order, depths, depth + 1, name_key)
    return order, depths

import matplotlib.pyplot as plt

def pretty_print(node, indent=0):
    """Recursively print out node names with indentation."""
    print("  " * indent + node.name)
    for child in node.children.values():
        pretty_print(child, indent + 1)


def plot_tree(node):
    """
    Assign each node an (x,y) based on depth and leaf order,
    then draw lines + labels in matplotlib.
    """
    # 1) Compute positions
    positions = {}
    def _assign(n, depth=0, counter=[0]):
        # if leaf, give it the next y
        if not n.children:
            y = counter[0]
            positions[n] = (depth, y)
            counter[0] += 1
        else:
            # assign positions for children first
            child_ys = []
            for c in n.children.values():
                _assign(c, depth + 1, counter)
                child_ys.append(positions[c][1])
            # place this node midway between its children
            positions[n] = (depth, sum(child_ys) / len(child_ys))
    _assign(node)

    # 2) Plot edges
    fig, ax = plt.subplots()
    for parent, (x1, y1) in positions.items():
        for child in parent.children.values():
            x2, y2 = positions[child]
            ax.plot([x1, x2], [y1, y2])  # default color

    # 3) Plot labels
    for n, (x, y) in positions.items():
        ax.text(x, y, n.name, va="center", ha="right" if n.children else "left",
                fontsize=8)

    ax.invert_yaxis()   # so top of tree is at the top
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_heatmaps_by_clade(
    df_cm: pd.DataFrame,
    tree,
    name_key: str,
    clade_level: int,
    ncols: int = 3,
    figsize_per: float = 4,
    fontsize: int = 6
):
    # 1) flatten with full paths
    records = []
    def _recurse(node, depth=0, path=None):
        if path is None: path = []
        label = node.metadata.get(name_key, node.name) if name_key else node.name
        cur_path = path + [label]
        records.append({'label': label, 'depth': depth, 'path': cur_path})
        for child in node.children.values():
            _recurse(child, depth+1, cur_path)
    _recurse(tree)

    df_rec = pd.DataFrame(records)
    # keep only those in the matrix
    df_rec = df_rec[df_rec['label'].isin(df_cm.index)]

    # 2) assign each node to its clade at clade_level
    # clade_level=1 → top‐level (resolution_cols[0]), etc.
    def _clade(path):
        return path[clade_level] if len(path) > clade_level else None
    df_rec['clade'] = df_rec['path'].apply(_clade)

    # 3) plot one heatmap per clade
    clades = [c for c in df_rec['clade'].unique() if c is not None]
    n = len(clades)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*figsize_per, nrows*figsize_per),
                             squeeze=False)

    for ax, clade in zip(axes.flat, clades):
        labels = df_rec.loc[df_rec['clade']==clade, 'label'].tolist()
        sub = df_cm.loc[labels, labels]
        sns.heatmap(sub, annot=True, fmt=".1f", cmap="Blues",
                    cbar=False, xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"Clade: {clade}", fontsize=fontsize+2)
        ax.tick_params(axis="x", labelrotation=90, labelsize=fontsize)
        ax.tick_params(axis="y", labelrotation=0, labelsize=fontsize)

    # turn off any unused axes
    for ax in axes.flat[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmaps_by_clade_iter(
    df_cm_pct: pd.DataFrame,
    df_counts: pd.DataFrame,
    tree,
    name_key: str,
    clade_level: int,
    fontsize: int = 6,
    size_per_label: float = 0.3,
):
    # 1) Flatten tree into records
    records = []
    def _recurse(n, depth=0, path=None):
        path = (path or []) + [n.metadata.get(name_key, n.name)]
        records.append({'label': path[-1], 'depth': depth, 'path': path})
        for c in n.children.values():
            _recurse(c, depth+1, path)
    _recurse(tree)
    df_rec = pd.DataFrame(records)
    df_rec = df_rec[df_rec['label'].isin(df_cm_pct.index)]
    df_rec['clade'] = df_rec['path'].apply(
        lambda p: p[clade_level] if len(p) > clade_level else None
    )
    clades = [c for c in df_rec['clade'].unique() if c]

    # 2) One full-width plot per clade
    for clade in clades:
        sub_rec = df_rec[df_rec['clade'] == clade]

        # dedupe in tree order
        seen = set()
        labels, depths = [], []
        for _, row in sub_rec.iterrows():
            lbl, dep = row['label'], row['depth']
            if lbl in seen:
                continue
            seen.add(lbl)
            labels.append(lbl)
            depths.append(dep)

        # extract the sub-matrices
        sub_pct    = df_cm_pct.loc[labels, labels].values
        sub_counts = df_counts.loc[labels, labels].values
        n = len(labels)

        # compute clade accuracy = sum(diag) / sum(all true)
        true_pos = np.diag(sub_counts).sum()
        total_true = sub_counts.sum(axis=1).sum()
        acc = true_pos / total_true * 100 if total_true > 0 else 0.0

        # compute internal boundaries for this subset
        boundaries = []
        for i in range(1, n):
            if depths[i] <= depths[i-1]:
                boundaries.append(i)
        boundaries.append(n)

        # mask NaNs so they render blank
        m_sub = np.ma.masked_invalid(sub_pct)
        cmap = plt.cm.Blues.copy()
        cmap.set_bad(color='white')

        fig, ax = plt.subplots(
            figsize=(n * size_per_label + 4, n * size_per_label + 4)
        )
        im = ax.imshow(
            m_sub, interpolation='none', aspect='equal', origin='upper'
        )

        # annotate only real values
        for i in range(n):
            for j in range(n):
                val = sub_pct[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}",
                            ha='center', va='center',
                            fontsize=fontsize)

        # tick labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
        ax.set_yticklabels(labels, fontsize=fontsize)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)

        # draw internal-level boundaries
        for b in boundaries:
            ax.axhline(b - 0.5, color='white', lw=1)
            ax.axvline(b - 0.5, color='white', lw=1)

        # colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label("% of true label", fontsize=fontsize)

        # title with accuracy
        ax.set_title(
            f"Clade: {clade}   (Accuracy: {acc:.1f} %)",
            fontsize=fontsize + 2
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        plt.tight_layout()
        plt.show()
