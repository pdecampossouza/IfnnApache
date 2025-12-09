# experiments/kgfuzzyrules.py
import re
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch

# -------- parsing das strings --------
_ANT_RE = re.compile(
    r"x(\d+)\s+is\s+(MF\d+)\s+with impact\s+([0-9.]+)",
    flags=re.IGNORECASE,
)
_OUT_RE = re.compile(r"THEN output is \[([^\]]+)\]", flags=re.IGNORECASE)


def build_fuzzy_rules_kg(rule_lines: List[str], out_graphml_path: str) -> str:
    """
    Constrói o KG e anota:
      - Rule: rule_score ([-1,1]), rule_class (+1 ou -1), confidence=abs(score)
      - Consequent: value, class, confidence
    """
    G = nx.DiGraph()
    G.graph["name"] = "FuzzyRulesKG"

    for i, line in enumerate(rule_lines, start=1):
        rnode = f"Rule_{i}"
        G.add_node(rnode, type="Rule", label=rnode, text=line)

        # antecedentes (terms) com impact
        for m in _ANT_RE.finditer(line):
            feat_idx = int(m.group(1))
            mf_name = m.group(2).upper()  # MF1, MF2, ...
            impact = float(m.group(3))

            term_key = f"Term_x{feat_idx}_{mf_name}"
            if term_key not in G:
                G.add_node(
                    term_key,
                    type="Term",
                    label=f"x{feat_idx}·{mf_name}",
                    feature=f"x{feat_idx}",
                    mf=mf_name,
                )
            G.add_edge(rnode, term_key, relation="hasAntecedent", impact=impact)

        # consequente e atributos de classe/confiança
        out_m = _OUT_RE.search(line)
        if out_m:
            y = float(out_m.group(1))  # [-1, 1]
            cls = 1 if y >= 0 else -1
            conf = abs(y)
            out_key = f"Out_{y:.2f}"
            if out_key not in G:
                G.add_node(
                    out_key,
                    type="Consequent",
                    label=f"{y:.2f}",
                    value=y,
                    class_label=cls,
                    confidence=conf,
                )
            G.add_edge(rnode, out_key, relation="hasConsequent")

            # anotar também no nó da regra
            G.nodes[rnode]["rule_score"] = y
            G.nodes[rnode]["rule_class"] = cls
            G.nodes[rnode]["confidence"] = conf

    nx.write_graphml(G, out_graphml_path)
    return out_graphml_path


# -------- visualização “paper-ready” --------


def _default_mf_color_map(mf_labels: Iterable[str]) -> Dict[str, str]:
    """
    Gera um dicionário MF -> cor (hex), color-blind friendly.
    Usa uma paleta fixa para MF1..MF8; acima disso, cai no tab20.
    """
    base = [
        "#4C72B0",  # MF1  - blue
        "#55A868",  # MF2  - green
        "#C44E52",  # MF3  - red
        "#8172B3",  # MF4  - purple
        "#CCB974",  # MF5  - brown/gold
        "#64B5CD",  # MF6  - cyan
        "#8C8C8C",  # MF7  - gray
        "#E17C05",  # MF8  - orange
    ]
    mf_labels = list(dict.fromkeys(mf_labels))  # unique & keep order
    cmap: Dict[str, str] = {}
    for i, mf in enumerate(mf_labels):
        if i < len(base):
            cmap[mf] = base[i]
        else:
            # fallback: use tab20 cycling
            color = plt.get_cmap("tab20")(i % 20)
            cmap[mf] = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
    return cmap


def save_fuzzy_rules_kg_png(
    graphml_path: str,
    out_png_path: str,
    seed: int = 42,
    mf_color_map: Optional[Dict[str, str]] = None,
    draw_edge_labels: bool = False,
) -> str:
    """
    - Terms: cor por MF (MF1, MF2, ...)
    - Rule & Consequent: cor por classe do consequente (diverging -1..+1)
      * azul = classe -1; vermelho = classe +1; claro = baixa confiança (|score|~0)
    - Aresta Rule->Term: largura ~ impact (0..1)
    """
    G = nx.read_graphml(graphml_path)

    rules = [n for n, d in G.nodes(data=True) if d.get("type") == "Rule"]
    terms = [n for n, d in G.nodes(data=True) if d.get("type") == "Term"]
    cons = [n for n, d in G.nodes(data=True) if d.get("type") == "Consequent"]

    # MF -> cor
    mfs_used = []
    for n in terms:
        mfs_used.append(str(G.nodes[n].get("mf")))
    mfs_used = list(dict.fromkeys(mfs_used))
    if mf_color_map is None:
        mf_color_map = _default_mf_color_map(mfs_used)

    # colormap para classe do consequente: azul(-1) ↔ branco(0) ↔ vermelho(+1)
    import numpy as np
    from matplotlib import cm

    cmap = cm.get_cmap("bwr")  # blue-white-red

    def score_to_color(y: float) -> str:
        # y em [-1,1] -> mapeia para 0..1
        t = 0.5 * (y + 1.0)
        r, g, b, _ = cmap(t)
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    # cores dos nós
    rule_colors = []
    for n in rules:
        y = float(G.nodes[n].get("rule_score", 0.0))
        rule_colors.append(score_to_color(y))

    cons_colors = []
    for n in cons:
        y = float(G.nodes[n].get("value", 0.0))
        cons_colors.append(score_to_color(y))

    term_colors = [
        mf_color_map.get(str(G.nodes[n].get("mf")), "#999999") for n in terms
    ]

    # layout
    pos = nx.kamada_kawai_layout(G)

    plt.figure(figsize=(10, 10))

    # desenhar nós
    nx.draw_networkx_nodes(
        G, pos, nodelist=terms, node_color=term_colors, node_size=560, label="Terms"
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=rules,
        node_color=rule_colors,
        node_size=740,
        label="Rules (colored by class)",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=cons,
        node_color=cons_colors,
        node_size=640,
        label="Consequents (colored by class)",
    )

    # arestas com espessura ~ impact
    widths = []
    edge_labels = {}
    edge_list = list(G.edges(data=True))
    for u, v, d in edge_list:
        if d.get("relation") == "hasAntecedent":
            w = float(d.get("impact", 0.0))
            widths.append(0.6 + 3.8 * w)  # 0.6..4.4
            if draw_edge_labels:
                edge_labels[(u, v)] = f"{w:.2f}"
        else:
            widths.append(1.2)

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v) for (u, v, _) in edge_list],
        width=widths,
        alpha=0.32,
        arrows=True,
        connectionstyle="arc3,rad=0.08",
    )

    # rótulos
    labels = {n: (G.nodes[n].get("label") or n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    if draw_edge_labels and edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # legenda: swatches para MFs + classes
    from matplotlib.patches import Patch

    mf_handles = [
        Patch(facecolor=mf_color_map[mf], edgecolor="none", label=mf) for mf in mfs_used
    ]

    class_handles = [
        Patch(
            facecolor=score_to_color(-1.0), edgecolor="none", label="Class −1 (strong)"
        ),
        Patch(facecolor=score_to_color(0.0), edgecolor="none", label="Uncertain (≈0)"),
        Patch(
            facecolor=score_to_color(+1.0), edgecolor="none", label="Class +1 (strong)"
        ),
    ]
    legend_handles = mf_handles + class_handles

    plt.legend(
        handles=legend_handles,
        title="MF Colors & Class (consequent score)",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # margem inferior para a legenda
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_png_path
