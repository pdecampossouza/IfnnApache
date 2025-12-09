# experiments/kg.py
import re
import math
import networkx as nx
from typing import List, Optional
import matplotlib.pyplot as plt

_ANT_RE = re.compile(
    r"x(\d+)\s+is around\s+\[([^\]]+)\]\s+with sigma\s+([0-9.]+)", flags=re.IGNORECASE
)
_OUT_RE = re.compile(r"THEN output is \[([^\]]+)\]\.", flags=re.IGNORECASE)


def _parse_axiom(ax: str):
    """
    Retorna:
      antecedents: lista de dicts {feature_idx, interval_start, interval_end, sigma, center}
      consequent: float
    """
    ants = []
    for m in _ANT_RE.finditer(ax):
        feat_idx = int(m.group(1))  # 1-based
        a_b = m.group(2).strip()
        sigma = float(m.group(3))
        # a_b pode ser "a - b" ou "a-b"; normaliza
        if " - " in a_b:
            a_str, b_str = a_b.split(" - ")
        else:
            a_str, b_str = a_b.split("-")
        interval_start = float(a_str)
        interval_end = float(b_str)
        center = 0.5 * (interval_start + interval_end)
        ants.append(
            {
                "feature_idx": feat_idx,
                "interval_start": interval_start,
                "interval_end": interval_end,
                "sigma": sigma,
                "center": center,
            }
        )

    out_match = _OUT_RE.search(ax)
    if not out_match:
        raise ValueError(f"Não encontrei consequente em: {ax}")
    consequent = float(out_match.group(1))

    return ants, consequent


def build_fuzzy_axioms_kg(
    axioms: List[str],
    out_graphml_path: str,
    feature_names: Optional[List[str]] = None,
):
    """
    Constrói um DiGraph com:
      - nó de regra: Rule_i
      - nós de antecedentes: Ante_x{j}_c{center:.2f}_s{sigma:.2f}
      - nó de consequente: Out_{value:.2f}
    Arestas:
      Rule_i --hasAntecedent--> Ante_...
      Rule_i --hasConsequent--> Out_...

    Salva em GraphML.
    """
    G = nx.DiGraph()
    G.graph["name"] = "FuzzyAxiomsKG"

    for i, ax in enumerate(axioms, start=1):
        rule_id = f"Rule_{i}"
        G.add_node(rule_id, type="Rule", label=rule_id, text=ax)

        antecedents, cons = _parse_axiom(ax)

        # nó de consequente (reuso por valor)
        cons_key = f"Out_{cons:.2f}"
        if cons_key not in G:
            G.add_node(
                cons_key, type="Consequent", label=f"{cons:.2f}", value=float(cons)
            )
        G.add_edge(rule_id, cons_key, relation="hasConsequent")

        # antecedentes
        for ant in antecedents:
            j = ant["feature_idx"]  # 1-based
            fname = (
                feature_names[j - 1]
                if feature_names and (1 <= j <= len(feature_names))
                else f"x{j}"
            )
            node_key = f"Ante_{fname}_c{ant['center']:.2f}_s{ant['sigma']:.2f}"
            if node_key not in G:
                G.add_node(
                    node_key,
                    type="Antecedent",
                    label=f"{fname}≈{ant['center']:.2f} (σ={ant['sigma']:.2f})",
                    feature=fname,
                    feature_idx=int(j),
                    center=float(ant["center"]),
                    sigma=float(ant["sigma"]),
                    interval_start=float(ant["interval_start"]),
                    interval_end=float(ant["interval_end"]),
                )
            G.add_edge(rule_id, node_key, relation="hasAntecedent")

    nx.write_graphml(G, out_graphml_path)
    return out_graphml_path


def save_fuzzy_axioms_kg_png(graphml_path: str, out_png_path: str, seed: int = 42):
    """
    Lê o .graphml, colore nós por tipo e salva um PNG.
    """
    G = nx.read_graphml(graphml_path)

    # separar por tipo
    rules = [n for n, d in G.nodes(data=True) if d.get("type") == "Rule"]
    ants = [n for n, d in G.nodes(data=True) if d.get("type") == "Antecedent"]
    cons = [n for n, d in G.nodes(data=True) if d.get("type") == "Consequent"]

    # layout estável
    pos = nx.spring_layout(G, seed=seed, k=0.9)

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(
        G, pos, nodelist=ants, node_color="skyblue", node_size=500, label="Antecedents"
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=rules, node_color="lightgreen", node_size=650, label="Rules"
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=cons, node_color="salmon", node_size=550, label="Consequents"
    )
    nx.draw_networkx_edges(G, pos, alpha=0.35, arrows=True, arrowsize=12, width=1.2)
    # rótulos: usa 'label' se existir, senão o id
    labels = {n: (d.get("label") or n) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=2)
    plt.legend(scatterpoints=1, fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200)
    plt.close()
    return out_png_path
