"""GELU as ONE ONNX node, for the models whose MLP uses the exact (erf) GELU.

torch exports nn.GELU at opset 13 as the loose chain Div > Erf > Add > Mul >
Mul. In a quantized export the activation x feeding that chain is the output
of a DequantizeLinear, consumed twice (by the Div and by the final Mul).
ONNX Runtime then duplicates that DequantizeLinear (its
EnsureUniqueDQForNodeUnit transformer), so the two ends of the chain read
different tensors and its GeluFusion no longer matches: the 12 GELUs of
DeiT-Tiny stay as loose Erf nodes, which do not scale with threads
(investigations/ort_graph_gelu.py). In FP32 the same chain IS fused by ORT.

replace_gelu_chains rewrites each chain as a single com.microsoft::Gelu node --
the exact erf GELU, the very op ORT's fusion produces in FP32 -- reading the
same x and writing the same output tensor. The Q/DQ nodes, their scales and
everything else are untouched: only how the GELU is written in the graph
changes. The formula is identical; a float32 kernel may differ from the five
separate ops in the last bit (ORT's Gelu multiplies by 1/sqrt(2) where the
chain divides by sqrt(2)), which is why the accuracy of a regenerated artifact
is re-measured rather than assumed.

Needs only `onnx` (no torch, no onnxruntime).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

CHAIN = ("Div", "Erf", "Add", "Mul", "Mul")


def index_graph(graph) -> Tuple[Dict, Dict]:
    producer, consumers = {}, defaultdict(list)
    for node in graph.node:
        for name in node.output:
            producer[name] = node
        for name in node.input:
            if name:
                consumers[name].append(node)
    return producer, consumers


def _find_chains(graph) -> List[list]:
    producer, consumers = index_graph(graph)
    chains = []
    for erf in (n for n in graph.node if n.op_type == "Erf"):
        chain = [producer.get(erf.input[0]), erf]
        node = erf
        for _ in range(3):
            following = consumers.get(node.output[0], [])
            if len(following) != 1:
                break
            node = following[0]
            chain.append(node)
        if len(chain) == 5 and all(n is not None for n in chain) and tuple(n.op_type for n in chain) == CHAIN:
            chains.append(chain)
    return chains


def replace_gelu_chains(src: Path, dst: Path) -> int:
    """Writes `dst`: `src` with every Div>Erf>Add>Mul>Mul chain replaced by one
    com.microsoft::Gelu node. Returns how many chains were replaced (0: `dst`
    is an unchanged copy)."""
    import onnx
    from onnx import helper

    model = onnx.load(str(src))
    graph = model.graph
    replaced = 0
    for div, erf, add, mul1, mul2 in _find_chains(graph):
        position = list(graph.node).index(mul2)
        gelu = helper.make_node("Gelu", [div.input[0]], [mul2.output[0]], name=f"gelu_{replaced}",
                                domain="com.microsoft")
        removed = [div, erf, add, mul1, mul2]
        for node in removed:
            graph.node.remove(node)
        graph.node.insert(position - 4, gelu)
        # the chain's own Constant nodes (sqrt(2), 1, 0.5) are now unused
        _, consumers = index_graph(graph)
        outputs = {o.name for o in graph.output}
        inputs = {i for node in removed for i in node.input}
        for node in list(graph.node):
            if node.op_type == "Constant" and node.output[0] in inputs and not consumers.get(node.output[0]) \
                    and node.output[0] not in outputs:
                graph.node.remove(node)
        replaced += 1
    if replaced and not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    return replaced


def fuse_gelu_in_place(path: Path) -> int:
    """replace_gelu_chains on an exported artifact, strictly: at least one chain
    must be replaced and no loose Erf may remain, else it raises (an artifact
    meant to carry single-node GELUs must not silently keep the chains)."""
    import onnx

    path = Path(path)
    replaced = replace_gelu_chains(path, path)
    if replaced == 0:
        raise ValueError(f"{path.name}: no Div>Erf>Add>Mul>Mul GELU chain found -- single_gelu does not apply to this model")
    leftover = sum(1 for n in onnx.load(str(path)).graph.node if n.op_type == "Erf")
    if leftover:
        raise ValueError(f"{path.name}: {leftover} Erf node(s) remain outside a recognised GELU chain")
    return replaced
