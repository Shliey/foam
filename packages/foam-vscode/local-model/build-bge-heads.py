"""Export BAAI's projection heads and candidate MaxSim as a small ONNX graph."""
import argparse
import hashlib
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

SOURCES = {
    "sparse": "45c93804d2142b8f6d7ec6914ae23a1eee9c6a1d27d83d908a20d2afb3595ad9",
    "colbert": "19bfbae397c2b7524158c919d0e9b19393c5639d098f0a66932c91ed8f5f9abb",
}


def value(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def save(graph, output):
    model = helper.make_model(
        graph, producer_name="Foam",
        opset_imports=[helper.make_opsetid("", 17)], ir_version=8,
    )
    onnx.checker.check_model(model)
    onnx.save(model, output)


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("source", type=Path, help="Directory with the pinned BAAI .pt files")
    args.add_argument("output", type=Path)
    options = args.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)

    for name, digest in SOURCES.items():
        path = options.source / f"{name}_linear.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unexpected SHA-256 for {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        width = 1 if name == "sparse" else 1024
        if tuple(state["weight"].shape) != (width, 1024) or tuple(state["bias"].shape) != (width,):
            raise ValueError(f"Unexpected projection shape for {path}")
        nodes = [
            helper.make_node("MatMul", ["hidden", "weight"], ["projected"]),
            helper.make_node("Add", ["projected", "bias"], ["logits"]),
            helper.make_node("Relu" if name == "sparse" else "Identity", ["logits"], ["output"]),
        ]
        save(helper.make_graph(
            nodes, f"BAAI-bge-m3-{name}", [value("hidden", ["tokens", 1024])],
            [value("output", ["tokens", width])],
            [
                numpy_helper.from_array(state["weight"].float().transpose(0, 1).contiguous().numpy(), "weight"),
                numpy_helper.from_array(state["bias"].float().numpy(), "bias"),
            ],
        ), options.output / f"bge-{name}.onnx")

    save(helper.make_graph([
        helper.make_node("Transpose", ["passage"], ["transposed"], perm=[1, 0]),
        helper.make_node("MatMul", ["query", "transposed"], ["similarities"]),
        helper.make_node("ReduceMax", ["similarities"], ["best"], axes=[1], keepdims=0),
        helper.make_node("ReduceMean", ["best"], ["score"], axes=[0], keepdims=0),
    ], "BAAI-bge-m3-maxsim",
        [value("query", ["query_tokens", 1024]), value("passage", ["passage_tokens", 1024])],
        [value("score", [])],
    ), options.output / "bge-maxsim.onnx")


if __name__ == "__main__":
    main()
