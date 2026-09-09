from pathlib import Path

from flowde.benchmarks.parsing.parsing_bench import ParsingBenchmark
from flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn import (
    levenshtein_with_nfc_and_space_normalisation,
)

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data/training-smoking-cessation/parsing"


def main() -> None:
    print("Validating data and matching diagrams...", flush=True)
    benchmark = ParsingBenchmark(
        pred_diagrams_dir=DATA_DIR / "pred/full-flowcharts-gemini31",
        distance_fn=levenshtein_with_nfc_and_space_normalisation,
        allow_missing_pred_diagrams=False,
        true_nodes_dir=DATA_DIR / "ground-truth/nodes",
        true_labels_dir=DATA_DIR / "ground-truth/labels",
        true_flow_dir=DATA_DIR / "ground-truth/flow",
        true_additional_texts_dir=DATA_DIR / "ground-truth/additional_texts",
    )
    diagram_matches = benchmark.diagram_matches()
    flow_scores = [match.flow_score for match in diagram_matches]
    n_diagrams = len(diagram_matches)

    total_node_cost = sum(match.total_node_text_cost for match in diagram_matches)
    total_label_cost = sum(match.total_label_error_cost for match in diagram_matches)
    total_additional_cost = sum(
        match.total_additional_text_cost for match in diagram_matches
    )
    total_tp = sum(score.tp for score in flow_scores)
    total_fp = sum(score.fp for score in flow_scores)
    total_fn = sum(score.fn for score in flow_scores)

    print("\nOVERALL RESULTS")
    print(f"Diagrams: {n_diagrams}")
    print(f"Total node-text cost: {total_node_cost:g}")
    print(f"Average node-text cost: {total_node_cost / n_diagrams:g}")
    print(
        "Perfect node-text diagrams: "
        f"{sum(match.total_node_text_cost == 0 for match in diagram_matches)}"
    )
    print(f"Total label cost: {total_label_cost:g}")
    print(f"Average label cost: {total_label_cost / n_diagrams:g}")
    print(
        "Perfect-label diagrams: "
        f"{sum(match.total_label_error_cost == 0 for match in diagram_matches)}"
    )
    print(f"Total additional-text cost: {total_additional_cost:g}")
    print(f"Average additional-text cost: {total_additional_cost / n_diagrams:g}")
    print(
        "Perfect additional-text diagrams: "
        f"{sum(match.total_additional_text_cost == 0 for match in diagram_matches)}"
    )
    print(f"Flow true positives: {total_tp}")
    print(f"Flow false positives: {total_fp}")
    print(f"Flow false negatives: {total_fn}")
    print(f"Micro flow Jaccard: {total_tp / (total_tp + total_fp + total_fn):g}")
    print(
        "Average diagram flow Jaccard: "
        f"{sum(score.jaccard for score in flow_scores) / n_diagrams:g}"
    )

    imperfect_diagrams = [
        match
        for match in diagram_matches
        if match.total_text_cost != 0 or match.flow_score.jaccard != 1
    ]
    print(f"\nIMPERFECT DIAGRAMS: {len(imperfect_diagrams)}")

    for diagram_match in sorted(
        imperfect_diagrams,
        key=lambda match: match.pred_diagram.parent_img_code,
    ):
        flow_score = diagram_match.flow_score
        print("\n" + "=" * 80)
        print(f"Diagram: {diagram_match.pred_diagram.parent_img_code}")
        print(
            "Selected ground-truth option: "
            f"{diagram_match.node_matches.true_diagram_option_idx}"
        )
        print(f"Node-text cost: {diagram_match.total_node_text_cost:g}")
        print(f"Label cost: {diagram_match.total_label_error_cost:g}")
        print(f"Additional-text cost: {diagram_match.total_additional_text_cost:g}")
        print(
            f"Flow: Jaccard={flow_score.jaccard:g}, "
            f"TP={flow_score.tp}, FP={flow_score.fp}, FN={flow_score.fn}"
        )

        for node_match in diagram_match.node_matches.matches:
            if node_match.node_text_cost != 0:
                true_number = (
                    node_match.true_node.node_number
                    if node_match.true_node is not None
                    else None
                )
                pred_number = (
                    node_match.pred_node.node_number
                    if node_match.pred_node is not None
                    else None
                )
                true_text = (
                    node_match.true_node.text
                    if node_match.true_node is not None
                    else None
                )
                pred_text = (
                    node_match.pred_node.text
                    if node_match.pred_node is not None
                    else None
                )
                print(
                    f"  Node: true #{true_number} {true_text!r} -> "
                    f"pred #{pred_number} {pred_text!r}; "
                    f"cost={node_match.node_text_cost:g}"
                )

            label_matches = node_match.label_matches
            if label_matches is not None:
                for label_match in label_matches.matches:
                    if label_match.cost != 0:
                        print(
                            f"  Label: {label_match.true_text!r} -> "
                            f"{label_match.pred_text!r}; cost={label_match.cost:g}"
                        )

        for text_match in diagram_match.additional_text_matches.matches:
            if text_match.cost != 0:
                print(
                    f"  Additional text: {text_match.true_text!r} -> "
                    f"{text_match.pred_text!r}; cost={text_match.cost:g}"
                )

        for edge in sorted(flow_score.missing_edges):
            print(f"  Missing edge: {edge}")
        for edge in sorted(flow_score.extra_edges):
            print(f"  Extra edge after node matching: {edge}")


if __name__ == "__main__":
    main()
