from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    OptimizeResult,
    linear_sum_assignment,
    milp,
)
from scipy.sparse import coo_array
from tqdm.auto import tqdm

from flowde.benchmarks.parsing.parsing_bench_types import (
    Diagram,
    DiagramMatch,
    DiagramOptions,
    Node,
    NodeMatches,
    TextListMatch,
    TextListMatches,
)
from flowde.benchmarks.parsing.text_distance_fns.distance_fn_protocol import (
    DistanceFnProtocol,
)

# TODO: Do we need to add any normalisation so that longer options aren't less
# favoured?

_BINARY_SELECTED_THRESHOLD = 0.5


def match_nodes(
    true_diagram_options: DiagramOptions,
    pred_diagram: Diagram,
    distance_fn: DistanceFnProtocol,
) -> NodeMatches:
    option_matches = [
        match_nodes_single_option(
            true_diagram=true_diagram,
            pred_diagram=pred_diagram,
            distance_fn=distance_fn,
            true_diagram_option_idx=i,
        )
        for i, true_diagram in enumerate(true_diagram_options.options)
    ]

    option_scores = [matches.total_node_text_cost for matches in option_matches]

    best_node_score = min(option_scores)
    candidate_option_indices = [
        i for i, score in enumerate(option_scores) if score == best_node_score
    ]

    if len(candidate_option_indices) > 1 and pred_diagram.flow_done:
        flow_scores = {
            i: option_matches[i].flow_score.jaccard for i in candidate_option_indices
        }
        best_flow_score = max(flow_scores.values())
        candidate_option_indices = [
            i for i in candidate_option_indices if flow_scores[i] == best_flow_score
        ]

    if len(candidate_option_indices) > 1 and pred_diagram.labels_done:
        label_scores = {}
        for i in candidate_option_indices:
            match_node_labels(node_matches=option_matches[i], distance_fn=distance_fn)
            label_scores[i] = option_matches[i].total_label_error_cost

        best_label_score = min(label_scores.values())
        candidate_option_indices = [
            i for i in candidate_option_indices if label_scores[i] == best_label_score
        ]

    if len(candidate_option_indices) > 1 and pred_diagram.additional_texts_done:
        additional_text_scores = {
            i: match_additional_texts(
                true_diagram=true_diagram_options.options[i],
                pred_diagram=pred_diagram,
                distance_fn=distance_fn,
            ).total_cost
            for i in candidate_option_indices
        }
        best_additional_text_score = min(additional_text_scores.values())
        candidate_option_indices = [
            i
            for i in candidate_option_indices
            if additional_text_scores[i] == best_additional_text_score
        ]

    return option_matches[candidate_option_indices[0]]


def match_nodes_single_option(
    true_diagram: Diagram,
    pred_diagram: Diagram,
    distance_fn: DistanceFnProtocol,
    true_diagram_option_idx: int,
) -> NodeMatches:
    if true_diagram.nodes is None or pred_diagram.nodes is None:
        msg = "Both diagrams must have nodes before matching nodes."
        raise ValueError(msg)

    if not true_diagram.text_done or not pred_diagram.text_done:
        msg = "Both diagrams must have text_done=True before matching nodes."
        raise ValueError(msg)

    true_nodes = true_diagram.nodes
    pred_nodes = pred_diagram.nodes

    node_matches = NodeMatches(
        matches=[],
        true_diagram_option_idx=true_diagram_option_idx,
    )

    n_true = len(true_nodes)
    n_pred = len(pred_nodes)

    if n_true == 0 and n_pred == 0:
        return node_matches

    text_costs = _build_node_text_costs(
        true_nodes=true_nodes,
        pred_nodes=pred_nodes,
        distance_fn=distance_fn,
    )
    row_ind, col_ind, minimum_text_cost = _minimum_text_assignment(text_costs)

    if (
        n_true == 0
        or n_pred == 0
        or (not pred_diagram.flow_done and not pred_diagram.labels_done)
    ):
        return _node_matches_from_linear_assignment(
            true_nodes=true_nodes,
            pred_nodes=pred_nodes,
            text_costs=text_costs,
            row_ind=row_ind,
            col_ind=col_ind,
            true_diagram_option_idx=true_diagram_option_idx,
        )

    return _optimise_text_optimal_node_matches(
        true_diagram=true_diagram,
        pred_diagram=pred_diagram,
        distance_fn=distance_fn,
        text_costs=text_costs,
        minimum_text_cost=minimum_text_cost,
        fallback_assignment=(row_ind, col_ind),
        true_diagram_option_idx=true_diagram_option_idx,
    )


def match_texts_lists(
    true_texts: list[str],
    pred_texts: list[str],
    distance_fn: DistanceFnProtocol,
) -> TextListMatches:
    n_true = len(true_texts)
    n_pred = len(pred_texts)

    if n_true == 0 and n_pred == 0:
        return TextListMatches(matches=[])

    size = n_true + n_pred
    BIG = 10**9

    cost = np.full((size, size), BIG, dtype=np.float64)

    # Real true -> real pred : Levenshtein distance
    for i, t_text in enumerate(true_texts):
        for j, p_text in enumerate(pred_texts):
            cost[i, j] = distance_fn(true_text=t_text, pred_text=p_text)

    # Real true -> its own dummy column : unmatched true cost
    for i, t_text in enumerate(true_texts):
        dummy_col = n_pred + i
        cost[i, dummy_col] = distance_fn(true_text=t_text, pred_text=None)

    # Its own dummy row -> real pred : unmatched pred cost
    for j, p_text in enumerate(pred_texts):
        dummy_row = n_true + j
        cost[dummy_row, j] = distance_fn(true_text=None, pred_text=p_text)

    # Dummy row <-> dummy col block : 0
    # This lets unused dummy rows/cols pair off harmlessly
    for j in range(n_pred):
        dummy_row = n_true + j
        for i in range(n_true):
            dummy_col = n_pred + i
            cost[dummy_row, dummy_col] = 0

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []

    for r, c in zip(row_ind, col_ind, strict=False):
        entry_cost = float(cost[r, c])

        # Real true matched to real pred
        if r < n_true and c < n_pred:
            matches.append(
                TextListMatch(
                    true_index=int(r),
                    pred_index=int(c),
                    true_text=true_texts[r],
                    pred_text=pred_texts[c],
                    cost=entry_cost,
                )
            )

        # Real true matched to dummy col => unmatched true
        elif r < n_true and c >= n_pred:
            matches.append(
                TextListMatch(
                    true_index=int(r),
                    pred_index=None,
                    true_text=true_texts[r],
                    pred_text=None,
                    cost=entry_cost,
                )
            )

        # Dummy row matched to real pred => unmatched pred
        elif r >= n_true and c < n_pred:
            matches.append(
                TextListMatch(
                    true_index=None,
                    pred_index=int(c),
                    true_text=None,
                    pred_text=pred_texts[c],
                    cost=entry_cost,
                )
            )

        # Dummy row to dummy col: ignore
        else:
            pass

    return TextListMatches(matches=matches)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class _NodeTextCosts:
    paired: FloatArray
    unmatched_true: FloatArray
    unmatched_pred: FloatArray


@dataclass(frozen=True)
class _MilpVariableLayout:
    """
    Map each semantic MILP decision variable to its decision-vector index.

    Each index also identifies the variable's objective-vector entry,
    constraint-matrix column, and position in the optimizer's returned solution.
    """

    n_true: int
    n_pred: int
    n_true_edges: int
    n_pred_edges: int

    @property
    def n_node_match_slots(self) -> int:
        return self.n_true * self.n_pred

    @property
    def unmatched_true_nodes_start_idx(self) -> int:
        return self.n_node_match_slots

    @property
    def unmatched_pred_nodes_start_idx(self) -> int:
        return self.unmatched_true_nodes_start_idx + self.n_true

    @property
    def edge_matches_start_idx(self) -> int:
        return self.unmatched_pred_nodes_start_idx + self.n_pred

    @property
    def n_edge_match_slots(self) -> int:
        return self.n_true_edges * self.n_pred_edges

    @property
    def n_variables(self) -> int:
        return self.edge_matches_start_idx + self.n_edge_match_slots

    def node_match_idx(self, true_node_idx: int, pred_node_idx: int) -> int:
        return true_node_idx * self.n_pred + pred_node_idx

    def unmatched_true_node_idx(self, true_node_idx: int) -> int:
        return self.unmatched_true_nodes_start_idx + true_node_idx

    def unmatched_pred_node_idx(self, pred_node_idx: int) -> int:
        return self.unmatched_pred_nodes_start_idx + pred_node_idx

    def edge_match_idx(self, true_edge_idx: int, pred_edge_idx: int) -> int:
        return (
            self.edge_matches_start_idx
            + true_edge_idx * self.n_pred_edges
            + pred_edge_idx
        )


class _LinearConstraintBuilder:
    """
    Collect constraint rows before building SciPy's sparse matrix.

    For a decision-variable vector `x`, each call to `add()` represents one row
    of `lower_bound <= matrix @ x <= upper_bound`. A
    `(column_idx, coefficient)` pair contributes
    `coefficient * x[column_idx]` to that row; omitted columns have a zero
    coefficient. The bounds supplied by the same call apply to that row.
    """

    def __init__(self) -> None:
        self._row_indices: list[int] = []
        self._column_indices: list[int] = []
        self._values: list[float] = []
        self._lower_bounds: list[float] = []
        self._upper_bounds: list[float] = []

    def add(
        self,
        coefficients: list[tuple[int, float]],
        *,
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        """
        Add one linear constraint as a new matrix row.

        Parameters
        ----------
        coefficients
            Pairs containing a variable's column index and its coefficient in
            this constraint. For example, `[(2, 1.0), (5, -1.0)]` represents
            `x[2] - x[5]`.
        lower_bound
            Minimum permitted value of the row's weighted sum.
        upper_bound
            Maximum permitted value of the row's weighted sum. Using the same
            value as `lower_bound` creates an equality constraint.

        """
        row_idx = len(self._lower_bounds)
        for column_idx, value in coefficients:
            self._row_indices.append(row_idx)
            self._column_indices.append(column_idx)
            self._values.append(value)
        self._lower_bounds.append(lower_bound)
        self._upper_bounds.append(upper_bound)

    def build(self, n_variables: int) -> LinearConstraint:
        matrix = coo_array(
            (
                np.asarray(self._values, dtype=np.float64),
                (
                    np.asarray(self._row_indices, dtype=np.int32),
                    np.asarray(self._column_indices, dtype=np.int32),
                ),
            ),
            shape=(len(self._lower_bounds), n_variables),
        ).tocsc()
        return LinearConstraint(
            matrix,
            np.asarray(self._lower_bounds, dtype=np.float64),
            np.asarray(self._upper_bounds, dtype=np.float64),
        )


def _checked_distance(
    *,
    true_text: str | None,
    pred_text: str | None,
    distance_fn: DistanceFnProtocol,
) -> float:
    cost = float(distance_fn(true_text=true_text, pred_text=pred_text))
    if not np.isfinite(cost) or cost < 0:
        msg = (
            "Text distance functions must return finite, non-negative costs. "
            f"Found {cost}."
        )
        raise ValueError(msg)
    return cost


def _build_node_text_costs(
    *,
    true_nodes: list[Node],
    pred_nodes: list[Node],
    distance_fn: DistanceFnProtocol,
) -> _NodeTextCosts:
    paired = np.empty((len(true_nodes), len(pred_nodes)), dtype=np.float64)
    for true_idx, true_node in enumerate(true_nodes):
        for pred_idx, pred_node in enumerate(pred_nodes):
            paired[true_idx, pred_idx] = _checked_distance(
                true_text=true_node.text,
                pred_text=pred_node.text,
                distance_fn=distance_fn,
            )

    unmatched_true = np.asarray(
        [
            _checked_distance(
                true_text=true_node.text,
                pred_text=None,
                distance_fn=distance_fn,
            )
            for true_node in true_nodes
        ],
        dtype=np.float64,
    )
    unmatched_pred = np.asarray(
        [
            _checked_distance(
                true_text=None,
                pred_text=pred_node.text,
                distance_fn=distance_fn,
            )
            for pred_node in pred_nodes
        ],
        dtype=np.float64,
    )
    return _NodeTextCosts(
        paired=paired,
        unmatched_true=unmatched_true,
        unmatched_pred=unmatched_pred,
    )


def _minimum_text_assignment(
    text_costs: _NodeTextCosts,
) -> tuple[IntArray, IntArray, float]:
    n_true, n_pred = text_costs.paired.shape
    size = n_true + n_pred
    all_costs = np.concatenate(
        (
            text_costs.paired.ravel(),
            text_costs.unmatched_true,
            text_costs.unmatched_pred,
        )
    )
    max_cost = float(np.max(all_costs, initial=0.0))
    forbidden_cost = (max_cost + 1.0) * (size + 1)
    cost_matrix = np.full((size, size), forbidden_cost, dtype=np.float64)

    cost_matrix[:n_true, :n_pred] = text_costs.paired
    for true_idx in range(n_true):
        cost_matrix[true_idx, n_pred + true_idx] = text_costs.unmatched_true[true_idx]
    for pred_idx in range(n_pred):
        cost_matrix[n_true + pred_idx, pred_idx] = text_costs.unmatched_pred[pred_idx]
    cost_matrix[n_true:, n_pred:] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    minimum_text_cost = float(cost_matrix[row_ind, col_ind].sum())
    return row_ind, col_ind, minimum_text_cost


def _node_matches_from_linear_assignment(
    *,
    true_nodes: list[Node],
    pred_nodes: list[Node],
    text_costs: _NodeTextCosts,
    row_ind: IntArray,
    col_ind: IntArray,
    true_diagram_option_idx: int,
) -> NodeMatches:
    n_true = len(true_nodes)
    n_pred = len(pred_nodes)
    node_matches = NodeMatches(
        matches=[],
        true_diagram_option_idx=true_diagram_option_idx,
    )

    for row_idx, column_idx in zip(row_ind, col_ind, strict=True):
        if row_idx < n_true and column_idx < n_pred:
            cost = text_costs.paired[row_idx, column_idx]
            node_matches.add(
                true_node=true_nodes[row_idx],
                pred_node=pred_nodes[column_idx],
                node_text_cost=cost,
            )
        elif row_idx < n_true and column_idx == n_pred + row_idx:
            cost = text_costs.unmatched_true[row_idx]
            node_matches.add(
                true_node=true_nodes[row_idx],
                pred_node=None,
                node_text_cost=cost,
            )
        elif row_idx == n_true + column_idx and column_idx < n_pred:
            cost = text_costs.unmatched_pred[column_idx]
            node_matches.add(
                true_node=None,
                pred_node=pred_nodes[column_idx],
                node_text_cost=cost,
            )

    return node_matches


def _edges_by_node_index(nodes: list[Node]) -> list[tuple[int, int]]:
    node_indices = {node.node_number: idx for idx, node in enumerate(nodes)}
    edges: list[tuple[int, int]] = []
    for source_idx, node in enumerate(nodes):
        if node.points_to is None:
            msg = "All nodes must have flow before extracting diagram edges."
            raise ValueError(msg)
        edges.extend(
            (source_idx, node_indices[target_number])
            for target_number in node.points_to
        )
    return edges


def _build_node_matching_constraints(
    *,
    layout: _MilpVariableLayout,
    text_costs: _NodeTextCosts,
    minimum_text_cost: float,
    true_edges_by_node_idx: list[tuple[int, int]],
    pred_edges_by_node_idx: list[tuple[int, int]],
) -> _LinearConstraintBuilder:
    constraints = _LinearConstraintBuilder()

    # Require each true node to match one predicted node or be marked unmatched.
    for true_idx in range(layout.n_true):
        constraints.add(
            [
                (layout.node_match_idx(true_idx, pred_idx), 1.0)
                for pred_idx in range(layout.n_pred)
            ]
            + [(layout.unmatched_true_node_idx(true_idx), 1.0)],
            lower_bound=1.0,
            upper_bound=1.0,
        )

    # Require each predicted node to match one true node or be marked unmatched.
    for pred_idx in range(layout.n_pred):
        constraints.add(
            [
                (layout.node_match_idx(true_idx, pred_idx), 1.0)
                for true_idx in range(layout.n_true)
            ]
            + [(layout.unmatched_pred_node_idx(pred_idx), 1.0)],
            lower_bound=1.0,
            upper_bound=1.0,
        )

    # Build (decision index, text cost) pairs for every node-pair and unmatched-node
    # option.
    text_coefficients = [
        (
            layout.node_match_idx(true_idx, pred_idx),
            text_costs.paired[true_idx, pred_idx],
        )
        for true_idx in range(layout.n_true)
        for pred_idx in range(layout.n_pred)
    ]
    text_coefficients.extend(
        (
            layout.unmatched_true_node_idx(true_idx),
            text_costs.unmatched_true[true_idx],
        )
        for true_idx in range(layout.n_true)
    )
    text_coefficients.extend(
        (
            layout.unmatched_pred_node_idx(pred_idx),
            text_costs.unmatched_pred[pred_idx],
        )
        for pred_idx in range(layout.n_pred)
    )
    # Require every MILP solution to retain the minimum node-text cost.
    constraints.add(
        [(idx, float(value)) for idx, value in text_coefficients],
        lower_bound=minimum_text_cost,
        upper_bound=minimum_text_cost,
    )

    for true_edge_idx, (
        true_source_node_idx,
        true_target_node_idx,
    ) in enumerate(true_edges_by_node_idx):
        for pred_edge_idx, (
            pred_source_node_idx,
            pred_target_node_idx,
        ) in enumerate(pred_edges_by_node_idx):
            edge_match_idx = layout.edge_match_idx(true_edge_idx, pred_edge_idx)
            source_node_match_idx = layout.node_match_idx(
                true_source_node_idx,
                pred_source_node_idx,
            )
            target_node_match_idx = layout.node_match_idx(
                true_target_node_idx,
                pred_target_node_idx,
            )

            # An edge match requires its source nodes to be matched.
            constraints.add(
                [(edge_match_idx, 1.0), (source_node_match_idx, -1.0)],
                lower_bound=-np.inf,
                upper_bound=0.0,
            )

            # An edge match requires its target nodes to be matched.
            constraints.add(
                [(edge_match_idx, 1.0), (target_node_match_idx, -1.0)],
                lower_bound=-np.inf,
                upper_bound=0.0,
            )

            # Matching both endpoint pairs requires the edges to be matched.
            constraints.add(
                [
                    (edge_match_idx, 1.0),
                    (source_node_match_idx, -1.0),
                    (target_node_match_idx, -1.0),
                ],
                lower_bound=-1.0,
                upper_bound=np.inf,
            )

    return constraints


def _solve_exact_node_matching_milp(
    *,
    objective: FloatArray,
    layout: _MilpVariableLayout,
    constraints: _LinearConstraintBuilder,
    stage: str,
    parent_img_code: str,
    true_diagram_option_idx: int,
) -> OptimizeResult:
    result = milp(
        c=objective,
        integrality=np.ones(layout.n_variables, dtype=np.uint8),
        bounds=Bounds(
            np.zeros(layout.n_variables, dtype=np.float64),
            np.ones(layout.n_variables, dtype=np.float64),
        ),
        constraints=constraints.build(layout.n_variables),
        options={"mip_rel_gap": 0.0},
    )
    if result.status != 0 or not result.success or result.x is None:
        msg = (
            f"Could not prove an optimal node matching during {stage}.\n"
            f"Diagram: {parent_img_code}\n"
            f"True option index: {true_diagram_option_idx}\n"
            f"Solver status: {result.status}\n"
            f"Solver message: {result.message}"
        )
        raise RuntimeError(msg)
    return result


def _label_cost_objective(
    *,
    true_nodes: list[Node],
    pred_nodes: list[Node],
    distance_fn: DistanceFnProtocol,
    layout: _MilpVariableLayout,
) -> FloatArray:
    objective = np.zeros(layout.n_variables, dtype=np.float64)
    cached_costs: dict[tuple[tuple[str, ...], tuple[str, ...]], float] = {}

    def label_cost(true_labels: list[str], pred_labels: list[str]) -> float:
        cache_key = (tuple(true_labels), tuple(pred_labels))
        if cache_key not in cached_costs:
            cached_costs[cache_key] = match_texts_lists(
                true_texts=true_labels,
                pred_texts=pred_labels,
                distance_fn=distance_fn,
            ).total_cost
        return cached_costs[cache_key]

    for true_idx, true_node in enumerate(true_nodes):
        if true_node.labels is None:
            msg = "True node labels cannot be None when optimising node labels."
            raise ValueError(msg)
        for pred_idx, pred_node in enumerate(pred_nodes):
            if pred_node.labels is None:
                msg = "Pred node labels cannot be None when optimising node labels."
                raise ValueError(msg)
            objective[layout.node_match_idx(true_idx, pred_idx)] = label_cost(
                true_node.labels,
                pred_node.labels,
            )
        objective[layout.unmatched_true_node_idx(true_idx)] = label_cost(
            true_node.labels,
            [],
        )

    for pred_idx, pred_node in enumerate(pred_nodes):
        if pred_node.labels is None:
            msg = "Pred node labels cannot be None when optimising node labels."
            raise ValueError(msg)
        objective[layout.unmatched_pred_node_idx(pred_idx)] = label_cost(
            [],
            pred_node.labels,
        )

    return objective


def _node_matches_from_milp_solution(
    *,
    solution: FloatArray,
    layout: _MilpVariableLayout,
    true_nodes: list[Node],
    pred_nodes: list[Node],
    text_costs: _NodeTextCosts,
    true_diagram_option_idx: int,
) -> NodeMatches:
    node_matches = NodeMatches(
        matches=[],
        true_diagram_option_idx=true_diagram_option_idx,
    )
    matched_pred_indices: set[int] = set()

    for true_idx, true_node in enumerate(true_nodes):
        matched_pred_idx = next(
            (
                pred_idx
                for pred_idx in range(layout.n_pred)
                if (
                    solution[layout.node_match_idx(true_idx, pred_idx)]
                    > _BINARY_SELECTED_THRESHOLD
                )
            ),
            None,
        )
        if matched_pred_idx is None:
            cost = text_costs.unmatched_true[true_idx]
            node_matches.add(
                true_node=true_node,
                pred_node=None,
                node_text_cost=cost,
            )
            continue

        matched_pred_indices.add(matched_pred_idx)
        cost = text_costs.paired[true_idx, matched_pred_idx]
        node_matches.add(
            true_node=true_node,
            pred_node=pred_nodes[matched_pred_idx],
            node_text_cost=cost,
        )

    for pred_idx, pred_node in enumerate(pred_nodes):
        if pred_idx in matched_pred_indices:
            continue
        cost = text_costs.unmatched_pred[pred_idx]
        node_matches.add(
            true_node=None,
            pred_node=pred_node,
            node_text_cost=cost,
        )

    return node_matches


def _optimise_text_optimal_node_matches(
    *,
    true_diagram: Diagram,
    pred_diagram: Diagram,
    distance_fn: DistanceFnProtocol,
    text_costs: _NodeTextCosts,
    minimum_text_cost: float,
    fallback_assignment: tuple[IntArray, IntArray],
    true_diagram_option_idx: int,
) -> NodeMatches:
    if true_diagram.nodes is None or pred_diagram.nodes is None:
        msg = "Both diagrams must have nodes before optimising node matches."
        raise ValueError(msg)

    true_nodes = true_diagram.nodes
    pred_nodes = pred_diagram.nodes
    true_edges_by_node_idx = (
        _edges_by_node_index(true_nodes) if pred_diagram.flow_done else []
    )
    pred_edges_by_node_idx = (
        _edges_by_node_index(pred_nodes) if pred_diagram.flow_done else []
    )
    layout = _MilpVariableLayout(
        n_true=len(true_nodes),
        n_pred=len(pred_nodes),
        n_true_edges=len(true_edges_by_node_idx),
        n_pred_edges=len(pred_edges_by_node_idx),
    )
    constraints = _build_node_matching_constraints(
        layout=layout,
        text_costs=text_costs,
        minimum_text_cost=minimum_text_cost,
        true_edges_by_node_idx=true_edges_by_node_idx,
        pred_edges_by_node_idx=pred_edges_by_node_idx,
    )

    final_result: OptimizeResult | None = None
    if layout.n_edge_match_slots > 0:
        flow_objective = np.zeros(layout.n_variables, dtype=np.float64)
        flow_objective[layout.edge_matches_start_idx :] = -1.0
        final_result = _solve_exact_node_matching_milp(
            objective=flow_objective,
            layout=layout,
            constraints=constraints,
            stage="flow optimisation",
            parent_img_code=pred_diagram.parent_img_code,
            true_diagram_option_idx=true_diagram_option_idx,
        )
        best_edge_matches = round(
            float(final_result.x[layout.edge_matches_start_idx :].sum())
        )
        # Require subsequent MILP solutions to retain the maximum edge-match count.
        constraints.add(
            [
                (edge_match_idx, 1.0)
                for edge_match_idx in range(
                    layout.edge_matches_start_idx,
                    layout.n_variables,
                )
            ],
            lower_bound=float(best_edge_matches),
            upper_bound=float(best_edge_matches),
        )

    if pred_diagram.labels_done:
        label_objective = _label_cost_objective(
            true_nodes=true_nodes,
            pred_nodes=pred_nodes,
            distance_fn=distance_fn,
            layout=layout,
        )
        final_result = _solve_exact_node_matching_milp(
            objective=label_objective,
            layout=layout,
            constraints=constraints,
            stage="label optimisation",
            parent_img_code=pred_diagram.parent_img_code,
            true_diagram_option_idx=true_diagram_option_idx,
        )

    if final_result is None:
        row_ind, col_ind = fallback_assignment
        return _node_matches_from_linear_assignment(
            true_nodes=true_nodes,
            pred_nodes=pred_nodes,
            text_costs=text_costs,
            row_ind=row_ind,
            col_ind=col_ind,
            true_diagram_option_idx=true_diagram_option_idx,
        )

    solution = np.asarray(final_result.x, dtype=np.float64)
    node_matches = _node_matches_from_milp_solution(
        solution=solution,
        layout=layout,
        true_nodes=true_nodes,
        pred_nodes=pred_nodes,
        text_costs=text_costs,
        true_diagram_option_idx=true_diagram_option_idx,
    )
    if not np.isclose(node_matches.total_node_text_cost, minimum_text_cost):
        msg = (
            "The node matching optimiser changed the minimum node-text cost.\n"
            f"Expected: {minimum_text_cost}\n"
            f"Found: {node_matches.total_node_text_cost}"
        )
        raise RuntimeError(msg)
    return node_matches


def match_node_labels(
    node_matches: NodeMatches,
    distance_fn: DistanceFnProtocol,
) -> NodeMatches:
    for match in node_matches.matches:
        true_labels = match.true_node.labels if match.true_node is not None else []
        pred_labels = match.pred_node.labels if match.pred_node is not None else []

        if true_labels is None or pred_labels is None:
            msg = "Node labels cannot be None when matching node labels."
            raise ValueError(msg)

        match.label_matches = match_texts_lists(
            true_texts=true_labels,
            pred_texts=pred_labels,
            distance_fn=distance_fn,
        )

    return node_matches


def match_additional_texts(
    true_diagram: Diagram,
    pred_diagram: Diagram,
    distance_fn: DistanceFnProtocol,
) -> TextListMatches:
    if true_diagram.additional_texts is None or pred_diagram.additional_texts is None:
        msg = "Additional texts cannot be None when matching additional texts."
        raise ValueError(msg)

    additional_text_matches = match_texts_lists(
        true_diagram.additional_texts,
        pred_diagram.additional_texts,
        distance_fn=distance_fn,
    )

    for text_match in additional_text_matches.matches:
        text_match.parent_img_code = true_diagram.parent_img_code
        text_match.true_option_idx = true_diagram.true_option_idx

    return additional_text_matches


def match_single_diagram(
    true_diagram_options: DiagramOptions,
    pred_diagram: Diagram,
    distance_fn: DistanceFnProtocol,
) -> DiagramMatch:
    node_matches = match_nodes(
        true_diagram_options=true_diagram_options,
        pred_diagram=pred_diagram,
        distance_fn=distance_fn,
    )

    # This true diagram is selected as the one with the best node matches out
    # of the options
    true_diagram = true_diagram_options.options[node_matches.true_diagram_option_idx]

    node_matches_with_labels = match_node_labels(
        node_matches=node_matches,
        distance_fn=distance_fn,
    )

    additional_text_matches = match_additional_texts(
        true_diagram=true_diagram,
        pred_diagram=pred_diagram,
        distance_fn=distance_fn,
    )

    return DiagramMatch(
        node_matches=node_matches_with_labels,
        additional_text_matches=additional_text_matches,
        pred_diagram=pred_diagram,
        true_diagram=true_diagram,
    )


def match_diagrams(
    true_diagrams_options_list: list[DiagramOptions],
    pred_diagrams: list[Diagram],
    distance_fn: DistanceFnProtocol,
) -> list[DiagramMatch]:
    diagram_pairs = zip(
        true_diagrams_options_list,
        pred_diagrams,
        strict=True,
    )

    return [
        match_single_diagram(
            true_diagram_options=true_diagram_options,
            pred_diagram=pred_diagram,
            distance_fn=distance_fn,
        )
        for true_diagram_options, pred_diagram in tqdm(
            diagram_pairs,
            total=len(pred_diagrams),
            desc="Matching diagrams",
            unit="diagram",
        )
    ]
