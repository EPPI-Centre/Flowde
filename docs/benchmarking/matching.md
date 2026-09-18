# How parsing matches diagrams

Before calculating parsing scores, Flowde must identify which predicted node
represents which ground-truth node. The two parsers may have numbered the same
visual boxes differently.

A **Node Match** pairs a predicted node with the ground-truth node it represents.
An extra predicted node or a missing ground-truth node can be recorded as
unmatched. A **Ground-Truth Option** is one complete accepted interpretation of
the diagram.

## Node numbers identify nodes within one diagram

Suppose the ground truth contains:

```text
Node 1: Screened → Node 2: Included
```

The prediction instead contains:

```text
Node 2: Screened → Node 1: Included
```

The text and direction are correct. Flowde can match predicted node `2` to true
node `1`, and predicted node `1` to true node `2`. The predicted connection
then becomes the correct ground-truth edge `1 → 2`.

Node numbers are therefore identifiers, not evidence for deciding which nodes
represent the same content.

## Find matches within an option

For each Ground-Truth Option, Flowde selects Node Matches in this order:

1. Minimise the total node-text cost, including unmatched nodes.
2. Among matches tied on node-text cost, maximise flow similarity if flow was
   predicted.
3. Among remaining ties, minimise node-label cost if labels were predicted.

The selected text distance determines the first cost. For example, raw
Levenshtein distance counts character edits. An unmatched node's text is
compared with an absent string using the same distance function.

Later criteria only resolve ties. A better flow match does not outweigh a
worse node-text cost. Additional text does not affect Node Matches because
additional text is not attached to individual nodes.

### Example: repeated node text

Suppose two ground-truth nodes both say `Lost to follow-up`, one on each study
arm. The prediction also contains two nodes with that text. Several assignments
can have the same text cost.

If flow was predicted, the connections can identify which predicted node
belongs to which arm. If the connections still leave a tie and labels were
predicted, labels provide the next comparison. A missing component is skipped.

The matcher uses an optimisation solver to select globally optimal matches at
each applicable stage. It raises an error if it cannot prove the required
optimal result; it does not silently return an unproven approximation.

## Choose a Ground-Truth Option

After evaluating the alternatives, Flowde selects an option using the same
ordered criteria:

1. lowest node-text cost;
2. highest flow similarity, if available;
3. lowest node-label cost, if available;
4. lowest additional-text cost, if available.

If every available score ties, the first option in the ground-truth `options`
list is selected. This final tie rule selects the option; it does not make node
number equality a matching criterion.

For example, one accepted interpretation might duplicate a shared box to keep
two branches separate, while another may represent the visual box once. The
prediction is evaluated against the available options using the rules above.
The selected option is then used consistently for node, label, flow and
additional-text results.

## Understand what changes the result

Changing the distance function can change the Node Matches and selected
Ground-Truth Option. Adding predicted flow or labels can also resolve ties
that a node-text-only prediction could not resolve. These are differences in
the available evidence, rather than separate matching policies for each
scoring method.

Use
[`benchmark.node_matches()`](../reference/benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_matches)
to inspect the selected option index and each
Node Match. For a complete prediction,
[`benchmark.diagram_matches()`](../reference/benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.diagram_matches)
also
provides the selected ground-truth diagram and additional-text matches.

See the [worked parsing benchmark](parsing.md) for concrete text and edge scores.
