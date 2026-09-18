# Few-shot examples

Few-shot examples show a model how you want images classified or flowcharts
parsed. Each example pairs an image with the expected JSON answer.

## Classification examples

Suppose we are doing flowchart classification with one of our [default
classification helpers](setup-default-funcs.md). We can use
[`VisionFewShotExample`](../reference/data-types.md#flowde.utils.VisionFewShotExample)
to supply a flowchart image, `data/examples/flowchart.png`, and its expected
answer, saved in `data/examples/flowchart.json`, as a worked example to help
improve classification performance:

```python
from pathlib import Path

from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn
from flowde.utils import VisionFewShotExample

example = VisionFewShotExample(
    img_path=Path("data/examples/flowchart.png"),
    expected_output_path=Path("data/examples/flowchart.json"),
)

classify_fn = make_openai_classify_fn(
    input_text="Return 1 if the image is a flowchart, otherwise return 0.",
    model="gpt-5.6-luna",
    effort="medium",
    few_shot_examples=[example],
)
```

Where `data/examples/flowchart.json` contains:

```json
{ "label": 1 }
```

You can include several examples in the `few_shot_examples` list. Flowde
includes every supplied example in each request, in list order. Omitting
`few_shot_examples` sends no worked examples.

## Parsing examples

Suppose we are parsing node text with one of our [default parsing
helpers](setup-default-funcs.md). We can use
[`VisionFewShotExample`](../reference/data-types.md#flowde.utils.VisionFewShotExample)
to supply a flowchart image, `data/examples/flowchart.png`, and its expected
answer, saved in `data/examples/node_text/flowchart.json`, as a worked example
to help improve parsing performance:

```python
from pathlib import Path

from flowde.parsing_fns.openai_parse import make_openai_parse_fn
from flowde.utils import VisionFewShotExample

example = VisionFewShotExample(
    img_path=Path("data/examples/flowchart.png"),
    expected_output_path=Path("data/examples/node_text/flowchart.json"),
)

parse_fn = make_openai_parse_fn(
    input_text="Parse the text of every node. Use node numbers starting at 1.",
    model="gpt-5.6-luna",
    effort="medium",
    parts_to_parse={"node_text"},
    few_shot_examples=[example],
)
```

In this example, `data/examples/node_text/flowchart.json` contains:

```json
{
  "nodes": [
    { "node_number": 1, "text": "Screened (n = 120)" },
    { "node_number": 2, "text": "Included (n = 100)" }
  ]
}
```

The example answer contains node numbers and text because
`parts_to_parse={"node_text"}` requests only node text. For other parsing
tasks, the example answer must match the selected `parts_to_parse` or custom
`result_structure`.

### Examples with previously parsed parts

When [parsing parts separately](parsing.md#parse-parts-separately), you can use
the `partial_flowchart` attribute of
[`VisionFewShotExample`](../reference/data-types.md#flowde.utils.VisionFewShotExample)
to supply previously parsed parts for the example image and show the model
how to parse new parts using previously parsed parts in a worked example:

```python
from pathlib import Path

from flowde.parsing_fns.openai_parse import make_openai_parse_fn
from flowde.parsing_fns.parsing_types import build_a_partial_flowchart
from flowde.utils import VisionFewShotExample

example = VisionFewShotExample(
    img_path=Path("data/examples/flowchart.png"),
    expected_output_path=Path("data/examples/labels/flowchart.json"),
    partial_flowchart=build_a_partial_flowchart(
        nodes_path=Path("data/examples/node_text/flowchart.json")
    ),
)

parse_fn = make_openai_parse_fn(
    input_text="Extract labels for the supplied nodes. Keep their node numbers.",
    model="gpt-5.6-luna",
    effort="medium",
    parts_to_parse={"labels"},
    few_shot_examples=[example],
)
```

In this example, `data/examples/labels/flowchart.json` contains:

```json
{
  "nodes": [
    { "node_number": 1, "labels": ["Screening"] },
    { "node_number": 2, "labels": ["Inclusion"] }
  ]
}
```

The `partial_flowchart`, `data/examples/node_text/flowchart.json`, contains:

```json
{
  "nodes": [
    { "node_number": 1, "text": "Screened (n = 120)" },
    { "node_number": 2, "text": "Included (n = 100)" }
  ]
}
```
