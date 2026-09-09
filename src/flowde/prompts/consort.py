consort_nodes_prompt = """
I am sending you an image of a participant flow diagram.

Your task is to parse only the node text and node numbers.

# Some background.

We have built a tool to parse data from participant flow diagrams and consort images.
To do this we have developed a JSON format for representing the image data in a
structured, non-visual way. The goal is convert all the text and visual information the
author has intended to portray into the JSON.

Our parsed format is represented by this pydantic model:

```python
from pydantic import BaseModel


class Node(BaseModel):
    node_number: int
    text: str
    labels: list[str]
    points_to: list[int]


class Flowchart(BaseModel):
    nodes: list[Node]
    additional_texts: list[str]
```

Each node represents one box or item in the flowchart.

The fields have the following meanings:

- `node_number`: a unique id given to each node in the flowchart;
- `text`: the text inside the node;
- `labels`: text labels that apply to the node;
- `points_to`: the node numbers that follow this node in the flow;
- `additional_texts`: text that is not assigned to a specific node.

You can use the few shot examples provided to get an idea for what sort of things
count as which field. But in general nodes aim to capture main flow items of the
flowchart. Labels are text that is not a main flow item, but text that applies
to a subset of nodes. Labels often add important context to nodes, meaning the nodes
themselves often lose meaning without them. E.g. we could have a bunch of nodes stating
numbers which are meaningless, until we see the label "6 Month Follow-Up" indicating
that the numbers indicate the number of patients attending the 6 month follow up. A
single Label often applies to multiple nodes, but does not apply to all nodes. When a
label applies to multiple nodes, we include the label text in the `labels` field of each
of those nodes. The `additional_texts` field is used to capture any text that is neither
a node or a label and includes things like captions, footnotes, and other text that is
not directly attached to a specific node or applies equally to all nodes. The
`points_to` captures the flow of the diagram and is pretty self explanatory.

## Example 1: Basic nodes and labels

```text
                    ┌──────────────────────────────┐
                    │ Assessed for eligibility     │
                    │ (n = 120)                    │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    │         ┌────────────┐       │
                    │         | Allocation |       │
                    │         └────────────┘       │
                    ▼                              ▼
        ┌──────────────────────────┐   ┌──────────────────────────┐
        │ Allocated to intervention│   │ Allocated to control     │
        │ (n = 60)                 │   │ (n = 60)                 │
        └────────────┬─────────────┘   └────────────┬─────────────┘
                     │        ┌────────────┐        │
                     │        | Follow-Up  |        │
                     │        └────────────┘        │
                     ▼                              ▼
        ┌──────────────────────────┐   ┌──────────────────────────┐
        │ Completed follow-up      │   │ Completed follow-up      │
        │ (n = 55)                 │   │ (n = 57)                 │
        └──────────────────────────┘   └──────────────────────────┘

Figure 1. Example flowchart for demonstrating parsing.
```

Is ideally parsed as:

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "Assessed for eligibility\n(n = 120)",
      "labels": [],
      "points_to": [2, 3]
    },
    {
      "node_number": 2,
      "text": "Allocated to intervention\n(n = 60)",
      "labels": ["Allocation"],
      "points_to": [4]
    },
    {
      "node_number": 3,
      "text": "Allocated to control\n(n = 60)",
      "labels": ["Allocation"],
      "points_to": [5]
    },
    {
      "node_number": 4,
      "text": "Completed follow-up\n(n = 55)",
      "labels": ["Follow-up"],
      "points_to": []
    },
    {
      "node_number": 5,
      "text": "Completed follow-up\n(n = 57)",
      "labels": ["Follow-up"],
      "points_to": []
    }
  ],
  "additional_texts": ["Figure 1. Example flowchart for demonstrating parsing."]
}
```

Notice how the "Allocation" and "Follow-Up" are not parsed as nodes, but instead are
parsed as labels because they are not connected to the flow, are not intended to be
and apply to multiple nodes on their respective rows.

## Example 2: Keeping branches separate

The goal of parsing is to interpret the visual content of an image and convert
it into a standardised JSON format. This means we may not always want to parse
the image exactly as it appears.

Commonly, a flowchart will have two branches that follow the same step,
see `C` in the example below.

```text
        ┌───┐                 ┌───┐
        │ A │                 │ B │
        └─┬─┘                 └─┬─┘
          │                     │
          └──────► ┌───┐ ◄──────┘
                   │ C │
          ┌─────── └───┘ ───────┐
          │                     │
          ▼                     ▼
        ┌───┐                 ┌───┐
        │ D │                 │ E │
        └───┘                 └───┘
```

The author may use a single node to describe the step, but
the branches are still very much consdiered separate. If we parsed this exactly
as it appears in our JSON format, we would lose track of which branch is which:

```text
A → C → [D, E]
B → C → [D, E]
```
To combat this, if a single node is used to describe a step that applies
separately to each branch, we duplicate the node to keep the branches separate.
Now we are able to capture the underlying meaning of the image in our JSON
format:

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "A",
      "labels": [],
      "points_to": [3]
    },
    {
      "node_number": 2,
      "text": "B",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 3,
      "text": "C",
      "labels": [],
      "points_to": [5]
    },
    {
      "node_number": 4,
      "text": "C",
      "labels": [],
      "points_to": [6]
    },
    {
      "node_number": 5,
      "text": "D",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 6,
      "text": "E",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```

This preserves the two intended branches as:

```text
A → C → D
B → C → E
```

### Example 3: Advanced labels

Complicated nodes and labels may require additional processing to convert the
visual content to JSON. Remember, all we care is about preserving the meaning of
the diagram and we are willing to bend ANY rules I may give you to do that.

Take this image:

```text
                         ┌──────────────────────────┐
                         │        Randomised        │
                         └────────────┬─────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
                 │              ┌────────────┐             │
                 │              │ Follow-Up  │             │
                 │              └────────────┘             │
                 ▼                                         ▼
                 ┌────────────┐                 ┌────────────┐
                 │            │                 │            │
       26 weeks  │     42     │                 │     40     │
                 │            │                 │            │
                 │     35     │                 │     33     │  52 weeks
                 │            │                 │            │
                 └────────────┘                 └────────────┘
```

If we parse this exactly as it appears, for the bottom left node, we get
something like this:

```json
...
    {
      "node_number": 2,
      "text": "42\n35",
      "labels": [
        "Follow-Up",
        "26 weeks",
        "52 weeks"
      ],
      "points_to": []
    },
...
```

By parsing `26 weeks` and `52 weeks` as separate labels, each labels appears to
apply to the entire node. This is incorrect. In reality, `26 weeks` applies to
`42` and `52 weeks` applies to `35`.

But remember, our goal is to correctly represent the information in the image,
not to parse the image exactly as it appears.

For this scenario, where labels appear to apply to separate parts of the node,
we have a couple of options:


#### Option 1: Join labels

We can join labels that apply to separate parts of a node, into a single label,
effectively capturing that each part of the label applies to a different part of
the node.

```json
...
    {
      "node_number": 2,
      "text": "42\n35",
      "labels": [
        "Follow-Up",
        "26 weeks\n52 weeks",
      ],
      "points_to": []
    },
...
```

Now `26 weeks\n52 weeks` correctly applies to `42\n35`

#### Option 2: Split nodes

In cases such as this one, the bottom left node would probably be more
accurately represented by 2 separate nodes. We can split the nodes like these to
more accurately capture the information in the image.

The the bottom left node would become:

```json
...
    {
      "node_number": 2,
      "text": "42",
      "labels": [
        "Follow-Up",
        "26 weeks"
      ],
      "points_to": [4]
    },
    ...
    {
      "node_number": 4,
      "text": "35",
      "labels": [
        "Follow-Up",
        "52 weeks"
      ],
      "points_to": []
    },
...
```

Now we correctly capture that 42 patients attended the 26 weeks follow-up and 35
patients attended the 52 weeks follow-up.

# Your Task: Parse the nodes

Your task is to parse only the node text and node numbers.
We will build up the other fields in later steps.

Do not parse:
- labels;
- arrows;
- connections;
- section headings;
- row labels;
- captions;
- footnotes;
- additional notes;
- legends or keys;
- any other non-node text.

Return only the parsed nodes requested by the response schema.

# Core rule: what counts as a node?

A node is a flowchart item that the diagram author intended as part of the main
study flow, participant flow, or process flow or just general flow.

Ask yourself:

"Is this text a connected item in the main flowchart, or one of the things that
participants, users, groups, schools, clusters, arms, or study units pass
through, are assigned to, receive, complete, fail to complete, are followed up
at, are excluded from, are lost from, are analysed in, or otherwise move
through?"

If yes, parse it as a node.

If no, do not parse it as a node.

A node does not have to be a participant event. If a connected diagram item
contains text, numbers, or symbols that preserve the meaning of a specific
branch, source, transition, comparison, or outcome row, parse it as a node.

If you think the author intended the text to be a node, parse it as a node.

Some example of what a node can be:
- a participant-flow item, where participants/users are assessed, recruited,
  randomised, allocated, followed up, excluded, lost, analysed, or otherwise move
  through the study;
- a group-flow item, where schools, clusters, practices, sites, arms, users, or
  other study units are randomised, allocated, followed up, assessed, or
  analysed;
- a study-process item that is drawn as part of the main flowchart sequence,
  even if no participants/users are inside that step yet;
- a branch-entry item that defines what a specific branch receives, represents,
  or is allocated to;
- a shared connected item that multiple branches pass through;
- a short line-node or branch-decision item written directly on a flow line when
  it marks what happens to that specific branch;
- a connected source, setting, recruitment-channel, site, clinic, centre, school,
  platform, or session item that forms the origin of a source stream;
- a connected quantitative or comparison annotation that belongs to a specific
  branch, transition, comparison, or outcome row.

Do not reject a boxed item merely because it has no participant count, because
it looks like an arm label, or because it occurs before recruitment, enrolment,
allocation, or randomisation. If it is drawn as a connected step in the main
flowchart sequence, you can parse it as a node, if you belive it is one.

Do not reject a connected item or split-point merely because it contains only a
single letter, symbol, or short abbreviation. If it represents a
genuine step or split on the flow path, parse it as a node.

Nodes are usually easy to recognise visually. They are usually the boxes,
circles, or standalone flow items that arrows point into or out of as part of
the main flow. Some nodes may be unboxed text placed directly on a flow line.

Sometimes a visual node may need to become more than one parsed node. This is
not because the diagram contains multiple boxes, but because the single visual
box contains separate information for separate participant branches, or because
multiple branches pass through the same shared flowchart item. In those cases,
preserve the branch-specific flow rather than blindly transcribing the whole
visual box as one node.

Strong evidence that text is a node:
- it is inside a box, circle, or other flowchart shape that belongs to a specific
  branch or connected main-flow sequence;
- arrows point to it or away from it as part of the participant flow, group flow,
  or main process flow;
- it is written directly on a flow line and marks a branch decision, branch
  state, assignment, response, or flow outcome;
- it is a CONNECTED flowchart item describing participants/users/groups being assessed,
  recruited, invited, randomised, allocated, followed up, excluded, lost, analysed,
  included, assigned, contacted, treated, or measured;
- it is a CONNECTED item describing an intervention, treatment, support condition,
  exposure, message, assessment, follow-up, or other study-process step connected to the
  main flow;
- it gives a participant/user/group count for a SPECIFIC flow event in a SPECIFIC branch
  or connected main-flow step;
- it is a CONNECTED item defining what a branch receives, represents, or does after
  randomisation or allocation;
- it is a CONNECTed item that preserves the meaning of a specific source, branch,
  transition, comparison, or outcome row in the connected diagram.

Being inside a box is not enough by itself. Some boxes are labels, headings,
stage markers, legends, or visual organisers. Only parse boxed text as a node if
the box represents a connected item in the main flow.

Typical nodes include connected flowchart items that represent:
- eligibility assessment;
- recruitment, enrolment, invitation, or screening;
- randomisation or allocation;
- branch decisions, responses, assignments, or branch states;
- intervention, exposure, treatment, support, or study-process steps;
- follow-up, assessment, contact, measurement, or outcome collection;
- exclusions, withdrawals, losses, opt-outs, failures, or non-response;
- analysis or inclusion in analysis;
- source streams or recruitment settings;
- connected quantitative comparisons or summaries.

## What is not a node?

Do not parse text as a node just because it is near the flowchart.

Do not parse text as a node just because it is inside a box or shape.

Do not parse text as a node if it is exclusively a label, heading, stage marker, row
label, column label, timepoint label, branch label, caption, footnote, legend,
or explanatory note with no clear belonging to a specific branch and not connected
by arrows or lines to the main flow.

If the node is NOT CONNECTED and the contents is intended to be applied to multiple
nodes, act accross multiple nodes, or sum accross multiple nodes, then it is a label,
not a node.

When the following are not in the main flow and not connected by arrows or lines,
they are usually not nodes:
- section headings;
- group headings;
- phase labels;
- side labels;
- stage label
- row labels;
- column labels;
- timepoint labels;
- captions;
- legends or keys;
- labels that sit outside the flow and apply to nearby boxes.

However, do not exclude text merely because it has a label-like wording. A phrase
that names an arm, treatment, intervention, follow-up point, assessment point, or
study phase should still be parsed as a node if it is drawn as a connected item
in the main flow.

Similarly, BEWARE THE KEYWORD TRAP: Do not be tricked by words like "assessed",
"analysed","allocated", or the presence of a participant count (e.g., "n=50").
Keywords and numbers DO NOT automatically make text a node. If text containing these
words acts as a marginal summary of other nodes, a floating note, or stage label without
being structurally connected to the flow path, it is NOT a node.

The key distinction is:

- If the text is a connected item that the flow passes through, parse it as a
  node.
- If the text is unconnected and merely labels, groups, or explains other connected
  items, do not parse it as a node. It is probably a label.
- If the text is unconnected and has no clear relation to the surrounding nodes,
  and does not appear in anyway related to the surrounding flow, even if it contains
  numbers, do not parse it as as node. It is probably an additional text.

Floating boxes or text blocks placed in the margins that provide detailed
breakdowns, legends, or expanded explanations but sit entirely outside the main
flow sequence without arrows physically pointing into or out of them, must NOT
be parsed as nodes. They are explanatory notes, even if they contain numbers,
counts, or look exactly like node boxes.

## Important judgement rule for deciding nodes.

I don't want to give you too many hard rules for what is a node and what isn't
because there are many edge cases and exceptions. Instead, you will need to use your
judgement decide whether something is a node or not, so feel free to break the rules
when the visual intent of the diagram clearly indicates that something is a node or
is not a node.

Again, if the text is simply labelling, categorising, or adding additional info to
something, that's label-like. If the text is something that happens or has happened
at a specific point in the flow, that's node-like. Use your judgement to decide which is
which.

## Empty boxes and empty shapes

Do not parse empty boxes, empty shapes, blank placeholders, or boxes with no
visible text as nodes.

A node must contain visible text, a visible number, or another visible mark that
can be transcribed as node text, even if it's only a single character.
If a box is connected to the flow but contains no transcribable text,
do not include it in the node output.

Do not invent placeholder text for an empty box.

## Source boxes and upstream category boxes

Some diagrams begin with several connected boxes that describe where
participants, users, groups, or study units came from.

Parse these as nodes when they are part of the connected flow and lead directly
to later flowchart items.

This includes source, setting, recruitment-channel, clinic, school, centre,
site, platform, or session boxes when:
- they are drawn inside the flowchart area;
- they are connected by lines or arrows to later nodes;
- later boxes depend on them as separate sources or categories;
- excluding them would remove the origin of a branch or source stream.

Do not exclude a connected source box merely because its wording looks like a
heading or category label.

Do not parse a source, setting, or category label as a node when it only sits
outside the flow, labels a column or section, or is not connected to later
flowchart items.

## Connected quantitative and comparison annotations

Some diagrams include connected quantitative annotations that are not participant
flow events themselves, but are still part of the connected diagram structure.

Parse these as nodes when they are visually connected to a specific branch,
comparison, outcome row, transition, or pair of nodes, and excluding them would
lose information about that specific part of the flowchart.

This includes connected boxes or line-adjacent text containing:
- statistical comparison values;
- p-values;
- percentages;
- totals or counts;
- quantitative summaries;
- comparison annotations between two branches or outcomes.

Parse a quantitative annotation as a node when:
- it is connected by a line or arrow to one or more specific nodes;
- it belongs to a specific branch, outcome row, comparison, or transition;
- it is not merely part of a legend, caption, footnote, axis, or general note;

Do not parse quantitative text as a node when it is only part of a caption,
legend, footnote, axis label, table outside the flowchart, or explanatory note
that is not connected to a specific flowchart item.

## Line nodes and branch-decision text

Some real nodes are not inside boxes. They may be written directly on a line, on
a split point, beside a line segment, or between two connected boxes.

Parse this kind of text as a node when it marks a specific branch of the flow.

This includes text that identifies:
- a response or decision branch;
- an assignment branch;
- an arm or selection branch;
- a branch state or outcome;
- a reason or condition for moving down one branch rather than another;
- a short flow event that happens between two boxed nodes.

Line text should be parsed as a node when:
- it lies directly on the flow path;
- it separates one branch from another;
- following the line through the diagram requires passing through that text;
- removing the text would lose information about what the branch means;
- the text belongs to one specific branch rather than merely labelling the whole
  diagram area.

When a line-node includes multiple nearby text fragments on the same visual line
or path, include all fragments that belong to that same line-node. Especially if the
text fragments are separated by the arrow or line itself, they likely belong together
as one line-node. If the text fragment on the right clearly does not makes sense to
continue the text on the left, then of course parse them separately.
Use your judgement.

Do not parse line text as a node when it is only a decorative line label, a row
label, a timepoint applying to several nodes, or a general
annotation that does not mark a specific branch of the flow.

If line text is a long text of explanatory text that would make more sense as a label
to another node or nodes, then it should be parsed as a label. If it is an explanatory
label that applies equally to the entire diagram, then it should be an additional text.

## Branch-entry boxes and arm-defining boxes

A box at the start of a branch should usually be parsed as a node when it is
connected to the randomisation, allocation, or branching point and defines what
that branch is, receives, or does.

This is true even if the text looks like an arm label or condition label.

Parse branch-entry boxes as nodes when they:
- are directly connected to the branch flow by arrows or lines;
- define the allocated group, intervention, support condition, treatment,
  exposure, comparison, or process for that branch;
- are followed by branch-specific outcomes, follow-up, assessment, or analysis
  boxes.
- list the number of patients, users, groups, or study units, general items
  entering that specific branch.

The descriptions above are loose, there are many more edge cases. Try to keep
node like boxes as nodes. This is just a case you sometimes falsely exclude
when you are trying to be too strict about what counts as a node.

## In-node notes and footnotes

If a note, footnote, asterisk explanation, or parenthetical explanation is
physically inside a node box, include it as part of the text in that node.

Do not include notes or footnotes that are outside the node box unless they are
clearly part of that specific node's text.

In general, if text inside a shape is being counted as a node, then all the text within
the shape belongs to the same node. Use your judgement to exclude for cases where this
is clearly not intended, such as multiple nodes inside a shape that are obviously
there separate nodes.

## Keep relevant unboxed text together.

When parsing nodes, particularly when the node text does not exist within a shape,
try to to keep together text that logically belongs together as part of the same node,
even if there are gaps between the text fragments or new lines.

### Example X:

```text

      8 chairs
      bought
        │
        │──────────► 5     require
        │                  varnish
        ▼
    3    do not
         require
         varnish
```

Here clearly, even though there is a big gap between 5 and "require varnish", and
multiple new lines in the text, they are obviously intended to belong together as one
node.

We would parse this like so:

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "8 chairs\nbought",
      "labels": [],
      "points_to": [2, 3]
      },
    {
      "node_number": 2,
      "text": "5 require\nvarnish",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 3,
      "text": "3 do not\nrequire\nvarnish",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}

# Splitting or duplicating nodes to preserve branch information

The goal is not just to transcribe the visible text. The goal is to preserve the
flow in a structured format so that each branch can later be parsed correctly.

As a human viewer, it may be obvious which branch passes through which part of
the diagram. However, if a shared visual item is parsed only once, or if a
combined visual node is parsed as one text block, the branch-specific flow may be
lost or become ambiguous in the parsed JSON, as we say in example 2 in the
background section.

When this happens, we duplicate the node so that the separate branch gets its own
version of the node and branch information remains separate.

Use this principle:

If parsing a visual item only once would make it unclear which branch continues
where, then split or duplicate the item so that each branch retains its own node
sequence.

If two or more branches pass through the same shared connected item and then continue as
separate branches, duplicate the shared item once for each branch.

This can be necessary even when the shared item is visually drawn only once and even
when the diagram does not draw separate connector lines through the
middle of the text.

The exact split should follow the visual meaning of the diagram. The important
thing is that the parsed node text should not collapse separate branch
information into one ambiguous text block, and should not collapse separate
branch paths into a single path.

Split or duplicate a converged node when:
- multiple branches enter, pass through, or leave the same visual node;
- parsing the whole node as one text block would make it unclear which
  information belongs to which branch;
- parsing the node only once would make it unclear how each separate branch
  continues through the flow.

This can be necessary even if the flowchart does not visibly split again after
the converged node. The reason for splitting is not only whether the visual flow
diverges afterwards. The reason is whether the parsed JSON would preserve or
lose the separate branch information.

Do not split or duplicate a node when:
- it explicitly represents a mathematical sum or total of the incoming branches;
- it is a terminal node at the very end of the flow that brings branches back
  together into a single unified count;
- it truly represents one shared combined group;
- all incoming branches have genuinely merged into a single participant, user,
  group, study-unit flow or any other flow that has genuinely merged into one combined
  flow;
- splitting would invent a separation that is not visually or logically present
  in the diagram.
- it is the active divergence or splitting point itself (e.g., "Randomized",
 "Allocated"). Parse the divergence node only ONCE, and let its `points_to`
 list capture the multiple branches. Only duplicate shared items that occur
 *after* the branches have already separated.

Only split when the image makes the branch-specific meaning clear. Do not guess
a split if the diagram does not provide enough visual evidence.

## Arrows to show where a node belongs

Some diagrams use arrows or connector lines to point to all locations where a shared
node belongs in multiple branch paths, rather than the usual use of arrows: showing a
simple one-way sequence. Authors will do this to save space, but clearly the intention
is that the event described in the shared node happens in all the branches that the
arrows point to at the location where the arrow is pointing to in the flow.

In such a case, the node should be duplicated for each branch path that the arrows
point to.

## Shared connected nodes versus shared labels

Some diagrams contain wide boxes, bars, shaded boxes, rounded rectangles, ovals,
braces, brackets, or central items that are shared by multiple branches.

These can be either shared connected nodes or shared labels.

CRITICAL DISTINCTION FOR SHARED ITEMS:
- If an item has multiple arrows going into it and out of it, applying to all
  branches and describes something that's happening to the items in the flowchart
  we count that as a shared connected node that requires splitting
  and duplicating for each branch.
- If it appears on one branch but clearly applies to multiple branches as labelling the
  next set of nodes, it must be classified as a label, not a node. Do not parse it.
- Beware of grouping symbols: If an item sits between or to the side of branches
  and uses large grouping symbols (like curly braces '{', brackets, or simple structural
  lines) to group a set of nodes together, does not follow the flow of the diagram,
  applies a shared label explaining those nodes and is not a sequential step in the flow,
  It is a shared label. Do not parse it as a node.

Parse a shared item as a node only when it lies on the actual flow path.

A shared item lies on the actual flow path when:
- arrows or lines enter it and leave it as part of the main flow; or
- the branch visually passes through the item between earlier and later nodes.

If multiple branches pass through the same shared connected node and those
branches are represented as separate branch paths in the parsed flow, duplicate
the shared node once for each branch path. Do this even if the node text is
identical for each branch and even if the visual diagram draws the shared node
only once.

I will give you some examples:

## Example A: Shared connected node

```text


                 ┌────────────┐         ┌────────────┐
                 │  100 dogs  │         │  100 cats  │
                 │            │         │            │
                 └────────────┘         └────────────┘
                       │                      │
                       │                      │
                       ▼                      ▼
                 ┌──────────────────────────────────┐
                 │         Received Cat Food        │
                 └──────────────────────────────────┘
                       │                      │
                       │                      │
                       ▼                      ▼
                 ┌────────────┐         ┌────────────┐
                 │   40 had   │         │   80 had   │
                 │   seconds  │         │   seconds  │
                 └────────────┘         └────────────┘

```

Clearly here, the "Received Cat Food" node is a shared connected node that lies on the
flow path of both branches. Therefore, we would parse it as a node and duplicate it
for each branch.

## Example B: Shared label

```text

 ┌────────────┐         ┌────────────┐         ┌────────────┐         ┌────────────┐
 │  102 took  │         │  100 took  │         │  100 took  │         │  100 took  │
 │  pill A    │         │  pill B    │         │  pill C    │         │  pill D    │
 └────────────┘         └────────────┘         └────────────┘         └────────────┘
       │                      │                      │                      │
    2 years                   │                      │                      │
       │                      │                      │                      │
       ▼                      ▼                      ▼                      ▼
 ┌────────────┐         ┌────────────┐         ┌────────────┐         ┌────────────┐
 │ 60 respond │         │ 80 respond │         │ 40 respond │         │ 70 respond │
 │ 1 died     │         │ 2 died     │         │ 5 died     │         │ 3 died     │
 └────────────┘         └────────────┘         └────────────┘         └────────────┘
```

Here the author has clearly used the "2 years" label to apply to all four branches,
but is saving time by writing it once and using arrows to point to all four branches.
This is shared label, and will apply to each of the bottom 4 nodes. We won't
parse it as a node.

e.g.
```json

...
    {
      "node_number": 5,
      "text": "60 respond\n1 died",
      "labels": ["2 years"],
      "points_to": []
    },
    {
      "node_number": 6,
      "text": "80 respond\n2 died",
      "labels": ["2 years"],
      "points_to": []
    },
...
```
## Example C: Shared label 2

```text

 ┌────────────┐         ┌────────────┐         ┌────────────┐
 │  102 took  │         │  100 took  │         │  100 took  │
 │  pill A    │         │  pill B    │         │  pill C    │
 └────────────┘         └────────────┘         └────────────┘
       │                      │                      │
       │                      │                      │
       │                 ┌──────────┐                │
       │                 │ Analysis │                │
       │                 └──────────┘                │
       │                      │                      │
       │                      │                      │
       ▼                      ▼                      ▼
 ┌────────────┐         ┌────────────┐         ┌────────────┐
 │     80     │         │     81     │         │     77     │
 │            │         │            │         │            │
 └────────────┘         └────────────┘         └────────────┘
```

Again, clearly "Analysis" is a shared label that applies to all three branches, despite
the fact it only exists on one.

e.g.
```json
...
    {
      "node_number": 4,
      "text": "80",
      "labels": ["Analysis"],
      "points_to": []
    },
    {
      "node_number": 5,
      "text": "81",
      "labels": ["Analysis"],
      "points_to": []
    },
...
```

## Example D: A strange case

Remember, the thing we care about is preserving the meaning of the diagram, not parsing
it exactly as it appears. If the diagram is very strange, we may need to break the rules
to preserve the meaning. There will be all sorts of edge cases we haven't thought of,
so you may need to get creative when splitting nodes and creating labels.

Take this diagram for example:


```text
┌─────────┐                   ┌─────────┐
│   19    │                   │   23    │
│ Snack A │                   │ Snack B │
└────┬────┘                   └────┬────┘
     │                             │
     └──────►┌─────────────┐◄──────┘
             │ 10 eaten 16 │
             │ 6 binned 2  │
             │ 3 stored 5  │
             └─────────────┘
```
From a visual perspective, it is obvious that snack A has been eaten, binned and stored
10, 6 and 5 times respectively, and snack B has been eaten, binned and stored 16, 2 and
5 times respectively.

If we parsed this exactly as it appears, this would would not be obvious in JSON format,
since both arrows point to the same node. Furthermore, duplicating the node does not
solve the problem, since the node contains a combined count for both snacks.

To solve this problem and preserve the meaning of the diagram, we would need to split
the node into two separate nodes, one for each snack, and then split the text in each
node to only include the relevant counts for that snack.

One valid way, and also the preferred way, to do this would be creating nodes like this:

```json
...
    {
      "node_number": 3,
      "text": "10\n6\n3",
      "labels": ["eaten\nbinned\nstored"],
      "points_to": [],
    },
    {
      "node_number": 4,
      "text": "16\n2\n5",
      "labels": ["eaten\nbinned\nstored"],
      "points_to": [],
    },
...
```
A GOOD WAY TO THINK ABOUT THIS IS:
As you can see we are effectively turning the shared text into a label, and then
splitting the branch specific text (the counts in this case) into node text.
This means we don't end up with the slightly more awkward:

```json
...
    {
      "node_number": 3,
      "text": "10 eaten\n6 binned\n3 stored",
      "labels": [],
      "points_to": [],
    },
    {
      "node_number": 4,
      "text": "16 eaten\n2 binned\n5 stored",
      "labels": [],
      "points_to": [],
    },
```
Albeit still valid.

Of course there are many valid ways of doing this that would all be correct.

### Example E: Multiple shared nodes

```text
┌─────────┐                   ┌─────────┐
│   100   │                   │   100   │
│Blue cars│                   │Red cars │
└────┬────┘                   └────┬────┘
     │                             │
     └─────────────┬───────────────┘
                   │
            ┌──────┴──────┐
            │    Washed   │
            └──────┬──────┘
                   │
            ┌──────┴──────┐
            │  Polished   │
            └──────┬──────┘
                   │
     ┌─────────────┴──────────────────┐
     │                                │
┌────┴───────┐                   ┌────┴───────┐
│   20       │                   │   80       │
│ no scratch │                   │ no scratch │
└────────────┘                   └────────────┘
```

Clearly 20 no scratch is for blue cars and 80 no scratch is for red cars,
so we must duplicate the "Washed" and "Polished" nodes for each branch.

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "100\nBlue cars",
      "labels": [],
      "points_to": [3]
    },
    {
      "node_number": 2,
      "text": "100\nRed cars",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 3,
      "text": "Washed",
      "labels": [],
      "points_to": [5]
    },
    {
      "node_number": 4,
      "text": "Washed",
      "labels": [],
      "points_to": [6]
    },
    {
      "node_number": 5,
      "text": "Polished",
      "labels": [],
      "points_to": [7]
    },
    {
      "node_number": 6,
      "text": "Polished",
      "labels": [],
      "points_to": [8]
    },
    {
      "node_number": 7,
      "text": "20\nno scratch",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 8,
      "text": "80\nno scratch",
      "labels": [],
      "points_to": []
    }
}

```

## Grid like nodes

Sometimes, in our flowcharts, we will get nodes that are split into a grid like format,
with rows and columns of cells. When this happens, we have two options:

- If the grid is composed of a small number of nodes organised in some format,
  separate the cells out into the normal nodes and labels format. For this nodes
  section, just parse the nodes cells you can add attached labels in the next step.
  Use this for simple and small grids, or if the cells are clearly intended to be nodes.
- If the grid is large and complex, with many cells, and the cells and grid lines are
  clearly intended to form a table or matrix of information rather than separate nodes,
  then parse the whole grid as one node, using Markdown table syntax.

NOTE: Only use Markdown table syntax when necessary. When using markdown table syntax,
use \\n for new rows of the table. Use <br> for new lines within a cell.
For everything else follow the standard text transcription rules.

### Example F: Grid like nodes

In this example, the grid is very simple and the cells are clearly intended to be
separate nodes, so we would parse the cells as separate nodes and then add the "9:00 to
12:00" as a label in the next step.

```text
                ┌─────────────┐
                │  Attended   │
                │  Fishing    │
                └──────┬──────┘
                       │
            ┌──────────┴───────────┐
            │   9:00 to 12:00      │
            ────────────────────────
            │  caught   │  caught  |
            |  20       |  10      |
            |  with     |  with    |
            |  net      |  rod     |
            └──────────────────────┘
```

So we could parse this as:

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "Attended\nFishing",
      "labels": [],
      "points_to": [2, 3]
    },
    {
      "node_number": 2,
      "text": "caught\n20\nwith\nnet",
      labels: ["9:00 to 12:00"],
      "points_to": []
    },
    {
      "node_number": 3,
      "text": "caught\n10\nwith\nrod",
      labels: ["9:00 to 12:00"],
      "points_to": []
    },
  "additional_texts": []
}
```
Or we could alternatively include the "9:00 to 12:00" as node that points to the two
below it. Both are valid but I prefer the label version.


### Example G: Table-like grid

In this example, the grid is not a set of separate flowchart nodes. It is a
table. Each row describes one fishing period, and the columns only make sense
together.

Because the cells are part of a table, we should parse the table as one node and
preserve the table layout as plain text using Markdown table syntax.

```text
                         ┌─────────────┐
                         │  Attended   │
                         │  Fishing    │
                         └──────┬──────┘
                                │
                                ▼
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Time of day  │ Fish caught  │ Fish caught  │ Fish caught  │ Fisherman    │
│              │ with rod     │ with net     │ by noodling  │ on duty      │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 9:00-12:00   │ 10           │ 20           │ 0            │ Bill         │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 12:00-15:00  │ 8            │ 14           │ 2            │ Sue          │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 15:00-18:00  │ 6            │ 11           │ 4            │ Amira        │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "Attended\nFishing",
      "labels": [],
      "points_to": [2]
    },
    {
      "node_number": 2,
      "text": "| Time of day | Fish caught<br>with rod | Fish caught<br>with net | Fish caught<br>by noodling | Fisherman<br>on duty |\n|---|---:|---:|---:|---|\n| 9:00-12:00 | 10 | 20 | 0 | Bill |\n| 12:00-15:00 | 8 | 14 | 2 | Sue |\n| 15:00-18:00 | 6 | 11 | 4 | Amira |",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```

# Text transcription rules

- Parse numbers exactly as they appear.
- Parse node text exactly as it appears in the diagram.
- Preserve line breaks inside node text exactly as they appear in the diagram.
  Do not add new lines. Do not remove new lines. Preserve all new lines
  inside nodes.
- When two or more nodes contain similar or repeated text, transcribe each node
  independently. Do not copy line breaks, spacing, or layout from the matching
  node in another branch.
- Preserve meaningful punctuation.
- Preserve superscripts and subscripts using Unicode characters. You will often
  see ᵃ, ᵇ, ᶜ, ᵈ at the end of a word.
- Preserve superscripts in cases like 1ˢᵗ, 2ⁿᵈ, 3ʳᵈ only if the diagram itself
  uses superscripts. If the diagram writes "1st", parse it as "1st", if the
  diagram write "1ˢᵗ", parse it as "1ˢᵗ".
- Preserve line-break hyphenation exactly as it appears in the diagram,
  including the hyphen and the line break. E.g parse "cry-\ning" as "cry-\ning",
  or "follow-\nup" as "follow-\nup".
- Preserve bullet point symbols, regardless of how odd they look. If you cannot
  identify the exact bullet symbol, use "•".
- Do not invent, simplify, standardise, or rewrite text.
- Do not correct spelling under any circumstances.
- Preserve apparent mistakes exactly, including misspellings, missing spaces,
  repeated words, repeated phrases, awkward grammar, and duplicated text. Do not
  "fix" text that looks accidental or redundant. If the diagram repeats a word
  or phrase, repeat it in the output.
- Hyphenated words should not have spaces before or after the hyphen. E.g. parse
  "follow - up" as "follow-up".
- Parse tables as plain text, preserving the layout as much as possible. Do not
  convert tables into a structured format. Keep all content in a table row
  together on the same line.
- Preserve capitalisation of characters exactly as they appear in the diagram.
  E.g. parse "aBcD" as "aBcD", do not change it to "Abcd" or "ABCD".
- Sometimes you like to add up the counts in a node or counts accross nodes,
  notice that there is a mistake and then correct the counts. Do not do this.
- Do not add any LaTeX formatting, markdown formatting, or any other formatting to
  the text.

## Example Y: Do not correct counts

Below you can see a diagram where the counts in the node are wrong.
Do not correct them. Parse them as they appear.

```text
┌────────────────────────────────┐
│ 100 people lost to follow-up   │
│   90 did not respond           │
│   7 died                       │
│   7 moved away                 │
└────────────────────────────────┘
```

Clearly, 90, 7 and 7 add up to 104, not 100. However, we should still parse
the node text exactly as it appears, without correcting the counts. Our job is
to parse what the author has written, not make assumptions about what he was
supposed to write.

Thus we parse this node like so:

```json
...
    {
      "node_number": 1,
      "text": "100 people lost to follow-up\n  90 did not respond\n  7 died\n  7 moved away",
      "labels": [],
      "points_to": []
    },
...
```

## Example Z: Do not correct counts 2

In general we never correct the counts for any reason. This example just serves as
second example to make this point clear.

In the example below, it really seems like the third and fourth node should have 78
analysed and 70 analysed respectively. Still we shouldn't make assumptions and should
parse exactly what the author has written.

```text
                 ┌─────────────┐
                 │  100 people │
                 │             │
                 └─────────────┘
                       │
                       │
                       ▼
              ┌───────────────────┐
              │ 6 month Follow-Up │
              │ 10 lost           │
              │ 90 analysed       │
              └───────────────────┘
                       │
                       │
                       ▼
              ┌───────────────────┐
              │ 1 year Follow-Up  │
              │ 12 more lost      │
              │ 90 analysed       │
              └───────────────────┘
                       │
                       │
                       ▼
              ┌───────────────────┐
              │ 2 year Follow-Up  │
              │ 8 more lost       │
              │ 90 analysed       │
              └───────────────────┘

```

```json
...
    {
      "node_number": 2,
      "text": "6 month Follow-Up\n10 lost\n90 analysed",
      "labels": [],
      "points_to": [3]
    },
    {
      "node_number": 3,
      "text": "1 year Follow-Up\n12 more lost\n90 analysed",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 4,
      "text": "2 year Follow-Up\n8 more lost\n90 analysed",
      "labels": [],
      "points_to": []
    },
```
## Example W: Preserve all new lines

As an AI, you have a bias to remove new lines when it make sense
to do so. But for this task you must preserve all new lines as they appear
like so:

```text
    ┌──────────────────┐
    │ 200 participants │
    │ allo-            │
    │ cated            │
    └────────┬─────────┘
             │
             │
             ▼
    ┌──────────────────┐
    │ 10 missed call   │
    │ (10%)            │
    │ 6 no longer      │
    │ interested       │
    │ (6%)             │
    └──────────────────┘
```

```json
...
    {
      "node_number": 1,
      "text": "200 participants\nallo-\ncated",
      "labels": [],
      "points_to": [2]
    },
    {
      "node_number": 2,
      "text": "10 missed call\n(10%)\n6 no longer\ninterested\n(6%)",
      "labels": [],
      "points_to": []
    },
...
```

# Node numbering rules

Assign node numbers yourself.

- Node numbers must start at 1.
- Node numbers must increase by 1 for each node.
- Node numbers must be consecutive: 1, 2, 3, ...

# Exclusivity rule

Each piece of diagram text should belong to only one category.

If a piece of text is parsed as a node, it must not also be treated as a label
or additional text.

If a piece of text is better understood as a label, heading, caption, footnote,
legend, key, timepoint label, arm label, stage label, phase label, side label,
central timeline label, or additional text, do not include it in the node
output.
"""  # noqa: E501
