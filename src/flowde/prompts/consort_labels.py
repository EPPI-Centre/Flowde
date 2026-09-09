from flowde.prompts.consort import consort_nodes_prompt

consort_labels_prompt_p1 = f"""

I am sending you an image of a particpant flow diagram and it's parsed nodes and
flow.
Your job will be to correctly add the labels to the nodes and send back
the nodes exactly as they were, with their added labels.

First, let me send you the prompt I used to parse the nodes, to help you understand the
format of the nodes and what they represent.

<node_parsing_prompt>
{consort_nodes_prompt}
</node_parsing_prompt>

Now I want you to parse the labels.

"""

consort_labels_prompt_p2 = """
# What is a label

A label is any piece of text left in the diagram that:
- has not been parsed as a node, and
- cannot be applied equally to all
  nodes in the diagram. If a piece of text left can be applied equally to all nodes,
  then it is not a label, but rather an additional text.

A label typically applies to a subset of nodes, such as a branch in the diagram,
a row of nodes, a column of nodes, or a single node.

Never hallucinate a label. Only apply a label if it exists in the diagram. For example,
if you know that a node represents a certain step in the trial, but there is no such
label in the image, then do not add the label to the node.

Some images will have chunks of text explaining the meaning of special characters,
superscripts or subscripts. These are not labels but rather additional text.

Anything that has already been parsed as a node cannot be a label. Do not reuse
nodes as labels.


# Which labels apply to which nodes

Mostly use your judgement to determine which labels apply to which nodes.

This will be done mostly visually, but if you read a label and it makes no sense to
apply it to the node it looks like it is applying to,  then it probably doesn't apply.
So try to look at a combination of the node text, the label location and human visual
rules, to determine which labels apply to which nodes.

## Enrollment Convention

As a convention we usually do not apply the `allocation` label (or equivalent)
to the node that splits the study items
into the different branch arms. Ususally this node is named `randomization` or
`randomisation`, but it can be named differently. Instead, as convention, if the,
`Enrollment` label is present, we apply that label.

We typically apply 'allocation` type labels only after the nodes have split, into their
allocated branches.

### Example 1: Enrollment Convention example

In the diagram below, `Allocation` is clearly written on the 3rd node.
But for our convention, we will ignore this and apply `Enrollment` to the 3rd node and
`Allocation` will still apply to the 4th, 5th and 6th nodes.

```text

Enrollment        ┌─────────────┐
                  │  150 food   │
                  │  testers    │
                  └──────┬──────┘
                         │             ┌───────────────┐
                         │────────────►│ 42 excluded   │
                         │             │ (failed blind │
                         │             │ taste test)   │
                         │             └───────────────┘
Allocation         ┌─────┴─────┐
                   │ 102 food  │
                   │ testers   │
                   └─────┬─────┘
                         │
      ┌──────────────────┼────────────────────┐
      │                  │                    │
┌────────────┐     ┌────────────┐        ┌────────────┐
│  34 given  │     │ 34 given   │        │ 34 given   │
│  crisps    │     │ chips      │        │ mash       │
└────────────┘     └────────────┘        └────────────┘
```

```json
...
{
  "nodes": [
    {
      "node_number": 1,
      "text": "150 food\ntesters",
      "labels": ["Enrollment"],
      "points_to": [2, 3]
    },
    {
      "node_number": 2,
      "text": "42 excluded\n(failed blind\ntaste test)",
      "labels": ["Enrollment"],
      "points_to": []
    },
    {
      "node_number": 3,
      "text": "102 food\ntesters",
      "labels": ["Enrollment"],
      "points_to": [4, 5, 6]
    },
    {
      "node_number": 4,
      "text": "34 given\ncrisps",
      "labels": ["Allocation"],
      "points_to": []
    },
    {
      "node_number": 5,
      "text": "34 given\nchips",
      "labels": ["Allocation"],
      "points_to": []
    },
    {
      "node_number": 6,
      "text": "34 given\nmash",
      "labels": ["Allocation"],
      "points_to": []
    }
  ],
  "additional_texts": []
}
...
```

### Example 2: Never hallucinate a label

Consider this diagram that is almost identical to the previous one, but it has
the labels missing. In this case, we should not add any labels.

```text

                  ┌─────────────┐
                  │  150 food   │
                  │  testers    │
                  └──────┬──────┘
                         │             ┌───────────────┐
                         │────────────►│ 42 excluded   │
                         │             │ (failed blind │
                         │             │ taste test)   │
                         │             └───────────────┘
                   ┌─────┴─────┐
                   │ 102 food  │
                   │ testers   │
                   └─────┬─────┘
                         │
      ┌──────────────────┼────────────────────┐
      │                  │                    │
┌────────────┐     ┌────────────┐        ┌────────────┐
│  34 given  │     │ 34 given   │        │ 34 given   │
│  crisps    │     │ chips      │        │ mash       │
└────────────┘     └────────────┘        └────────────┘
```
So we end up with:

```json
...
{
  "nodes": [
    {
      "node_number": 1,
      "text": "150 food\ntesters",
      "labels": [],
      "points_to": [2, 3]
    },
    {
      "node_number": 2,
      "text": "42 excluded\n(failed blind\ntaste test)",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 3,
      "text": "102 food\ntesters",
      "labels": [],
      "points_to": [4, 5, 6]
    },
    {
      "node_number": 4,
      "text": "34 given\ncrisps",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 5,
      "text": "34 given\nchips",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 6,
      "text": "34 given\nmash",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
...
```


## Splitting and Merging labels

As you have seen in the node parsing prompt, for some nodes that contain rows of
different items, you may need to combine labels into a single label in order to
correctly convey the information in JSON format. Remember, our goal is to accurately
represent the information in the image in JSON format, not parse it exactly as
it appears.

## A Main label and Secondary labels

Sometimes there is a main label for a section of nodes, then a series of sub labels
within that section. Make sure you apply both the main label and the sub labels to any
nodes which the sub labels apply.

For example:

```text
                   ┌──────────────────────────────┐
                   │ Assessed for eligibility     │
                   │ (n = 120)                    │
                   └──────────────┬───────────────┘
                                  │
            ┌─────────────────────┴─────────────────────┐
            │               ┌────────────┐              │
            │               | Allocation |              │
            │               └────────────┘              │
            ▼                                           ▼
┌──────────────────────────┐              ┌──────────────────────────┐
│ Allocated to intervention│              │ Allocated to control     │
│ (n = 60)                 │              │ (n = 60)                 │
└────────────┬─────────────┘              └────────────┬─────────────┘
             │              ┌────────────┐             │
             │              | Follow-Up  |             │
             │              └────────────┘             │
             ▼                                         ▼
┌──────────────────────────┐              ┌──────────────────────────┐
│ Completed follow-up      │   6 month    │ Completed follow-up      │
│ (n = 55)                 │              │ (n = 57)                 │
└────────────┬─────────────┘              └────────────┬─────────────┘
             │                                         │
             │                                         │
             ▼                                         ▼
┌──────────────────────────┐              ┌──────────────────────────┐
│ Completed follow-up      │   1 year     │ Completed follow-up      │
│ (n = 51)                 │              │ (n = 54)                 │
└──────────────────────────┘              └──────────────────────────┘

```


Would have the follow-up nodes parsed like this:

```json
...
    {
      "node_number": 4,
      "text": "Completed follow-up\n(n = 55)",
      "labels": [
        "Follow-Up",
        "6 month"
      ],
      "points_to": [6]
    },
    {
      "node_number": 5,
      "text": "Completed follow-up\n(n = 57)",
      "labels": [
        "Follow-Up",
        "6 month"
      ],
      "points_to": [7]
    },
    {
      "node_number": 6,
      "text": "Completed follow-up\n(n = 51)",
      "labels": [
        "Follow-Up",
        "1 year"
      ],
      "points_to": []
    },
    {
      "node_number": 7,
      "text": "Completed follow-up\n(n = 54)",
      "labels": [
        "Follow-Up",
        "1 year"
      ],
      "points_to": []
    }
...
```

## Multiple Labelled diagrams in one image

Sometimes, the author will include multiple diagrams in the same image and label each
one. In such a case, for each diagram in the image, apply the diagram label to all nodes
in the diagram.

```text

A

          ┌────────────────┐
          │ 100 randomised │
          └───────┬────────┘
       ┌──────────┴──────────┐
       │                     │
       ▼                     ▼
┌──────────────┐      ┌─────────────┐
│ 50 treatment │      │ 50 control  │
└──────────────┘      └─────────────┘



B

          ┌────────────────┐
          │ 100 randomised │
          └───────┬────────┘
       ┌──────────┴──────────┐
       │                     │
       ▼                     ▼
┌──────────────┐      ┌─────────────┐
│ 80 treatment │      │ 20 control  │
└──────────────┘      └─────────────┘

```

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "100 randomised",
      "labels": [
        "A"
      ],
      "points_to": [
        2,
        3
      ]
    },
    {
      "node_number": 2,
      "text": "50 treatment",
      "labels": [
        "A"
      ],
      "points_to": []
    },
    {
      "node_number": 3,
      "text": "50 control",
      "labels": [
        "A"
      ],
      "points_to": []
    },
    {
      "node_number": 4,
      "text": "100 randomised",
      "labels": [
        "B"
      ],
      "points_to": [
        5,
        6
      ]
    },
    {
      "node_number": 5,
      "text": "80 treatment",
      "labels": [
        "B"
      ],
      "points_to": []
    },
    {
      "node_number": 6,
      "text": "20 control",
      "labels": [
        "B"
      ],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```


# Text transcription rules

Use the exact same transcription rules for labels as the one you used for nodes
in the `# Text transcription rules` section of the node parsing prompt.
"""

consort_labels_prompt = consort_labels_prompt_p1 + consort_labels_prompt_p2
