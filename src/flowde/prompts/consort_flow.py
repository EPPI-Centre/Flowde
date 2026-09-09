from flowde.prompts.consort import consort_nodes_prompt

consort_flow_prompt_p1 = f"""

I am sending you an image of a particpant flow diagram and it's parsed nodes.
Your job will be to correctly parse the flow of the nodes and send back
the nodes exactly as they were, with the added flow.

For each node, you will need to list all nodes it points to.

First, let me send you the prompt I used to parse the nodes, to help you understand the
format of the nodes and what they represent.

<node_parsing_prompt>
{consort_nodes_prompt}
</node_parsing_prompt>

Now I want you to parse the flow.

"""

consort_flow_prompt_p2 = """

For each node in the diagram, list all nodes it points to.

Do not add any nodes that are not already parsed in the nodes. Only add flow between
nodes that have already been parsed.

Nodes that do not point to any other nodes should have an empty list of nodes they point
to.

Your job will to be parse the flow, mostly based on geometrics. But if you can, try to
keep each treatment branch in series as much as possible.

# Keep in series nodes in series

In general, if a set a boxes are connected in series, with no splits in between them,
then we should usually parse their flow in series exactly as it appears.

## Example 1: Follow-ups in series.


If one follow up comes after another in the same branch, then we usually represent this
in the flow by having the first follow up point to the second follow up.

```text
                 ┌─────────────┐
                 │ 100 people  │
                 │ given drug  │
                 │  A          │
                 └─────────────┘
                       │
                       │
                       ▼
              ┌───────────────────┐
              │ 1 month Follow-Up │
              │ 90 remain         │
              │ (90%)             │
              └───────────────────┘
                       │
                       │
                       ▼
              ┌───────────────────┐
              │ 1 year Follow-Up  │
              │ 40 remain         │
              │ (40%)             │
              └───────────────────┘
                       │
                       │
                       ▼
              ┌───────────────────┐
              │ 2 year Follow-Up  │
              │ 36 remain         │
              │ (36%)             │
              └───────────────────┘

```



```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "100 people\ngiven drug\nA",
      "labels": [],
      "points_to": [2]
    },
    {
      "node_number": 2,
      "text": "1 month Follow-Up\n90 remain\n(90%)",
      "labels": [],
      "points_to": [3]
    },
    {
      "node_number": 3,
      "text": "1 year Follow-Up\n40 remain\n(40%)",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 4,
      "text": "2 year Follow-Up\n36 remain\n(36%)",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```

## Example 2: In series unconnected

Sometimes nodes may not be connected by arrows, but the flow is still clear that one
comes after the other.

```text
                 ┌─────────────┐
                 │ 100 people  │
                 │ given drug  │
                 │  A          │
                 └─────────────┘

              ┌───────────────────┐
              │ 1 month Follow-Up │
              │ 90 remain         │
              │ (90%)             │
              └───────────────────┘

              ┌───────────────────┐
              │ 1 year Follow-Up  │
              │ 40 remain         │
              │ (40%)             │
              └───────────────────┘

              ┌───────────────────┐
              │ 2 year Follow-Up  │
              │ 36 remain         │
              │ (36%)             │
              └───────────────────┘

```



```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "100 people\ngiven drug\nA",
      "labels": [],
      "points_to": [2]
    },
    {
      "node_number": 2,
      "text": "1 month Follow-Up\n90 remain\n(90%)",
      "labels": [],
      "points_to": [3]
    },
    {
      "node_number": 3,
      "text": "1 year Follow-Up\n40 remain\n(40%)",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 4,
      "text": "2 year Follow-Up\n36 remain\n(36%)",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```

# T-junctions

Usually a t-junction in the diagram indicates a split in the flow.

## Example 3: T-junctions creating a split

Lone t-junctions like this should almost always be represented as a split.

1 points to 2 and 3. 2 points to nothing and 3 points to 4.

```text
            ┌──────────────────┐
            │ 100 potential    │
            │ car buyers       │
            └────────┬─────────┘
                     │
┌────────────────┐   │
│ 80 ineligible  ├───┤
│ for loan       │   │
└────────────────┘   │
                     │
               ┌─────┴──────┐
               │ 20 eligible│
               │ for loan   │
               └─────┬──────┘
               ┌─────┴──────┐
               │ 11/20 loan │
               │ car        │
               └────────────┘
```

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "100 potential\ncar buyers",
      "labels": [],
      "points_to": [2, 3]
    },
    {
      "node_number": 2,
      "text": "80 ineligible\nfor loan",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 3,
      "text": "20 eligible\nfor loan",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 4,
      "text": "11/20 loan\ncar",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```

## Example 4: Multiple T-junctions in series should be flattened.

Sometimes an author will use multiple t-junction in series to indicate a single flow.
In this case, you should almost always flatten the t-junctions into series, with an
exception mentioned in the next section.

Furthermore, in the case that this occurs in a follow up branch, or a treatment branch
you should almost always flatten series of t-junctions into series. Some authors use
them when they are meant to indicate that one comes after the other.

```text
   ┌─────────────────┐
   │  100 kids visit │
   │  waterpark      │
   └────────┬────────┘
            │           ┌───────────────┐
            │──────────►│ 80 stay until │
            │           │ 12:00         │
            │           └───────────────┘
            │
            │           ┌───────────────┐
            │──────────►│ 60 stay until │
            │           │ 14:00         │
            │           └───────────────┘
            │
            │           ┌───────────────┐
            │──────────►│ 40 stay until │
            │           │ 16:00         │
            │           └───────────────┘
            ▼
     ┌─────────────┐
     │ 20 stay     │
     │ until 18:00 │
     └─────────────┘
```

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "100 kids visit\nwaterpark",
      "labels": [],
      "points_to": [2]
    },
    {
      "node_number": 2,
      "text": "80 stay until\n12:00",
      "labels": [],
      "points_to": [3]
    },
    {
      "node_number": 3,
      "text": "60 stay until\n14:00",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 4,
      "text": "40 stay until\n16:00",
      "labels": [],
      "points_to": [5]
    },
    {
      "node_number": 5,
      "text": "20 stay\nuntil 18:00",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```
Example 5: Exception where multiple T-junction in series should not be flattened

The only time when multiple t-junctions in series should not be flattened is when the
T-junction arrow itself is important information, that would be lost if they were parsed
in series.

Take a look at the the following diagram.

```text
   ┌─────────────────┐
   │  1000  patients │
   │  screened       │
   └────────┬────────┘
            │           ┌───────────────┐
            │──────────►│ 80 not        │
            │           │ smokers       │
            │           └───────────────┘
            │
            │           ┌───────────────┐
            │──────────►│ 40 > 65 years │
            │           │               │
            │           └───────────────┘
            │
            │           ┌───────────────┐
            │──────────►│ 80 pregnant   │
            │           │               │
            │           └───────────────┘
            ▼
     ┌─────────────┐
     │ 800         │
     │ randomized  │
     └─────────────┘
```

Clearly, the exclusion boxes themselves do not have enough information in their texts to
ascertain that they are exclusion boxes. As a result, if we parse them in series, it
just looks like we are giving some facts about the patients. But the arrows themselves
are what is telling us that these are exclusions. As a result we parse like so:

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "1000  patients\nscreened",
      "labels": [],
      "points_to": [2, 3, 4, 5]
    },
    {
      "node_number": 2,
      "text": "80 not\nsmokers",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 3,
      "text": "40 > 65 years",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 4,
      "text": "80 pregnant",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 5,
      "text": "800\nrandomized",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```

# Logical can override sloppy arrows

Diagram authors often take visual shortcuts to save space. Do not be a blind
pixel-follower.

Sometimes, a single arrow can be applied to many boxes.
Somtimes an arrow that crosses over many boxes it only meant to apply to a subset.

If you are absolutely sure that the author is just being sloppy with the arrows, then
you should parse the flow as it should be parsed.

## Example X: Implicit shared arrows for parallel boxes

Often, a diagram will feature a row of parallel boxes. The author may use a single
arrow to signal that the flow passes from the whole row or column to the next box,
instead of drawing a separate arrow from each box in the row to the next box.



```text
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ 100 floor    │   │ 100 side     │   │ 100 back     │
  │ tickets sold │   │ tickets sold │   | tickets sold │
  └──────────────┘   └──────────────┘   └──────────────┘
                            │
                            ▼
                      ┌────────────┐
                      │ 288 attend │
                      │ concert    │
                      └────────────┘
```
```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "100 floor\ntickets sold",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 2,
      "text": "100 side\ntickets sold",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 3,
      "text": "100 back\ntickets sold",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 4,
      "text": "288 attend\nconcert",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```

## Example Y: Long spanning arrows and intersecting lines

Sometimes, a long arrow crosses over several other flow lines to reach a box.
This does not necessarily mean that the long arrow applies to all flow lines it crosses
over. In this scenario, you should read the text of the destination node to determine
which nodes the long arrow applies to.

For example here, the long arrow accross the middle only applies to the floor and side
tickets, not the vip tickets, because the destination node mentions the floor and side
tickets, but not the vip tickets.

```text

  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ 100 floor    │   │ 100 side     │   │ 10 vip       │
  │ tickets sold │   │ tickets sold │   | tickets sold │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                  │                  │               ┌───────────────────────┐
         │                  │                  │               │ 3 floor didn't attend │
         ├──────────────────┼──────────────────┼───────────────┤ 4 side didn't attend  │
         │                  │                  │               └───────────────────────┘
  ┌──────┴───────┐   ┌──────┴───────┐   ┌──────┴───────┐
  │ 100 floor    │   │ 100 side     │   │ 10 vip       │
  │ tickets sold │   │ tickets sold │   | tickets sold │
  └──────────────┘   └──────────────┘   └──────────────┘

```

{
  "nodes": [
    {
      "node_number": 1,
      "text": "100 floor\ntickets sold",
      "labels": [],
      "points_to": [4, 5]
    },
    {
      "node_number": 2,
      "text": "100 side\ntickets sold",
      "labels": [],
      "points_to": [4, 6]
    },
    {
      "node_number": 3,
      "text": "10 vip\ntickets sold",
      "labels": [],
      "points_to": [7]
    },
    {
      "node_number": 4,
      "text": "3 floor didn't attend\n4 side didn't attend",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 5,
      "text": "100 floor\ntickets sold",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 6,
      "text": "100 side\ntickets sold",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 7,
      "text": "10 vip\ntickets sold",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}

If you are unsure, you can be safe and apply the long arrow to all the boxes it crosses
over.

# Terminal sounding nodes are not always terminal

As an AI, you have a strong semantic bias to assume that any terminal sounding node,
such as those containing words like, "Lost", "Excluded", "Failed", or "Discarded" is
a terminal node (i.e., it points to nothing). However, in many reporting workflows,
these items are mathematically carried forward into another analysis node and the
authors will use arrows to indicate that.

If the diagram draws an explicit arrow or line from an off-ramp terminal node to a
downstream node, you should respect the arrow. If the arrow connects them, the off-ramp
node points_to the downstream node.

Example J: Off-ramps can point to downstream nodes.

```text
                  ┌────────────┐
                  │ 200 given  │
                  │ drug a     │
                  └────────────┘
                        │
                        │
              ┌─────────┴──────────┐
              │                    │
              │                    │
       ┌──────┴───────┐     ┌──────┴───────┐
       │ 160 attend   │     │ 40 missed    │
       │ follow up    │     │ follow up    │
       └──────┬───────┘     └──────┬───────┘
              │                    │
              ▼                    ▼
          ┌───────────────────────────┐
          │   189 analysed            │
          │                           │
          └───────────────────────────┘

```

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "200 given\ndrug a",
      "labels": [],
      "points_to": [2, 3]
    },
    {
      "node_number": 2,
      "text": "160 attend\nfollow up",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 3,
      "text": "40 missed\nfollow up",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 4,
      "text": "189 analysed",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```

##Example K: T-junctions aren't alway terminal

```text
        ┌────────────┐
        │ 200 units  │
        │ processed  │
        └─────┬──────┘          ┌──────────────────────┐
              │                 │ 2 bypassed initial   │
              ├─────────────────┤ scan                 │
              │                 └──────────┬───────────┘
       ┌──────┴────────────┐               │
       │ 198 passed        │               │
       │ initial scan      │               │
       └──────┬────────────┘               │
              │                            │
              ├────────────────────────────┘
              │
   ┌──────────┴────────────────┐
   │   200 logged in system    │
   │                           │
   └───────────────────────────┘

```

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "200 units\nprocessed",
      "labels": [],
      "points_to": [2, 3]
    },
    {
      "node_number": 2,
      "text": "198 passed\ninitial scan",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 3,
      "text": "2 bypassed initial\nscan",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 4,
      "text": "200 logged in system",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```

# T-junctions can be detours

## Example W: T-junction is a detour

Sometimes an author will draw a continuous vertical line down a main trunk of nodes, but
between two nodes, they will draw an arrow out to a side-box, or an arrow from that
side-box returning to the main trunk. This often looks like a T-junction, but may not be
a split.

If it represents an intermediate chronological step, then it should be parsed as such.
Flatten it into series and do not let continuous background lines trick you into adding
direct shortcut connections that bypass the detour.

```text
      ┌──────────────┐
      │ Went to shop │
      └────┬─────────┘
           │         ┌──────────────────┐
           │◄────────┤ bought ice cream │
           │         └──────────────────┘
           ▼
      ┌───────────┐
      │ Went home │
      └───────────┘
```

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "Went to shop",
      "labels": [],
      "points_to": [2]
    },
    {
      "node_number": 2,
      "text": "bought ice cream",
      "labels": [],
      "points_to": [3]
    },
    {
      "node_number": 3,
      "text": "Went home",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```

# X-junctions

In complex state-transition diagrams, arrows will frequently cross over each other
diagonally, forming an 'X' shape. The lines do not merge, split, or interact at this
intersection; they are merely passing over one another in 2D space.

When tracing a line that crosses another, you must follow the line straight through the
intersection along its continuous trajectory/angle. Do not "switch tracks", and do not
assume the flow merges.

## Example K: X-junctions do not merge flow

```text
          Source A    Source B
              |         |
               |       |
                |     |
                 |   |
                  | |   <-- The lines cross here. Do NOT turn!
                 |   |
                |     |
               |       |
              |         |
         Dest C        Dest D

```

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "Source A",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 2,
      "text": "Source B",
      "labels": [],
      "points_to": [3]
    },
    {
      "node_number": 3,
      "text": "Dest C",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 4,
      "text": "Dest D",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```

# The flow serves to relate nodes

Visually, it can be very obvious which nodes are associated with which branches of the
diagram. However, sometimes if we only parse exactly the arrows that exist, we can miss
parsed nodes that obviously belong to a branch but that are not actually connected by
arrows.

In such cases, add a connection in the place you feel is most appropriate so that the
information of the unconnected node is not lost.

In general, any parsed node that does not make sense independently MUST be parsed as a
connected node so that it makes sense. If you think a label has been mistakenly parsed
as a node, you can parse the flow as the label pointing to the node it applies to.

## Example S

For us, it's very clear that the left follow up branch belongs to the intervention group
and the right follow up branch belongs to the control group.
In such a scenario, if we do not attach them to the branches via arrows, then we will
lose the information that the follow up nodes belong to the intervention and control
groups respectively as there isn't enough information inside the node texts to determine
which is which.

```text
                  ┌────────────┐
                  │  200       │
                  │ randomized │
                  └────────────┘
                        │
                        │
        ┌───────────────┴───────────────┐
        │                               │
        │                               │
 ┌──────┴───────┐                ┌──────┴───────┐
 │ 100          │                │ 100          │
 │ Intervention │                │ Control      │
 └──────────────┘                └──────────────┘




        ┌──────────────┐                ┌──────────────┐
        │ 80 1 month   │                │ 76 1 month   │
        │   follow up  │                │   follow up  │
        │              │                │              │
        │ 54 6 month   │                │ 49 6 month   │
        │   follow up  │                │   follow up  │
        └──────────────┘                └──────────────┘
```

As a result, it makes sense to parse it like so:


```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "200\nrandomized",
      "labels": [],
      "points_to": [2, 3]
    },
    {
      "node_number": 2,
      "text": "100\nIntervention",
      "labels": [],
      "points_to": [4]
    },
    {
      "node_number": 3,
      "text": "100\nControl",
      "labels": [],
      "points_to": [5]
    },
    {
      "node_number": 4,
      "text": "80 1 month\nfollow up\n\n54 6 month\nfollow up",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 5,
      "text": "76 1 month\nfollow up\n\n49 6 month\nfollow up",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```

# Items can be duplicated (No Conservation of Mass)

As an AI, you have a logical bias to assume that items are strictly conserved.
For example, if a source box contains 100 items and splits into two paths,
you expect those items to be divided (e.g., 50 left, 50 right).

However, sometimes, the FULL amount of the a node is sent down MULTIPLE paths
simultaneously. The items are effectively "double-counted" in the downstream boxes.

If visual arrows show a source pointing to multiple destinations, unless you are
absolutely certain that this is not the case, you should respect all drawn arrows.

## Example N: Double-Counting Paths

```text
               ┌────────────┐
               │ Source A   │
               │ 100 items  │
               └─┬────────┬─┘
                 │        │
                 ▼        ▼
        ┌──────────┐    ┌──────────┐
        │ Dest B   │    │ Dest C   │
        │ 100 items│    │ 100 items│
        └──────────┘    └──────────┘
```

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "Source A\n100 items",
      "labels": [],
      "points_to": [2, 3]
    },
    {
      "node_number": 2,
      "text": "Dest B\n100 items",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 3,
      "text": "Dest C\n100 items",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": []
}
```



"""

consort_flow_prompt = consort_flow_prompt_p1 + consort_flow_prompt_p2
