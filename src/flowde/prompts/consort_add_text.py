from flowde.prompts.consort import consort_nodes_prompt
from flowde.prompts.consort_labels import consort_labels_prompt_p2

consort_add_text_prompt = f"""

I have sent you an image of a participant flow diagram.
Your job will be to parse out the additional text that has not yet already be parsed
as a node or a label. Any unparsed text remaining in the image is an additional_text.
After this step, all text in the image should be parsed.

Here is the prompt used to parse the nodes:

<node_parsing_prompt>
{consort_nodes_prompt}
</node_parsing_prompt>

Here is the prompt used to parse the labels:

<label_parsing_prompt>
{consort_labels_prompt_p2}
</label_parsing_prompt>

Try to keep large chunks of text together.

# Text Transcription rules

Use the same rules as described in the nodes prompt

If an additional text is almost entirely out of the image, do not parse the text.
Do not hallucinate additional texts.
"""
