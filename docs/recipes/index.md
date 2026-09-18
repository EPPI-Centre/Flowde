# Recipes

Recipes connect several Flowde stages into a complete workflow. The individual
[pipeline guides](../pipeline/index.md) explain each function using general
example directories.

## CONSORT diagrams

The [CONSORT recipe](consort.md) uses the public example PDFs and supplied
prompts to extract, classify, rotate and parse participant flow diagrams. It
then compares the parsed results with the supplied manual annotations.

Use the website recipe with a local Python environment and an OpenAI API key.
The linked Colab notebook demonstrates the same package using workshop Azure
credentials. Model predictions can differ between runs, models and prompts.
