# Documentation guidance

## Structure

- In tutorials and usage guides, generally use this order: a brief introduction,
  a concrete example, then explanations. Let readers see typical package usage
  before introducing parameter options or detailed rules.
- Put parameter explanations, supported alternatives and important usage details
  after the example. State which choices the example makes without implying
  those choices are requirements for every user.
- Preserve useful sections and examples. Change only the parts that need
  improvement rather than rewriting a whole page for consistency alone.

## API reference

- When writing or updating a Python docstring for a user-facing function
  mentioned in the main guides, ensure the function is included on the relevant
  API reference page using a mkdocstrings directive. The API reference should
  display the docstring from the code. Verify that the new or updated docstring
  appears on the rendered API reference page.

## Writing style

- Prefer active voice: name who or what performs the action, such as "Flowde
  saves the results" rather than "The results are saved."
- Use specific nouns when pronouns or phrases such as "it", "that" or "the next
  stage" could refer to several things. Name the function, file or pipeline
  stage.
- Distinguish options ("you can"), requirements ("must"), recommendations
  ("we recommend") and example choices ("this example uses"). Do not present
  optional features or parameters as unconditional instructions.
- When introducing an option, briefly explain what happens without the option
  if the default behaviour helps readers decide whether to use it.
- Direct instructions can suit a clearly framed worked example, but judge the
  surrounding context. Make clear that an instruction belongs to the example;
  do not imply that every user must make the same choice.
- Keep explanations concise. Remove ambiguity by naming the relevant objects,
  rather than adding repeated explanations or unnecessary implementation detail.
