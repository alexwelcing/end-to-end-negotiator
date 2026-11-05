"""Prompts that turn an interface description into a behavior spec.

Intake accepts three descriptions of the same thing: the code, what it looks
like, or how it works in words. A harness agent (Claude Code on the user's
subscription, or DeepSeek Harness) reads the input and writes a ``.nerve``
spec. The spec is then type-checked and self-checked by ``nerve check``; the
agent iterates until it passes. Nothing downstream ever sees the input again.
"""

SPEC_GRAMMAR = """
A .nerve spec is YAML:

nerve: 1
name: <identifier>
state:                      # every field is typed; prefer finite types
  <field>: {type: int, range: [lo, hi]} | {type: enum, values: [...]} | {type: bool} | {type: float, range: [lo, hi], step: s}
actions:
  <action>: {}  |  {params: {<p>: <type as above>}}
transitions:                # pure expressions over fields, params, and old.<field>
  <action>: { when: "<guard>", <field>: "<expr>", ... }
invariants: ["<expr over state>", ...]
ensures:    [{after: <action>|omit, holds: "<expr over new state, params, old.*>"}]
futures:    { <gesture>: {kind: closest|between|vary, over: <field>, values: [...]} }   # dragology
learn:      [{name, action, fallback: {<field>: "<expr>"}, examples: [{state: {...}, next: {...}}]}]
initial:    { <field>: <value> }

Expression language: literals, field names, params, old.<field>, + - * / // % **, comparisons (chained ok),
and/or/not, `x if c else y`, min max abs clamp len int bool. Strings in single quotes. No other syntax.
Rules: every action needs a transition (or a learn hole). Guards that fail leave state unchanged — write
ensures clauses with that in mind ("value == to or not old.grabbed"). Behaviour the input shows but you cannot
write as an expression goes in a `learn` hole with a safe fallback and ≥5 examples.
"""

FROM_CODE = """You are converting SOURCE CODE of an interface into a Nerve behavior spec.
Read the code. Identify the state it holds, the actions a user can take, and exactly how each action changes state.
Write invariants the code maintains and ensures clauses per action. Do not describe the code; describe the behaviour.
{grammar}
Return only the YAML spec in a ```yaml block, then run `python -m nerve check <file>` on it and fix problems until it reports 0.
SOURCE CODE ({path}):
{body}
"""

FROM_IMAGE = """You are converting a SCREENSHOT of an interface into a Nerve behavior spec.
Look at the image at {path}. Name every control and every piece of visible state. For each control, decide what state
it changes and to what. Where the screenshot leaves behaviour ambiguous, prefer a `learn` hole with a conservative fallback
over guessing. Use `futures` for anything draggable.
{grammar}
Return only the YAML spec in a ```yaml block, then run `python -m nerve check <file>` on it and fix problems until it reports 0.
"""

FROM_PROSE = """You are converting a WRITTEN DESCRIPTION of how an interface works into a Nerve behavior spec.
Treat the text as requirements. Every "must", "never", "always" becomes an invariant or ensures clause. Every verb a
user performs becomes an action. Bound every numeric field.
{grammar}
Return only the YAML spec in a ```yaml block, then run `python -m nerve check <file>` on it and fix problems until it reports 0.
DESCRIPTION:
{body}
"""
