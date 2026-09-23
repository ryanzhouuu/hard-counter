# Project Rules

## Repository discipline

- Never track or commit local working documents. Keep them under ignored `docs/.local/`.
- Never mention untracked roadmap or design documents in tracked files, commit messages, pull requests, or other external-facing text.
- Commit frequently. Keep each commit at or below 250 changed source lines and 250 changed test lines when the work can be divided coherently.
- Use a single-line commit message in the form `<type>: <imperative summary>`. Do not add a body.

## Code quality

- Prefer clear names, small cohesive functions, and straightforward control flow over explanatory comments.
- Keep comments and docstrings concise. Use them only for non-obvious intent, constraints, invariants, tradeoffs, or external behavior that code cannot express clearly.
- Do not add slop comments. Never narrate obvious control flow, restate identifiers, paraphrase implementations, repeat type information, or add decorative section commentary.

## Evidence and decisions

- Ground responses, plans, designs, and implementations in repository state, tool output, tests, documented requirements, and authoritative primary sources.
- Inspect relevant code, configuration, data, or documentation before making factual claims. Label assumptions explicitly.
- Ask the user before an ambiguous choice that could materially affect architecture, behavior, scope, data semantics, compatibility, cost, or user experience.
- Ask the user when a required claim or decision cannot be grounded in available evidence. Do not invent details to keep moving.
