# DeepSeek Harness profile

`dsh` is an everything-is-a-plugin agent harness (Cordis). We use it for two things:

1. **Headless intake** — `nerve intake <source> --model dsh` runs `dsh --profile headless -p <prompt>`.
2. **Role agents on a team** — the `nerve` profile exposes the pipeline verbs as the only shell tools, and the
   session log (`session.vN.jsonl`, append-only) is kept beside Nerve's own `episodes.v1.jsonl`. Both follow the
   same rule: everything the model saw is reconstructable from the log.

Setup: `npx @deepseek-ai/dsh web` once to create `$DSH_HOME`, then copy `cordis.patch.yml` to
`$DSH_HOME/profiles/nerve/cordis.patch.yml`. Inspect with `dsh --profile nerve --dump-config`.
