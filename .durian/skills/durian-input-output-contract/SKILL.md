---
name: durian-input-output-contract
description: Use when you are a child harness in a Durian agtask. Defines who owns inputs, outputs, artifacts, and the final handoff, and forbids unlogged side channels between harnesses.
---

# Durian input/output contract

Durian is a **meta-harness**. It calls child harnesses (pi, Claude Code, Codex, or
its own self-exec loop) but retains ownership of the work unit and its evidence.
You are the child; your job is to produce a bounded, inspectable result without
taking ownership of the surrounding system.

## Inputs belong to Durian

Durian owns and records:

- the agtask goal, method, verification command, and required artifacts
- the workspace, repository, branch, and baseline commit
- the input datasets and their provenance
- secrets and credential selection
- permissions, approvals, budgets, and quota
- the run record, transcript, events, and final report

Do not replace, relocate, or silently mutate Durian's inputs. Do not pull new
external data, rewrite the verification command, change the workspace, or switch
credentials unless the agtask explicitly authorizes it. If an input is missing or
wrong, report it to Durian instead of improvising a new source.

## Outputs belong to Durian too

Produce outputs only where the agtask says:

- code changes in the workspace
- declared verification artifacts and required report/notebook files
- verification output and command results
- a concise final handoff

Durian owns the report, artifact registry, acceptance decision, and integration.
By default, Durian finalization pushes the task branch. Follow an explicit
operator or repository instruction to push your assigned branch after each
commit; that does not authorize merging or pushing main, deployment, or accepting
your own work. Do not merge, accept your own work, or declare the agtask complete.

## Handoff format

End your final message with:

- **Status:** complete, partial, blocked, or needs review
- **Changed files:** scoped paths
- **Verification:** exact command and result
- **Artifacts:** paths, sizes, and hashes when available
- **Decisions / assumptions:** anything a reviewer must know
- **Blockers / next step:** what Durian or the operator should do

Keep it factual. A long narrative is less useful than an inspectable artifact and
a short handoff.

## No side channels

All coordination with other harnesses, agents, machines, or the operator goes
through Durian's persisted A2A channel (the `durian-agent-mesh` skill has the
commands). Do not hand off files, patches, or instructions through SSH, shared
directories, git commits, or another harness's session. Never send secrets
through A2A; refer to a secret by name and let Durian resolve it.

If you are working on Durian itself, preserve this contract: the meta-harness must
remain the owner of inputs, outputs, and the communication record.
