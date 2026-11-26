---
name: credential-walkthrough
description: When a tool, command or task fails because a credential is missing, locked or not mapped into its secret-exec profile ("not in the secret store", "no secret profile", a 401 from a vendor), or the operator asks "what tokens are missing?", offer the credential walk-through - run secrets_doctor (or tell the operator to send /tokens) instead of asking for the value in chat.
tools: [secrets_doctor, secret_list]
requires: []
---
# Credential walk-through

A credential never passes through you. You find the gap and hand the operator to
the walk-through; the operator types every value at a hidden terminal prompt.

1. A tool or command failed for a credential (`not in the secret store: X`,
   `no secret profile P`, a vendor 401/403 under `durian secret-exec`), or the
   operator asked which tokens are missing: call `secrets_doctor`. It lists every
   declared credential that is missing, locked or unmapped (names and states only)
   and sends the operator the first step in Telegram.
2. Tell the operator in one line what is missing and that the first step is in
   Telegram (or, with no Telegram, that `durian secrets doctor` in a terminal
   lists the same gaps and the exact commands).
3. Never ask for, accept or repeat a value, and never edit `durian.config.json`:
   a mapping step shows the operator the exact lines to add themselves.
4. If the credential is not declared anywhere (the doctor reports no gap for it),
   say so: the project needs a `durian.requirements.json` entry
   (wiki/internal/secrets.md, "Credential doctor").

The operator can leave the walk-through at any step; `/tokens` (or the tool
again) resumes it at the same step.
