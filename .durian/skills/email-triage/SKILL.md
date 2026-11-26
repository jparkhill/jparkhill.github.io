---
name: email-triage
description: Triage an inbox — summarize threads, flag what needs a reply, and propose actions. Use when the user asks about email, inbox, or messages.
tools: [read_file, write_file, memory_search, memory_append, note_add, todo_add]
requires: []
---
# Email triage

You help the user stay on top of email. Inputs may be raw pasted messages or a
mailbox export placed in the workspace.

## Workflow
1. Read the provided messages (files or pasted text).
2. Group by thread and sender. Identify the *ask* in each.
3. Classify: needs-reply now / delegate / archive / schedule.
4. Produce a short digest:
   - **Action needed** (with a one-line suggested reply)
   - **FYI**
   - **Ignored**
5. Record anything time-sensitive with the todo/note tools.
6. Never send email without explicit approval. Draft only.

Keep the digest skimmable. Prefer bullet points.
