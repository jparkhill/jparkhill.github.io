---
name: github-artifacts
description: Version artifacts and long-running job output with git, and prepare private-repo commits. Use when the user mentions repos, commits, artifacts, or versioning.
tools: [shell, git_status, git_commit, write_file, read_file]
requires: [bin:git]
---
# GitHub artifacts

## Workflow
1. Inspect state with `git_status`.
2. Organize artifacts under a clear directory (e.g. `projects/<name>/artifacts/`).
3. Write a short `NOTES.md` describing what changed and why.
4. Commit with a descriptive message via `git_commit`. Push only when asked.

Keep commits focused. Never commit secrets or credentials.
