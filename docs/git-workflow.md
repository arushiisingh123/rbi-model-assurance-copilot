# Git Workflow

This is the standalone git-workflow reference (CLAUDE.md §17 lists it as a
required doc). It reflects the **approved, current** model recorded in
`docs/decisions.md` ("Git branching model: per-task branches") and now
matches CLAUDE.md §8. For the full day-to-day sequence of what to do before,
during, and after a task (including how Claude Code and Antigravity fit
in), see `docs/team-workflow.md`.

## Branching model

One short-lived branch **per task**, not one long-lived branch per person.

Branch naming: `feature/<name>-<short-task>`

Examples:

- `feature/namitha-model-foundation`
- `feature/manas-explainability-foundation`
- `feature/arushi-fairness-drift-foundation`
- `feature/nidhi-rbi-foundation`
- `feature/khushi-api-foundation`

`main` is the shared source of truth. Never develop feature work directly
on `main`.

## Before starting work

```bash
git checkout main
git pull
git checkout -b feature/yourname-short-task
```

Confirm `git status` shows a clean working tree before branching.

## Before committing

- Review the changes (`git status`, `git diff`).
- Run relevant tests/checks.
- Confirm unrelated files were not modified.
- Confirm the work belongs to the assigned task.
- Use a clear, imperative commit message.

```bash
git add <specific files>
git commit -m "feat: add explainability foundation"
git push -u origin feature/yourname-short-task
```

Avoid `git add .` / `git add -A` when unsure what's staged — add specific
files so you don't accidentally commit another module's in-progress work,
`.venv/`, or cache files.

## Pull requests

Merge through a Pull Request whenever practical. Before merging:

- Relevant tests pass.
- The task is within scope (Phase 0 stays Phase 0, etc.).
- Another team member has reviewed the change.
- Cross-module impacts have been considered and communicated.

Do not merge unfinished or experimental work into `main`.

## Merge conflicts

If a conflict touches another person's module:

- Do not blindly pick one side.
- Inspect both changes.
- Explain the conflict.
- Involve the affected team member before resolving it.

## After merge

```bash
git checkout main
git pull
git branch -d feature/yourname-short-task
```

Branches are short-lived and numerous under this model — delete them after
merge rather than accumulating stale branches.
