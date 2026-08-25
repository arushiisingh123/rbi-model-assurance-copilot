# Team Member Start Prompt

This is the **common prompt** all five team members (Namitha, Manas,
Arushi, Nidhi, Khushi) use to start a Claude Code session for a new task.
Copy the block below, replace `[FULL NAME]`, and paste it in.

> **Note on file names:** the repository's decisions log lives at
> `docs/decisions.md` (there is no separate root-level `DECISIONS.md`
> file). The prompt below refers to it by its actual path so Claude Code
> can find it directly.

---

```text
I am [FULL NAME].

Read CLAUDE.md, docs/decisions.md, docs/TASK.md, and any other relevant
documentation in docs/ (architecture.md, development-phases.md,
module-interfaces.md, git-workflow.md, team-workflow.md).

Determine:

1. Current phase.
2. My ownership (which module(s) I'm responsible for).
3. My current task.
4. What I should do next.
5. Files I should modify.
6. Files I must not modify.
7. Dependencies (what my task needs from other modules, and what depends
   on mine).
8. Tests required.
9. Acceptance criteria.
10. The correct branch name for this task (per docs/git-workflow.md).

Explain my task in beginner-friendly language before doing anything else.

Then generate a copy-paste-ready implementation prompt for Antigravity.

The Antigravity prompt must:

- Tell it to read CLAUDE.md.
- Tell it to read docs/decisions.md.
- Tell it to read docs/TASK.md.
- Tell it to read relevant docs/ files.
- State the exact task.
- State allowed files.
- State forbidden files.
- Tell it to explain its plan before coding.
- Tell it to implement only the assigned task.
- Tell it to avoid future-phase functionality.
- Tell it to run tests.
- Tell it to report files changed.
- Tell it to report tests run and their results.
- Tell it to report anything another team member needs to know (e.g.
  interface changes, new dependencies, cross-module impacts).

Do not start implementing yourself yet — wait for me to confirm the plan
first.
```

---

## Why this is the same prompt for everyone

The prompt is generic on purpose: `[FULL NAME]` is the only thing that
changes. Claude Code determines the rest (phase, ownership, task, files,
branch name) by reading the repository and `CLAUDE.md`/`docs/`, matching
that name against the team table in CLAUDE.md §2. This keeps every
teammate's session grounded in the same source of truth instead of five
different half-remembered instructions.
