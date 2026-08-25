# Team Workflow

This is the common workflow every team member (Namitha, Manas, Arushi,
Nidhi, Khushi) follows for every task, from picking it up to merging it.
It combines the git steps from `docs/git-workflow.md` with how Claude Code
and Antigravity fit into each task (CLAUDE.md §9–§13).

## Before starting

1. Make sure your current work is committed or safely stashed
   (`git status` should be clean, or `git stash -u` if you have
   in-progress changes you're not ready to commit).
2. Switch to `main`:
   ```bash
   git checkout main
   ```
3. Pull latest `main`:
   ```bash
   git pull
   ```
4. Confirm the working tree is clean:
   ```bash
   git status
   ```
5. Create a **new** task branch (see `docs/git-workflow.md` for naming):
   ```bash
   git checkout -b feature/yourname-short-task
   ```
6. Start Claude Code from the project root.
7. Tell Claude Code your full name.
8. Claude Code reads `CLAUDE.md`, `docs/decisions.md`, `docs/TASK.md`, and
   any other relevant docs.
9. Claude Code determines the current phase and your assigned task.
10. Claude Code explains the task to you in plain language.
11. Claude Code generates a copy-paste-ready implementation prompt for
    Antigravity (see `docs/team-member-start-prompt.md`).
12. Only then begin implementation.

## During work

- Stay on your task branch — don't hop back to `main` or someone else's
  branch mid-task.
- Work only on the assigned task; resist scope creep.
- Do not silently modify another member's module (CLAUDE.md §2, §12).
- Do not silently change a module interface (CLAUDE.md §7) — if a change
  is needed, name it, explain the impact, and get approval first.
- Do not implement future-phase functionality (see
  `docs/development-phases.md` for what's in/out of the current phase).
- Run relevant tests as you go, not just at the end.
- Use Claude Code for planning, review, and debugging.
- Use Antigravity for scoped implementation, following the brief Claude
  Code generated.

## Before finishing

1. `git status`
2. `git diff`
3. Run tests (e.g. `pytest tests/<your-module>` and/or the full suite)
4. Ask Claude Code to review the diff against the brief and against
   CLAUDE.md.
5. Fix anything flagged.
6. Run tests again.
7. Commit:
   ```bash
   git add <specific files>
   git commit -m "feat: <clear imperative message>"
   ```
8. Push the branch:
   ```bash
   git push -u origin feature/yourname-short-task
   ```
9. Open a Pull Request into `main`.
10. Get at least one teammate review.
11. Communicate any changes and integration implications (e.g. "I added an
    optional key to my output dict" or "this doesn't affect anyone else's
    module").

## After merge

1. Switch to `main`:
   ```bash
   git checkout main
   ```
2. Pull latest `main`:
   ```bash
   git pull
   ```
3. Delete the old branch if it's fully merged and no longer needed:
   ```bash
   git branch -d feature/yourname-short-task
   ```
4. Create a new task branch for the next task (back to "Before starting").

## Where things live

- **Your assigned task and files:** `docs/TASK.md` §10–§11, and
  `docs/architecture.md` §4 (module boundaries).
- **What's in/out of scope for the current phase:**
  `docs/development-phases.md`.
- **Shared data contracts between modules:** `docs/module-interfaces.md`.
- **Approved team decisions:** `docs/decisions.md`.
- **The common starting prompt for Claude Code:**
  `docs/team-member-start-prompt.md`.
