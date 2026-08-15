---
name: git-commit
description: Commit discipline for this repo. Invoke BEFORE and AFTER every code change. Ensures the tree is clean before new work starts (committing any stray changes first), then commits the new change. All commit subjects use the form "claude - <code-change>".
---

# git-commit

Enforces the commit-before / commit-after discipline for this repository. Every code
change lands as its own commit, and no change is ever mixed with uncommitted work that
preceded it.

## When to run

Run this **every time a code change is made** — before starting the change and again
after finishing it. Never batch several unrelated changes into one commit.

## Procedure

### Step 1 — Pre-flight: is the tree clean?

```bash
git status --porcelain
```

- **Empty output** → tree is clean. Go to Step 2.
- **Non-empty output** → there are uncommitted changes that predate the work you are
  about to do. Commit them **first**, on their own:
  1. Inspect what changed so the message is accurate — `git status` and `git diff` (plus
     `git diff --staged` if anything is already staged).
  2. Stage and commit with a subject describing *those* pending changes:

     ```bash
     git add -A
     git commit -m "$(cat <<'EOF'
     claude - <describe the pre-existing pending change>

     Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
     EOF
     )"
     ```

  Do not describe the change you are *about* to make — this commit is for work already
  in the tree.

### Step 2 — Make the code change

Do the actual work. Keep it scoped to one logical change.

### Step 3 — Commit the change

```bash
git add -A
git commit -m "$(cat <<'EOF'
claude - <describe the change you just made>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

### Step 4 — Verify

```bash
git status --porcelain   # must be empty
git log --oneline -3     # confirm the commit(s) landed with the right subjects
```

If `git status --porcelain` is still non-empty, something was missed (often an untracked
file excluded by `.gitignore`, or a failed pre-commit hook). Resolve it before reporting
the change as done.

## Commit message rules

- Subject line is **always** `claude - <code-change>`.
  - lowercase `claude`, then space-hyphen-space, then the description.
  - `<code-change>` is a concise, specific description of what changed, in the
    imperative or descriptive present — e.g. `claude - add ffmpeg pipe video writer`,
    not `claude - changes` or `claude - updates to files`.
- Keep the subject under ~72 characters.
- Add a body only when the *why* is not obvious from the subject.
- Always end the message with the trailer:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## Hard rules

- Never use `git commit --amend` on a commit that is already pushed.
- Never use `git push --force` against a shared branch.
- Never commit generated data. `data/`, `runs/`, `*.mp4`, `*.jpg` and friends are in
  `.gitignore` — if a generated artifact shows up in `git status`, fix `.gitignore`
  rather than committing it.
- Never skip Step 1. A dirty tree at the start means the previous change was not
  committed, and silently folding it into the next commit destroys that history.
- If a commit fails because of an empty identity or a hook, report the error — do not
  work around it with `--no-verify` unless explicitly asked.
