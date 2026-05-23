---
name: push-curriculum
description: Commit and push the current state of the image-gen-curriculum repo to GitHub using the project's git conventions. Use when the user wants to publish the latest changes.
---

# push-curriculum — commit + push helper

When invoked, stage, commit, and push current changes in this repo to GitHub. Do not invent or expand scope — push what's there.

## Repository facts

- **Remote:** `git@github.com:ms802x/image-gen-curriculum.git` (already configured as `origin`)
- **Branch:** `main`
- **Git user:** `ms802x` / `aalhejab.ee@gmail.com` (set globally)
- **SSH:** `core.sshCommand = ssh` is set globally to bypass the Coder/VSCode wrapper. No `GIT_SSH_COMMAND=ssh` prefix needed.

## What must not be committed

- `data/` — datasets are too large; regenerate via `shared/fetch_dataset.py`
- `.claude/*` except `.claude/skills/` (already covered by `.gitignore`)
- `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`
- Editor/OS junk (`.DS_Store`, `.vscode/`, `.idea/`, `*.swp`)

The repo's `.gitignore` enforces this. **Do not edit `.gitignore` to widen the ignore list without asking.**

## Procedure

Run these in parallel for fast inspection:
```bash
git status
git diff --stat HEAD
git log --oneline -5
```

Then:

1. **Inspect the diff carefully.** Look for accidentally-staged secrets, large files, dataset blobs. If `git status` shows anything under `data/` or otherwise suspicious, stop and confirm with the user.

2. **Stage specific files**, not `git add -A`. Pick the files relevant to the change. Avoid sweeping in editor temp files.

3. **Write a commit message** in this style:
    - Imperative title under ~70 chars (`Add Day 2 DDPM notebook`, `Fix v2 loss weighting`)
    - Optional body explaining the *why* in 1–4 short bullets

4. **Commit** with the HEREDOC pattern for proper formatting:
   ```bash
   git commit -m "$(cat <<'EOF'
   {title}

   {optional body}
   EOF
   )"
   ```

5. **Push:**
   ```bash
   git push
   ```

6. **Report back:** print the commit SHA + the GitHub URL of the commit (`https://github.com/ms802x/image-gen-curriculum/commit/{sha}`).

## Do not

- Force-push to `main` (without explicit user request)
- Amend an existing commit unless the user asks (`--amend` on already-pushed commits requires a force-push to land)
- Skip git hooks (`--no-verify`)
- Commit a notebook that hasn't been executed end-to-end (outputs should be embedded for the shared, executed-notebook convention this project follows)
- Add the dataset folder by mistake
- Use `git add .` or `git add -A` in directories that might contain untracked junk

## If something looks off

- Untracked notebook output checkpoints? They should already be ignored by `.gitignore` — if they appear, something is wrong with the ignore patterns. Confirm with the user before adjusting.
- Large file warning at push time (>50 MB)? Stop. The repo policy is no dataset blobs. The 10 MB executed notebooks are intentional; anything substantially bigger probably shouldn't be there.
- SSH error "Repository not found" right after first-time setup? Wait a few seconds for GitHub propagation; retry. If still failing, verify `ssh -T git@github.com` returns `Hi ms802x!` — if not, the key may not be added to GitHub yet.
