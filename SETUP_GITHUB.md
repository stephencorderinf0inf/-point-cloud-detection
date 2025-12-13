# GitHub Setup Guide - Point Cloud Detection Scanner

Step-by-step instructions to create your private GitHub repository with manual push control.

---

## Prerequisites

1. **GitHub Account** - [Sign up free](https://github.com/join)
   - Private repos are FREE (no subscription needed)
   
2. **Git Installed** - Check by running:
   ```bash
   git --version
   ```
   If not installed: Download from [git-scm.com](https://git-scm.com/)

---

## Step 1: Initialize Local Git Repository

Open PowerShell/Terminal in the `point_cloud_detection` folder:

```bash
cd "D:\Users\Planet UI\point_cloud_detection"
```

Initialize git:
```bash
git init
```

You should see: `Initialized empty Git repository in D:/Users/Planet UI/point_cloud_detection/.git/`

---

## Step 2: Configure Git (First Time Only)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Replace with your actual GitHub username/email.

---

## Step 3: Stage Files for Commit

```bash
# Add all files (respects .gitignore)
git add .

# Check what will be committed
git status
```

You should see files in green like:
```
        modified:   README.md
        new file:   requirements.txt
        new file:   .gitignore
        new file:   scanning/laser_3d_scanner_advanced.py
        ...
```

### Files EXCLUDED (by .gitignore):
- `__pycache__/` folders
- `.ply`, `.obj`, `.npz` scan files
- Camera calibration files (user-specific)
- Test output files

---

## Step 4: Create First Commit

```bash
git commit -m "Initial commit: Point cloud scanner with 3D viewer"
```

You should see:
```
[master (root-commit) abc1234] Initial commit: Point cloud scanner with 3D viewer
 52 files changed, 12847 insertions(+)
 create mode 100644 README.md
 create mode 100644 requirements.txt
 ...
```

---

## Step 5: Create GitHub Repository

### Option A: Via GitHub Website (Recommended)

1. Go to [github.com/new](https://github.com/new)
2. Fill in:
   - **Repository name**: `point-cloud-detection`
   - **Description**: `Advanced 3D scanner with laser triangulation & AI depth`
   - **Visibility**: ✅ **Private** (FREE - no subscription needed)
   - **Initialize**: ❌ Leave ALL checkboxes UNCHECKED
3. Click **Create repository**

### Option B: Via GitHub CLI (Advanced)

```bash
gh repo create point-cloud-detection --private --source=. --remote=origin --description="Advanced 3D scanner with laser triangulation & AI depth"
```

---

## Step 6: Link Local Repo to GitHub

GitHub will show you commands - use these:

```bash
# Add GitHub as remote
git remote add origin https://github.com/YOUR_USERNAME/point-cloud-detection.git

# Rename branch to 'main' (GitHub standard)
git branch -M main

# STOP HERE - Don't push yet!
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### Verify remote:
```bash
git remote -v
```

Should show:
```
origin  https://github.com/YOUR_USERNAME/point-cloud-detection.git (fetch)
origin  https://github.com/YOUR_USERNAME/point-cloud-detection.git (push)
```

---

## Step 7: Manual Review Before Push ⚠️

**THIS IS THE CONTROL POINT - Review before pushing:**

### Check what will be pushed:
```bash
# See all commits
git log --oneline

# See file changes
git diff --stat origin/main..HEAD
```

### If everything looks good:
```bash
git push -u origin main
```

### If you want to review more:
```bash
# See full diff
git diff origin/main..HEAD

# See specific file
git show HEAD:scanning/laser_3d_scanner_advanced.py
```

---

## Step 8: Verify Upload

1. Go to `https://github.com/YOUR_USERNAME/point-cloud-detection`
2. You should see:
   - ✅ `README.md` displayed on homepage
   - ✅ `scanning/` folder with scanner files
   - ✅ `calibration/` folder
   - ✅ `requirements.txt`
   - ✅ `SETUP_GITHUB.md` (this file)
   - ❌ No `.ply`, `.obj`, `.npz` files (excluded by `.gitignore`)
   - ❌ No `__pycache__/` folders

---

## Daily Workflow: Making Changes

### 1. Make code changes
Edit files in VS Code as normal.

### 2. Check what changed
```bash
git status
```

### 3. Review specific changes
```bash
# See all changes
git diff

# See changes in specific file
git diff scanning/laser_3d_scanner_advanced.py
```

### 4. Stage changes
```bash
# Stage specific file
git add scanning/laser_3d_scanner_advanced.py

# Or stage all changes
git add .
```

### 5. Commit locally
```bash
git commit -m "Fix: 3D viewer window positioning bug"
```

### 6. REVIEW BEFORE PUSH ⚠️
```bash
# See what will be pushed
git log origin/main..HEAD

# See file changes
git diff origin/main..HEAD
```

### 7. Push when ready
```bash
git push
```

---

## Common Commands Reference

### Status & History
```bash
git status              # See current changes
git log                 # See commit history
git log --oneline       # Compact history
git diff                # See unstaged changes
git diff --staged       # See staged changes
```

### Staging & Committing
```bash
git add <file>          # Stage specific file
git add .               # Stage all changes
git commit -m "msg"     # Commit staged changes
git commit -am "msg"    # Stage + commit (tracked files only)
```

### Remote Operations
```bash
git push                # Push commits to GitHub
git pull                # Get updates from GitHub
git fetch               # Check for updates (no merge)
```

### Undo Changes
```bash
git restore <file>      # Discard unstaged changes
git reset HEAD <file>   # Unstage file
git reset --soft HEAD~1 # Undo last commit (keep changes)
git reset --hard HEAD~1 # Undo last commit (DELETE changes) ⚠️
```

---

## Best Practices

### 1. **Commit Often**
```bash
# Good: Small, focused commits
git commit -m "Add 3D viewer hotkey 'O'"
git commit -m "Fix depth mode lazy loading"
git commit -m "Center window on startup"

# Bad: Large, vague commits
git commit -m "Fixed stuff"
git commit -m "WIP"
```

### 2. **Review Before Push**
Always run `git diff origin/main..HEAD` before `git push`

### 3. **Use Descriptive Messages**
```bash
# Good commit messages:
git commit -m "Fix: 3D viewer shrinking video window"
git commit -m "Add: Interactive Open3D point cloud visualization"
git commit -m "Refactor: Rename get_depth_estimator to load_depth_model"

# Bad commit messages:
git commit -m "fix bug"
git commit -m "update"
git commit -m "asdfasdf"
```

### 4. **Keep `.gitignore` Updated**
If you create new file types that shouldn't be in git:
```bash
# Edit .gitignore, then:
git add .gitignore
git commit -m "Update gitignore: exclude AI model cache"
```

---

## Troubleshooting

### "Permission denied (publickey)"
```bash
# Use HTTPS instead of SSH
git remote set-url origin https://github.com/YOUR_USERNAME/point-cloud-detection.git
```

### "Large files detected"
```bash
# Remove file from staging
git rm --cached scanning/data/large_file.ply

# Add to .gitignore
echo "*.ply" >> .gitignore
git add .gitignore
git commit -m "Exclude PLY files from git"
```

### "Merge conflict"
```bash
# Pull first, resolve conflicts, then push
git pull
# Edit conflicted files (marked with <<<<<<)
git add .
git commit -m "Resolve merge conflict"
git push
```

### "Accidentally committed sensitive data"
```bash
# Remove file from all history (dangerous!)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/file" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (overwrites GitHub history)
git push --force
```

---

## GitHub Web Features

### Useful Features:
1. **Releases** - Tag stable versions (v1.0, v2.0)
2. **Issues** - Track bugs/features
3. **Wiki** - Extended documentation
4. **Actions** - Auto-testing (advanced)
5. **Projects** - Kanban board for tasks

### Creating a Release:
1. Go to repository page
2. Click "Releases" → "Create a new release"
3. Tag: `v2.0`
4. Title: `Version 2.0 - 3D Viewer + Lazy Loading`
5. Description: List changes
6. Upload `.zip` of project (optional)
7. Publish

---

## Privacy & Security

### Your Private Repo is Safe:
- ✅ Only YOU can see it
- ✅ Only you can clone it
- ✅ Only you can invite collaborators
- ❌ NOT visible in search engines
- ❌ NOT in GitHub explore/trending

### Make it Public Later:
Settings → Danger Zone → Change visibility → Make public

### Invite Collaborators:
Settings → Collaborators → Add people → Enter username

---

## Next Steps

1. ✅ **Initialize git** - `git init`
2. ✅ **Create GitHub repo** - [github.com/new](https://github.com/new)
3. ✅ **Link remote** - `git remote add origin ...`
4. ✅ **Review changes** - `git diff origin/main..HEAD`
5. ⬜ **First push** - `git push -u origin main`
6. ⬜ **Make changes** - Edit code, commit, review, push

---

**You're all set! Happy coding! 🎯✨**

For help: [GitHub Docs](https://docs.github.com/) | [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
