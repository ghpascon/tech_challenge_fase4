"""
poetry run python git_commit_all.py 
"""

import subprocess
from datetime import datetime
from pathlib import Path
import os
import webbrowser

username = "ghpascon"

def run_git_command(args, repo_path, check=True, capture_output=False):
    """Run git command and handle errors gracefully."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=repo_path,
            check=check,
            capture_output=capture_output,
            text=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        return e  # Return error object to continue

def ensure_git_repo(repo_path: Path):
    if not (repo_path / ".git").exists():
        print("No Git repository found. Initializing...")
        run_git_command(["init"], repo_path)
        print("Initialized empty Git repository.")

def get_remote_url(repo_path: Path):
    """Return the URL of origin remote if it exists, else None."""
    result = run_git_command(["remote", "get-url", "origin"], repo_path, check=False, capture_output=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return None

def ensure_remote(repo_path: Path, repo_name: str = None):
    """Set remote only if it doesn't exist."""
    remote_url = get_remote_url(repo_path)
    if remote_url:
        print(f"Using existing remote: {remote_url}")
        return remote_url

    if not repo_name:
        repo_name = input("Enter GitHub repository name: ").strip()
    remote_url = f"https://github.com/{username}/{repo_name}.git"

    print("Remote repository not found.")
    print("Opening GitHub to create a new repository...")
    webbrowser.open("https://github.com/new")
    input("Press Enter after creating the repository on GitHub...")

    # Add remote
    run_git_command(["remote", "add", "origin", remote_url], repo_path)
    print(f"Remote 'origin' set to {remote_url}")
    return remote_url

def safe_add(repo_path):
    """Add all files safely, skipping .git and .vscode folders silently."""
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.git') and not d.startswith('.vscode')]
        for file in files:
            file_path = Path(root) / file
            run_git_command(["add", str(file_path)], repo_path, check=False)

def git_commit_all():
    repo_path = Path.cwd()
    ensure_git_repo(repo_path)

    # Stage all files safely
    safe_add(repo_path)

    # Ask for commit title and description
    commit_title = input("Enter commit title (short): ").strip()
    if not commit_title:
        commit_title = f"Auto-commit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    commit_description = input("Enter commit description (optional): ").strip()
    commit_message = f"{commit_title}\n\n{commit_description}" if commit_description else commit_title

    # Commit changes if any
    status = run_git_command(["status", "--porcelain"], repo_path, capture_output=True)
    if status.stdout.strip():
        run_git_command(["commit", "-m", commit_message], repo_path, check=False)
        print(f"Committed changes:\n{commit_message}")
    else:
        print("Nothing to commit. Working tree clean.")

    # Ensure remote exists (only asks first time)
    remote_url = ensure_remote(repo_path)

    branch = "main"
    run_git_command(["branch", "-M", branch], repo_path, check=False)

    # Fetch remote safely
    run_git_command(["fetch", "origin"], repo_path, check=False)

    # Merge remote changes automatically without prompting
    run_git_command([
        "merge", f"origin/{branch}",
        "-X", "theirs",
        "--allow-unrelated-histories",
        "-m", "Merged remote changes, remote takes priority"
    ], repo_path, check=False)

    # First try a normal push
    push_result = run_git_command(["push", "-u", "origin", branch], repo_path, check=False)
    if isinstance(push_result, subprocess.CalledProcessError) or push_result.returncode != 0:
        print("Normal push failed. Retrying with force...")
        run_git_command(["push", "-u", "origin", branch, "--force"], repo_path, check=False)
        print(f"Force-pushed changes to remote branch '{branch}'.")
    else:
        print(f"Pushed changes to remote branch '{branch}' successfully.")

if __name__ == "__main__":
    git_commit_all()
