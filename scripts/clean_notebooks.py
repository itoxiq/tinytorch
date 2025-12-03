#!/usr/bin/env python3
"""
Clean all Jupyter notebooks in the modules directory.

This script removes all .ipynb files from the modules/ directory. This is useful
when you want to regenerate notebooks from .py files or clean up the workspace.

Usage:
    python scripts/clean_notebooks.py [module_name]

    # Clean all notebooks
    python scripts/clean_notebooks.py

    # Clean specific module notebooks
    python scripts/clean_notebooks.py 19_benchmarking

    # Dry run (show what would be deleted without actually deleting)
    python scripts/clean_notebooks.py --dry-run
"""

import sys
from pathlib import Path


def find_notebooks(modules_dir: Path, module_name: str = None):
    """
    Find all Jupyter notebooks.

    Args:
        modules_dir: Path to modules directory
        module_name: Optional specific module to clean

    Returns:
        List of notebook paths
    """
    notebooks = []

    if module_name:
        # Clean specific module
        module_path = modules_dir / module_name
        if not module_path.exists():
            print(f"❌ Module not found: {module_name}")
            return []

        notebooks = list(module_path.glob("*.ipynb"))
    else:
        # Clean all modules
        for module_dir in sorted(modules_dir.iterdir()):
            if not module_dir.is_dir() or module_dir.name.startswith("_"):
                continue

            notebooks.extend(module_dir.glob("*.ipynb"))

    return notebooks


def clean_notebooks(notebooks: list, dry_run: bool = False):
    """
    Delete notebook files.

    Args:
        notebooks: List of notebook paths to delete
        dry_run: If True, only show what would be deleted

    Returns:
        Number of files deleted
    """
    if not notebooks:
        return 0

    deleted_count = 0

    for notebook in notebooks:
        if dry_run:
            print(f"🔍 Would delete: {notebook.relative_to(notebook.parent.parent)}")
        else:
            try:
                notebook.unlink()
                print(f"🗑️  Deleted: {notebook.relative_to(notebook.parent.parent)}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Failed to delete {notebook.name}: {e}")

    return deleted_count


def main():
    """Main script entry point."""
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    modules_dir = project_root / "modules"

    # Check if modules directory exists
    if not modules_dir.exists():
        print(f"❌ Modules directory not found: {modules_dir}")
        sys.exit(1)

    # Parse command line arguments
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    module_name = None

    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            module_name = arg
            break

    print("╭" + "─" * 78 + "╮")
    print("│" + " " * 78 + "│")
    print("│" + "  🧹 Notebook Cleaner".center(78) + "│")
    print("│" + " " * 78 + "│")
    print("╰" + "─" * 78 + "╯")
    print()

    if dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")
        print()

    if module_name:
        print(f"🎯 Cleaning module: {module_name}")
    else:
        print("🎯 Cleaning all modules")
    print()

    # Find notebooks
    notebooks = find_notebooks(modules_dir, module_name)

    if not notebooks:
        print("✅ No notebooks found to clean")
        return

    print(f"📦 Found {len(notebooks)} notebook(s) to delete")
    print()

    # Clean notebooks
    deleted_count = clean_notebooks(notebooks, dry_run)

    # Print summary
    print()
    print("=" * 80)
    if dry_run:
        print(f"📊 Would delete {len(notebooks)} notebook(s)")
        print()
        print("💡 Run without --dry-run to actually delete files")
    else:
        print(f"📊 Deleted {deleted_count} notebook(s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
