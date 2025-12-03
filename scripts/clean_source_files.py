#!/usr/bin/env python3
"""
Delete the Python file that matches the module folder name.

Example:
    modules/01_tensor/tensor.py → deleted
    modules/19_benchmarking/benchmarking.py → deleted

This script NEVER deletes anything else.

Usage:
    python scripts/clean_source_files.py
    python scripts/clean_source_files.py 19_benchmarking
    python scripts/clean_source_files.py --dry-run
"""

import sys
from pathlib import Path


def module_expected_file(module_dir: Path) -> Path:
    """
    Given a folder like '01_tensor', return the expected file path:
    'tensor.py'
    """
    parts = module_dir.name.split("_", 1)
    name_part = parts[1] if len(parts) > 1 else parts[0]
    return module_dir / f"{name_part}.py"


def find_target_files(modules_dir: Path, module_name: str = None):
    """
    Return the list of files to delete, based on the matching rule.
    """
    targets = []

    if module_name:
        module_dir = modules_dir / module_name
        if not module_dir.exists() or not module_dir.is_dir():
            print(f"❌ Module not found: {module_name}")
            return []

        expected = module_expected_file(module_dir)
        if expected.exists():
            targets.append(expected)
        else:
            print(f"ℹ️ No matching file in module: {module_name}")
        return targets

    # Process all modules
    for module_dir in sorted(modules_dir.iterdir()):
        if not module_dir.is_dir() or module_dir.name.startswith("_"):
            continue

        expected = module_expected_file(module_dir)
        if expected.exists():
            targets.append(expected)

    return targets


def delete_files(files, dry_run=False):
    deleted = 0
    for f in files:
        relative = f.relative_to(f.parent.parent)

        if dry_run:
            print(f"🔍 Would delete: {relative}")
        else:
            try:
                f.unlink()
                print(f"🗑️  Deleted: {relative}")
                deleted += 1
            except Exception as e:
                print(f"❌ Failed to delete {relative}: {e}")

    return deleted


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    modules_dir = project_root / "modules"

    if not modules_dir.exists():
        print(f"❌ Modules directory not found: {modules_dir}")
        sys.exit(1)

    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    module_name = None

    # First non-flag argument is treated as module name
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            module_name = arg

    print("╭" + "─" * 78 + "╮")
    print("│" + " " * 78 + "│")
    print("│" + "  🧹 Module File Cleaner".center(78) + "│")
    print("│" + " " * 78 + "│")
    print("╰" + "─" * 78 + "╯")
    print()

    if dry_run:
        print("🔍 DRY RUN MODE — No files will be deleted\n")

    if module_name:
        print(f"📁 Target module: {module_name}\n")
    else:
        print("📁 Target: All modules\n")

    files = find_target_files(modules_dir, module_name)

    if not files:
        print("✅ No matching files to delete")
        return

    print(f"📦 Found {len(files)} file(s) to delete\n")

    deleted = delete_files(files, dry_run)

    print("\n" + "=" * 80)
    if dry_run:
        print(f"📊 Would delete {len(files)} file(s)")
    else:
        print(f"📊 Deleted {deleted} file(s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
