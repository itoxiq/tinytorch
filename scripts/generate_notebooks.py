#!/usr/bin/env python3
"""
Generate Jupyter notebooks from Python module files.

This script converts all .py files in the modules/ directory to .ipynb format
using jupytext. This is useful when modules are developed as .py files but need
to be exported as notebooks for the nbdev workflow.

Usage:
    python scripts/generate_notebooks.py [module_name]

    # Convert all modules
    python scripts/generate_notebooks.py

    # Convert specific module
    python scripts/generate_notebooks.py 19_benchmarking
"""

import subprocess
import sys
from pathlib import Path


def find_module_py_files(modules_dir: Path, module_name: str = None):
    """
    Find Python module files whose name matches their parent folder.
    E.g., folder '01_tensor' -> file 'tensor.py'.
    """
    module_files = []

    def expected_filename(folder: Path) -> str:
        # Remove leading numeric prefix (e.g. "01_")
        parts = folder.name.split("_", 1)
        name_part = parts[1] if len(parts) > 1 else folder.name
        return f"{name_part}.py"

    if module_name:
        module_path = modules_dir / module_name
        if not module_path.exists():
            print(f"❌ Module not found: {module_name}")
            return []

        expected = expected_filename(module_path)
        target = module_path / expected

        if target.exists():
            module_files.append((target, module_name))
        else:
            print(f"❌ Expected file not found: {expected}")
        return module_files

    # Convert all modules
    for module_dir in sorted(modules_dir.iterdir()):
        if not module_dir.is_dir() or module_dir.name.startswith("_"):
            continue

        expected = expected_filename(module_dir)
        target = module_dir / expected

        if target.exists():
            module_files.append((target, module_dir.name))

    return module_files


def convert_py_to_ipynb(py_file: Path, verbose: bool = True):
    """
    Convert a Python file to Jupyter notebook using jupytext.

    Args:
        py_file: Path to Python file
        verbose: Print conversion status

    Returns:
        True if successful, False otherwise
    """
    ipynb_file = py_file.with_suffix('.ipynb')

    try:
        # Check if notebook already exists and is newer than .py file
        if ipynb_file.exists():
            py_mtime = py_file.stat().st_mtime
            ipynb_mtime = ipynb_file.stat().st_mtime

            if ipynb_mtime > py_mtime:
                if verbose:
                    print(f"⏭️  Skipping {py_file.name} (notebook is up to date)")
                return True

        if verbose:
            print(f"🔄 Converting {py_file.name} → {ipynb_file.name}")

        # Run jupytext conversion
        result = subprocess.run(
            ['jupytext', '--to', 'ipynb', str(py_file)],
            capture_output=True,
            text=True,
            cwd=py_file.parent
        )

        if result.returncode != 0:
            print(f"❌ Failed to convert {py_file.name}")
            if verbose and result.stderr:
                print(f"   Error: {result.stderr}")
            return False

        if verbose:
            print(f"✅ Created {ipynb_file.name}")

        return True

    except FileNotFoundError:
        print("❌ jupytext not found. Install with: pip install jupytext")
        return False
    except Exception as e:
        print(f"❌ Error converting {py_file.name}: {e}")
        return False


def main():
    """Main script entry point."""
    # Get project root (script is in scripts/, project root is parent)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    modules_dir = project_root / "modules"

    # Check if modules directory exists
    if not modules_dir.exists():
        print(f"❌ Modules directory not found: {modules_dir}")
        sys.exit(1)

    # Parse command line arguments
    module_name = sys.argv[1] if len(sys.argv) > 1 else None

    print("╭" + "─" * 78 + "╮")
    print("│" + " " * 78 + "│")
    print("│" + "  📓 Jupyter Notebook Generator".center(78) + "│")
    print("│" + " " * 78 + "│")
    print("╰" + "─" * 78 + "╯")
    print()

    if module_name:
        print(f"🎯 Converting module: {module_name}")
    else:
        print("🎯 Converting all modules")
    print()

    # Find module files
    module_files = find_module_py_files(modules_dir, module_name)

    if not module_files:
        print("❌ No module files found")
        sys.exit(1)

    print(f"📦 Found {len(module_files)} file(s) to process")
    print()

    # Convert each file
    success_count = 0
    failed_count = 0
    skipped_count = 0

    for py_file, mod_name in module_files:
        # Check if notebook exists and is newer
        ipynb_file = py_file.with_suffix('.ipynb')
        if ipynb_file.exists():
            py_mtime = py_file.stat().st_mtime
            ipynb_mtime = ipynb_file.stat().st_mtime

            if ipynb_mtime > py_mtime:
                skipped_count += 1
                continue

        if convert_py_to_ipynb(py_file):
            success_count += 1
        else:
            failed_count += 1

    # Print summary
    print()
    print("=" * 80)
    print("📊 Conversion Summary:")
    print(f"   ✅ Successfully converted: {success_count}")
    if skipped_count > 0:
        print(f"   ⏭️  Skipped (up to date): {skipped_count}")
    if failed_count > 0:
        print(f"   ❌ Failed: {failed_count}")
    print("=" * 80)

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
