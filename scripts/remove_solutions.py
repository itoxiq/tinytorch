#!/usr/bin/env python3
"""
Script to remove solution code blocks from Jupyter notebooks.
Removes content between # BEGIN SOLUTION and # END SOLUTION markers.
Excludes files ending with _solution.ipynb
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Set


def process_notebook(notebook_path: Path) -> bool:
    """
    Process a single notebook to remove solution blocks.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        True if the notebook was modified, False otherwise
    """
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    modified = False

    # Process each cell
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, str):
                source = source.split('\n')
                source = [line + '\n' for line in source[:-1]] + [source[-1]]

            # Check if this cell has solution markers
            has_solution = any('BEGIN SOLUTION' in line and '#' in line for line in source)

            if has_solution:
                result = []
                in_solution = False

                for line in source:
                    # Check for BEGIN SOLUTION marker
                    if 'BEGIN SOLUTION' in line and '#' in line:
                        in_solution = True
                        result.append(line)  # Keep the marker
                        continue

                    # Check for END SOLUTION marker
                    if 'END SOLUTION' in line and '#' in line:
                        in_solution = False
                        result.append(line)  # Keep the marker
                        continue

                    # Only add lines that are not in a solution block
                    if not in_solution:
                        result.append(line)

                if result != source:
                    cell['source'] = result
                    modified = True

    # Write back if modified
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)

    return modified


def parse_range(range_str: str) -> List[int]:
    """
    Parse a range string like '01-05' or '1-5' into a list of numbers.

    Args:
        range_str: Range string (e.g., '01-05' or '1-5')

    Returns:
        List of numbers in the range
    """
    parts = range_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid range format: {range_str}. Use format like '01-05'")

    start = int(parts[0])
    end = int(parts[1])

    if start > end:
        raise ValueError(f"Invalid range: {range_str}. Start must be <= end")

    return list(range(start, end + 1))


def get_folders_to_process(folders: List[str], folder_range: str, base_path: Path) -> Set[Path]:
    """
    Get the set of folders to process based on input arguments.

    Args:
        folders: List of individual folder numbers (e.g., ['01', '03', '07'])
        folder_range: Range specification (e.g., '01-05')
        base_path: Base path to search from

    Returns:
        Set of folder paths to process
    """
    folder_numbers = set()

    # Process range if specified
    if folder_range:
        folder_numbers.update(parse_range(folder_range))

    # Process individual folders
    if folders:
        for folder in folders:
            folder_numbers.add(int(folder))

    # Find actual folder paths
    folders_to_process = set()
    for num in sorted(folder_numbers):
        folder_name = f"{num:02d}"
        # Find folders that start with this number
        for path in base_path.iterdir():
            if path.is_dir() and path.name.startswith(folder_name):
                folders_to_process.add(path)
                break

    return folders_to_process


def find_notebooks(folders: Set[Path]) -> List[Path]:
    """
    Find all notebook files in the specified folders.

    Args:
        folders: Set of folder paths to search

    Returns:
        List of notebook file paths
    """
    notebooks = []

    for folder in folders:
        # Find all .ipynb files in this folder, excluding hidden and checkpoint directories
        for notebook_path in folder.rglob("*.ipynb"):
            # Skip hidden directories and .ipynb_checkpoints
            if any(part.startswith('.') for part in notebook_path.parts):
                continue

            # Skip _solution.ipynb files
            if notebook_path.name.endswith('_solution.ipynb'):
                continue

            notebooks.append(notebook_path)

    return sorted(notebooks)


def main():
    parser = argparse.ArgumentParser(
        description='Remove solution blocks from Jupyter notebooks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                    # Process all module folders
  %(prog)s 01                       # Process only folder 01
  %(prog)s 01 03 07                 # Process folders 01, 03, and 07
  %(prog)s --range 01-05            # Process folders 01 through 05
  %(prog)s --range 01-05 07 09      # Process folders 01-05, plus 07 and 09
        """
    )

    parser.add_argument(
        'folders',
        nargs='*',
        help='Individual folder number(s) to process (e.g., 01 03 07). Omit to use --all or --range.'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all module folders'
    )

    parser.add_argument(
        '--range',
        type=str,
        metavar='START-END',
        help='Range of folders to process (e.g., 01-05). Can be combined with individual folders.'
    )

    parser.add_argument(
        '--base-path',
        type=Path,
        default=Path('.'),
        help='Base path to search for folders (default: current directory)'
    )

    args = parser.parse_args()

    # Handle --all flag
    if args.all:
        all_folders = set()
        for path in args.base_path.iterdir():
            if path.is_dir() and not path.name.startswith('.'):
                if path.name[:2].isdigit():
                    all_folders.add(path)
        folders_to_process = all_folders
    elif args.folders or args.range:
        # Get folders to process from individual folders and/or range
        folders_to_process = get_folders_to_process(args.folders, args.range, args.base_path)
    else:
        parser.error("Must specify either --all, --range, or folder numbers")

    if not folders_to_process:
        print("No folders found to process.")
        sys.exit(1)

    print("Removing solution blocks from .ipynb files...")
    print("=" * 50)
    print(f"Processing folders: {', '.join(sorted(f.name for f in folders_to_process))}")
    print("=" * 50)

    # Find all notebooks in the selected folders
    notebooks = find_notebooks(folders_to_process)

    if not notebooks:
        print("No notebooks found in the specified folders.")
        sys.exit(0)

    # Process each notebook
    modified_count = 0
    total_count = len(notebooks)

    for notebook_path in notebooks:
        print(f"Processing: {notebook_path}")

        was_modified = process_notebook(notebook_path)

        if was_modified:
            print(f"  ✓ Modified")
            modified_count += 1
        else:
            print(f"  - No changes needed")

    print("=" * 50)
    print("Summary:")
    print(f"  Total notebooks processed: {total_count}")
    print(f"  Notebooks modified: {modified_count}")
    print(f"  Notebooks unchanged: {total_count - modified_count}")


if __name__ == '__main__':
    main()
