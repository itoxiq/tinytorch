#!/usr/bin/env python3
"""
Script to process all TinyTorch modules using tito commands.
Notebooks (.ipynb) are the source of truth - they already exist.
Usage: python scripts/process_all_modules.py [start_module] [end_module]
"""

import subprocess
import sys
from pathlib import Path
from typing import Tuple

# Colors
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_header(text: str, color: str = Colors.BLUE):
    """Print a formatted header."""
    print(f"{color}{'═' * 60}{Colors.NC}")
    print(f"{color}{text:^60}{Colors.NC}")
    print(f"{color}{'═' * 60}{Colors.NC}")
    print()


def complete_module(module_num: str) -> bool:
    """Complete module using tito module complete (converts .ipynb to .py and runs tests)."""
    print(f"{Colors.YELLOW}🔄 Running tito module complete {module_num}...{Colors.NC}")

    log_file = Path(f"/tmp/module_{module_num}_complete.log")

    try:
        result = subprocess.run(
            ["tito", "module", "complete", module_num, "--skip-export"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Write logs
        with open(log_file, 'w') as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)

        if result.returncode == 0:
            print(f"{Colors.GREEN}✅ Module completed successfully!{Colors.NC}")
            return True
        else:
            print(f"{Colors.RED}❌ Module completion failed!{Colors.NC}")
            print(f"{Colors.YELLOW}   View logs: {log_file}{Colors.NC}")

            # Show last 15 lines of error
            print(f"{Colors.RED}   Last errors:{Colors.NC}")
            error_lines = (result.stdout + "\n" + result.stderr).split('\n')
            for line in error_lines[-15:]:
                if line.strip():
                    print(f"   {line}")

            return False

    except subprocess.TimeoutExpired:
        print(f"{Colors.RED}❌ Module completion timed out after 5 minutes{Colors.NC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}❌ Error completing module: {e}{Colors.NC}")
        return False


def process_module(module_num: str) -> Tuple[str, str]:
    """
    Process a single module.
    Returns: (module_num, status) where status is 'success' or 'failed'
    """
    print(f"{Colors.BLUE}{'━' * 60}{Colors.NC}")
    print(f"{Colors.YELLOW}📦 Processing Module {module_num}{Colors.NC}")
    print(f"{Colors.BLUE}{'━' * 60}{Colors.NC}")

    # Complete module (converts .ipynb to .py and runs tests)
    success = complete_module(module_num)

    print()

    if success:
        return (module_num, 'success')
    else:
        return (module_num, 'failed')


def main():
    """Main entry point."""
    # Parse arguments
    start_module = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end_module = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    print_header(f"TinyTorch Module Processing Pipeline\nNotebooks (.ipynb) are source of truth\nProcessing modules {start_module} through {end_module}")

    # Track results
    successful_modules = []
    failed_modules = []

    # Process each module
    for i in range(start_module, end_module + 1):
        module_num = f"{i:02d}"
        _, status = process_module(module_num)

        if status == 'success':
            successful_modules.append(module_num)
        else:
            failed_modules.append(module_num)

    # Print summary
    print_header("Processing Complete - Summary")

    print(f"{Colors.GREEN}✅ Successful ({len(successful_modules)}): {', '.join(successful_modules) if successful_modules else 'None'}{Colors.NC}")
    print(f"{Colors.RED}❌ Failed ({len(failed_modules)}): {', '.join(failed_modules) if failed_modules else 'None'}{Colors.NC}")
    print()

    # Exit with error if any failed
    if failed_modules:
        print(f"{Colors.RED}Some modules failed. Check logs in /tmp/module_*_complete.log{Colors.NC}")
        sys.exit(1)
    else:
        print(f"{Colors.GREEN}🎉 All modules processed successfully!{Colors.NC}")
        sys.exit(0)


if __name__ == "__main__":
    main()
