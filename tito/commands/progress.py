"""
Progress command for TinyTorch CLI: manage module progress tracking.
"""

import json
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.table import Table

from .base import BaseCommand


class ProgressCommand(BaseCommand):
    """Command to manage module progress tracking."""

    @property
    def name(self) -> str:
        return "progress"

    @property
    def description(self) -> str:
        return "Manage module progress tracking"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add progress command arguments."""
        subparsers = parser.add_subparsers(
            dest='progress_command',
            help='Progress tracking operations',
            metavar='SUBCOMMAND'
        )

        # RESET command - reset progress for a module
        reset_parser = subparsers.add_parser(
            'reset',
            help='Reset progress tracking for a module (without changing files)'
        )
        reset_parser.add_argument(
            'module_number',
            nargs='?',
            help='Module number to reset progress for (01-20), defaults to last worked'
        )
        reset_parser.add_argument(
            '--all',
            action='store_true',
            help='Reset all progress tracking'
        )
        reset_parser.add_argument(
            '--force',
            action='store_true',
            help='Skip confirmation prompt'
        )

        # SHOW command - display current progress
        show_parser = subparsers.add_parser(
            'show',
            help='Show current progress tracking'
        )

    def get_progress_data(self) -> dict:
        """Get current progress data."""
        progress_file = self.config.project_root / "progress.json"
        
        try:
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        return {
            'started_modules': [],
            'completed_modules': [],
            'last_worked': None,
            'last_completed': None,
            'last_updated': None
        }

    def save_progress_data(self, progress: dict) -> None:
        """Save progress data."""
        progress_file = self.config.project_root / "progress.json"
        
        try:
            progress['last_updated'] = datetime.now().isoformat()
            
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            self.console.print(f"[yellow]⚠️  Could not save progress: {e}[/yellow]")

    def get_module_mapping(self) -> dict:
        """Get mapping from numbers to module names."""
        return {
            "01": "01_tensor",
            "02": "02_activations",
            "03": "03_layers",
            "04": "04_losses",
            "05": "05_autograd",
            "06": "06_optimizers",
            "07": "07_training",
            "08": "08_dataloader",
            "09": "09_spatial",
            "10": "10_tokenization",
            "11": "11_embeddings",
            "12": "12_attention",
            "13": "13_transformers",
            "14": "14_profiling",
            "15": "15_quantization",
            "16": "16_compression",
            "17": "17_memoization",
            "18": "18_acceleration",
            "19": "19_benchmarking",
            "20": "20_capstone",
        }

    def normalize_module_number(self, module_input: str) -> str:
        """Normalize module input to 2-digit format."""
        if module_input.isdigit():
            return f"{int(module_input):02d}"
        return module_input

    def reset_module_progress(self, module_number: str, force: bool = False) -> int:
        """Reset progress tracking for a specific module."""
        console = self.console
        progress = self.get_progress_data()
        module_mapping = self.get_module_mapping()
        
        if module_number not in module_mapping:
            console.print(f"[red]Invalid module number: {module_number}[/red]")
            return 1

        module_name = module_mapping[module_number]
        
        # Check if module has any progress
        is_started = module_number in progress.get('started_modules', [])
        is_completed = module_number in progress.get('completed_modules', [])
        
        if not is_started and not is_completed:
            console.print(Panel(
                f"[yellow]Module {module_number} ({module_name}) has no recorded progress.[/yellow]",
                title="No Progress to Reset",
                border_style="yellow"
            ))
            return 0

        # Ask for confirmation unless --force
        if not force:
            console.print(f"\n[bold yellow]Reset progress for module {module_number} ({module_name})?[/bold yellow]\n")
            console.print("This will:")
            if is_started:
                console.print("  • Remove from started modules list")
            if is_completed:
                console.print("  • Remove from completed modules list")
            console.print("\n[dim]Note: This will NOT change any files in modules/ or tinytorch/[/dim]\n")

            try:
                response = input("Continue? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    console.print(Panel(
                        "[cyan]Reset cancelled.[/cyan]",
                        title="Cancelled",
                        border_style="cyan"
                    ))
                    return 0
            except KeyboardInterrupt:
                console.print(Panel(
                    "\n[cyan]Reset cancelled.[/cyan]",
                    title="Cancelled",
                    border_style="cyan"
                ))
                return 0

        # Reset progress
        if 'started_modules' in progress and module_number in progress['started_modules']:
            progress['started_modules'].remove(module_number)
        
        if 'completed_modules' in progress and module_number in progress['completed_modules']:
            progress['completed_modules'].remove(module_number)
        
        # Update last_worked if it was this module
        if progress.get('last_worked') == module_number:
            # Find the most recent module from started or completed
            all_modules = list(set(progress.get('started_modules', []) + progress.get('completed_modules', [])))
            if all_modules:
                progress['last_worked'] = max(all_modules)
            else:
                progress['last_worked'] = None
        
        # Update last_completed if it was this module
        if progress.get('last_completed') == module_number:
            completed = progress.get('completed_modules', [])
            if completed:
                progress['last_completed'] = max(completed)
            else:
                progress['last_completed'] = None

        self.save_progress_data(progress)

        console.print(Panel(
            f"[green]✅ Progress reset for module {module_number} ({module_name})[/green]\n\n"
            f"You can now:\n"
            f"  • [cyan]tito module start {module_number}[/cyan] - Start the module fresh\n"
            f"  • [cyan]tito module complete {module_number}[/cyan] - Re-complete the module",
            title="Progress Reset Complete",
            border_style="green"
        ))

        return 0

    def reset_all_progress(self, force: bool = False) -> int:
        """Reset all progress tracking."""
        console = self.console

        # Ask for confirmation unless --force
        if not force:
            console.print("\n[bold red]⚠️  Warning: This will reset ALL module progress tracking[/bold red]\n")
            console.print("[yellow]This will clear:[/yellow]")
            console.print("  • All started modules")
            console.print("  • All completed modules")
            console.print("  • Last worked module")
            console.print("  • Last completed module\n")
            console.print("[dim]Note: This will NOT change any files in modules/ or tinytorch/[/dim]\n")

            try:
                response = input("Continue? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    console.print(Panel(
                        "[cyan]Reset cancelled.[/cyan]",
                        title="Cancelled",
                        border_style="cyan"
                    ))
                    return 0
            except KeyboardInterrupt:
                console.print(Panel(
                    "\n[cyan]Reset cancelled.[/cyan]",
                    title="Cancelled",
                    border_style="cyan"
                ))
                return 0

        # Reset all progress
        progress = {
            'started_modules': [],
            'completed_modules': [],
            'last_worked': None,
            'last_completed': None
        }
        
        self.save_progress_data(progress)

        console.print(Panel(
            "[green]✅ All progress tracking reset![/green]\n\n"
            "You're ready to start fresh.\n"
            "Run: [cyan]tito module start 01[/cyan]",
            title="Progress Reset Complete",
            border_style="green"
        ))

        return 0

    def show_progress(self) -> int:
        """Show current progress tracking."""
        console = self.console
        progress = self.get_progress_data()
        module_mapping = self.get_module_mapping()

        # Create progress table
        table = Table(title="Module Progress Tracking", show_header=True)
        table.add_column("Module", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Started", style="yellow")
        table.add_column("Completed", style="green")

        started = progress.get('started_modules', [])
        completed = progress.get('completed_modules', [])

        for num, name in sorted(module_mapping.items()):
            is_started = num in started
            is_completed = num in completed
            table.add_row(
                num,
                name,
                "✓" if is_started else "",
                "✓" if is_completed else ""
            )

        console.print(table)
        console.print(f"\n[cyan]Last worked:[/cyan] {progress.get('last_worked', 'None')}")
        console.print(f"[green]Last completed:[/green] {progress.get('last_completed', 'None')}")
        console.print(f"[dim]Last updated:[/dim] {progress.get('last_updated', 'Never')}")

        return 0

    def run(self, args: Namespace) -> int:
        """Execute the progress command."""
        console = self.console

        if not hasattr(args, 'progress_command') or not args.progress_command:
            console.print(Panel(
                "[bold cyan]Progress Commands[/bold cyan]\n\n"
                "Available subcommands:\n"
                "  • [bold]reset[/bold]  - Reset progress tracking for a module\n"
                "  • [bold]show[/bold]   - Display current progress tracking\n\n"
                "[dim]Examples:[/dim]\n"
                "[dim]  tito progress reset 01       # Reset progress for module 01[/dim]\n"
                "[dim]  tito progress reset          # Reset progress for last worked module[/dim]\n"
                "[dim]  tito progress reset --all    # Reset all progress[/dim]\n"
                "[dim]  tito progress show           # Show current progress[/dim]",
                title="Progress Command Group",
                border_style="bright_yellow"
            ))
            return 0

        # Execute the appropriate subcommand
        if args.progress_command == 'reset':
            if args.all:
                return self.reset_all_progress(args.force)
            else:
                # Get module number
                module_number = args.module_number
                if not module_number:
                    # Default to last worked module
                    progress = self.get_progress_data()
                    module_number = progress.get('last_worked')
                    if not module_number:
                        console.print("[red]No module specified and no last worked module found.[/red]")
                        console.print("[dim]Usage: tito progress reset <module_number>[/dim]")
                        return 1
                    console.print(f"[dim]Using last worked module: {module_number}[/dim]\n")
                else:
                    module_number = self.normalize_module_number(module_number)
                
                return self.reset_module_progress(module_number, args.force)
        
        elif args.progress_command == 'show':
            return self.show_progress()
        
        else:
            console.print(Panel(
                f"[red]Unknown progress subcommand: {args.progress_command}[/red]",
                title="Error",
                border_style="red"
            ))
            return 1
