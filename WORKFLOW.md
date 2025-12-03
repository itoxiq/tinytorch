# TinyTorch Development Workflow

## Source of Truth: Jupyter Notebooks (.ipynb)

The TinyTorch project uses **Jupyter notebooks as the source of truth** for all module development. Python files (`.py`) are generated from notebooks automatically.

## Workflow Overview

```
┌─────────────────┐
│  Start Work     │
│  tito module    │──┐
│  start 01       │  │
└─────────────────┘  │
                     │
                     ▼
           ┌─────────────────────┐
           │  Edit Notebook      │
           │  tensor.ipynb       │
           │  (in Jupyter Lab)   │
           └─────────────────────┘
                     │
                     ▼
           ┌─────────────────────┐
           │  Complete Module    │
           │  tito module        │
           │  complete 01        │
           └─────────────────────┘
                     │
                     ▼
           ┌─────────────────────┐
           │  Auto-converts:     │
           │  .ipynb → .py       │
           │  Runs tests         │
           │  Exports to package │
           └─────────────────────┘
```

## Commands

### 1. Start Working on a Module

```bash
tito module start 01
```

**What it does:**
- Opens Jupyter Lab in the module directory
- Notebook (`tensor.ipynb`) is already there - ready to edit
- **Does NOT generate notebook from .py** (notebooks are source of truth!)

### 2. Edit the Notebook

- Work in Jupyter Lab
- Edit the `.ipynb` file directly
- Save your changes (Ctrl+S)
- Exit Jupyter Lab (Ctrl+C in terminal) when done

### 3. Complete the Module

```bash
tito module complete 01
```

**What it does:**
1. **Converts** `.ipynb` → `.py` using jupytext
2. **Runs tests** from the generated `.py` file
3. **Exports** to the TinyTorch package (if tests pass)
4. **Updates** progress tracking

### Flags

```bash
# Skip export (just convert and test)
tito module complete 01 --skip-export

# Skip tests (just convert and export)
tito module complete 01 --skip-tests
```

## Batch Processing

Process all modules at once:

```bash
# Process all modules (1-20)
python scripts/process_all_modules.py

# Process specific range
python scripts/process_all_modules.py 5 10
```

**What the script does:**
- For each module, runs `tito module complete XX`
- Converts notebooks to Python
- Runs all tests
- Shows summary of successes/failures
- Logs saved to `/tmp/module_XX_complete.log`

## File Structure

Each module has:

```
modules/01_tensor/
├── tensor.ipynb        ← SOURCE OF TRUTH (edit this!)
├── tensor.py           ← Generated from .ipynb (don't edit directly)
├── tensor_solution.py  ← Reference solution
└── README.md           ← Module documentation
```

## Key Points

✅ **DO:**
- Edit `.ipynb` notebooks in Jupyter Lab
- Use `tito module start XX` to open Jupyter Lab
- Use `tito module complete XX` to convert, test, and export
- Commit both `.ipynb` and `.py` files to git

❌ **DON'T:**
- Edit `.py` files directly (they'll be overwritten!)
- Manually run jupytext conversion
- Delete the generated `.py` files

## Why Notebooks as Source?

1. **Interactive Development**: Jupyter provides rich output, visualization, and debugging
2. **Single Source**: No confusion about which file is correct
3. **Automatic Sync**: `.py` files always match notebooks
4. **Better Learning**: Students can see outputs and experiment interactively

## Migration Note

Previously, the workflow was:
- `.py` files were source → notebooks generated for viewing

Now:
- `.ipynb` notebooks are source → `.py` files generated for testing/export

All existing `.ipynb` files should already exist in the repository!
