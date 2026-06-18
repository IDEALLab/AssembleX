# Python Ruff Conda Template

ETHZ IDEAL Lab Python project template with Ruff linting/formatting, MyPy type checking, and pre-commit hooks.

## Quick Setup

### Prerequisites
- [Miniforge](https://github.com/conda-forge/miniforge) (conda package manager)
- [VS Code](https://code.visualstudio.com/) with recommended extensions (prompted on first open)

### Setup (all platforms)

From the project root, run:
```bash
python bootstrap_env.py
```

This single script works on **Windows, macOS, and Linux**. It will:
1. Create the conda environment from `environment.yml`
2. Install dev tools (Ruff, MyPy, pytest, pre-commit)
3. Install pre-commit git hooks
4. Verify everything works

> **Tip:** If your environment gets into a bad state, recreate it from scratch:
> ```bash
> python bootstrap_env.py --recreate
> ```

### Manual Setup

If you prefer to set things up step by step:

```bash
conda env create -f environment.yml
conda activate <env-name>    # see environment.yml for the name
pip install .[dev]
pre-commit install
```

### VS Code Configuration

1. Open the project in VS Code — it will prompt you to install recommended extensions
2. Copy `.vscode/settings_template.json` to `.vscode/settings.json`

## Usage

```bash
conda activate <env-name>    # see environment.yml for the name
```

### Code Quality
```bash
ruff check .          # Lint
ruff check --fix .    # Lint + auto-fix
ruff format .         # Format
mypy .                # Type-check
```

### Testing
```bash
pytest                # Run tests
```

### Pre-commit Hooks
```bash
pre-commit run --all-files   # Run all hooks manually
```

Hooks run automatically on every `git commit` — Ruff will lint, fix, and format your code before it enters the repository.

## What's Included

- **Python 3.11** via conda-forge
- **Ruff** for fast linting and formatting (configured in `pyproject.toml`)
- **MyPy** for static type checking
- **pytest** for testing
- **Pre-commit hooks** — Ruff lint/format + standard checks (trailing whitespace, YAML validation, merge conflicts, etc.)
- **VS Code integration** with recommended extensions and settings template

## Project Structure

```
├── .vscode/
│   ├── extensions.json          # Recommended VS Code extensions
│   └── settings_template.json   # VS Code settings template
├── src/                         # Source code
│   ├── __init__.py
│   └── example.py               # Example module (intentionally messy — try ruff on it)
├── tests/                       # Tests
│   ├── __init__.py
│   └── test_example.py
├── bootstrap_env.py             # Cross-platform setup script
├── environment.yml              # Conda environment definition
├── pyproject.toml               # Project config, Ruff & MyPy settings
└── .pre-commit-config.yaml      # Pre-commit hook configuration
```

## Adding Dependencies

**Conda packages** (compiled/scientific libraries like NumPy, SciPy, etc.):

Edit `environment.yml`:
```yaml
dependencies:
  - python=3.11.8
  - numpy        # add conda packages here
  - pip
```

**Python packages** (pure Python libraries, dev tools):

Edit `pyproject.toml`:
```toml
[project.optional-dependencies]
dev = [
  "mypy",
  "pytest",
  "ruff",
  "pre-commit",
  "some-new-tool",   # add dev-only packages here
]
```

Then update your environment:
```bash
conda env update -f environment.yml --prune
pip install .[dev]
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Ruff not found in VS Code | Restart VS Code after activating the conda environment |
| Pre-commit not working | Run `pre-commit install` inside the activated environment |
| Environment issues | `python bootstrap_env.py --recreate` |
| Wrong Python interpreter in VS Code | Open Command Palette → "Python: Select Interpreter" → pick the conda env |

## Starting a New Project

1. Click **"Use this template"** on the [GitHub repository page](https://github.com/IDEALLab/python-ruff-conda-template) to create your own repo
2. Clone your new repository and run `python bootstrap_env.py`
3. Replace all placeholder names with your project's details:

| File | What to change |
|------|----------------|
| `pyproject.toml` | `name`, `description`, `authors` |
| `environment.yml` | `name` (this becomes your `conda activate` name) |
| `src/__init__.py` | Package docstring |
| `tests/__init__.py` | Package docstring |
| `README.md` | Title, description |
