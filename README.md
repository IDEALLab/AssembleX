# AssembleX

AssembleX plans physically feasible disassembly sequences for multi-part
product assemblies and generates the matching human-readable assembly manuals.

Given the part meshes (OBJ files) of one assembly, it does three things:

1. **Disassembly sequence planning.** It computes an order in which the parts
   can be removed, checking each step for collisions, gravity stability, the
   number of parts that must be held, and optionally robot-arm reachability.
2. **Rendering.** It produces a per-step disassembly GIF and the per-frame
   motion path for each removed part.
3. **Manual generation.** It writes per-step instructions, tool decisions,
   manual pages, and failure feedback using LLM and VLM models.

The tool exists to benchmark and compare planning strategies (random, geometric
heuristics, a learned graph neural network, and LLM-guided search) and to
measure the cost of the resulting plans, for example the estimated robot-arm
assembly time. Planning runs on the RedMax physics simulator through a fork of
the [ASAP](https://github.com/yunshengtian/ASAP) sequence planner.

> **Platform.** RedMax is a C++ simulator that builds on Linux only. On Windows,
> use [WSL](https://learn.microsoft.com/windows/wsl/install).

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Credits and Acknowledgments](#credits-and-acknowledgments)
- [Contact](#contact)

## Architecture

```
raw OBJ files
   │
   ▼
core/preprocess.py ──► normalised assembly ──► Assembly / Eval  (core/assembly.py)
                                            │
                                            ▼
                              SequencePlanner  (core/sequence_planner.py)
                              ├── ASAPx backend  (active, RedMax physics)
                              └── ATA backend    (legacy)
                                            │
                                  tree.pkl + stats.json
                                            │
                                            ▼
                  renderer ──────────► GIFs and per-step motion paths
                  Feedback ──────────► manuals, instructions, failure feedback
                  ToolAnalyzer ──────► tool decisions and poses
```

ASAPx is the active backend. It evaluates every candidate step against gravity
stability, tool feasibility, and grasp constraints, and it can score sequences
to select a low-cost plan. The ATA backend is kept for one reason: its
path planner is more general and is configured to find more complex disassembly
paths than ASAPx. It reasons only about geometric assemblability, ignoring
gravitational and tool failures, and it has no mechanism for optimality. ATA is
therefore the better choice when theoretical assemblability is the only
criterion of interest.

The planning graph (`tree.pkl`) and the run metadata (`stats.json`, holding the
sequence, timings, and CLI arguments) are the interchange format between the
planner and every downstream stage. A detailed map of the codebase, including
the core classes, the two planner backends, and the generator, planner, and
optimizer registries, lives in [CLAUDE.md](CLAUDE.md).

## Installation

The four numbered steps below are the supported install path. A convenience
`setup.sh` runs steps 1 through 3 in one go.

Before starting, make sure you have:

- Linux, or Windows with WSL.
- [Conda](https://docs.conda.io/) (Miniconda or Miniforge).
- `cmake` and a C++ toolchain (`build-essential` or `g++`) to build RedMax.
- An OpenAI API key, needed only for the LLM and VLM manual-generation steps.

### 1. Clone the repository and fetch the planner backends

The planner backends are git submodules. **ASAPx** is the active backend and is
required. **ATA** is a legacy backend, only needed if you run with
`--seq-planner ATA`, so installing it is optional.

```bash
git clone https://github.com/IDEALLab/AssembleX.git
cd AssembleX
# Required: the ASAPx backend (and its nested submodules)
git submodule update --init --recursive ASAPx
```

To also install the optional ATA backend:

```bash
git submodule update --init --recursive ATA
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate assemblex
```

### 3. Build the RedMax physics binding (ASAPx backend)

```bash
cd ASAPx/simulation
python setup.py install
cd ../..
```

If `g++` fails, run `sudo apt update && sudo apt install build-essential` and
retry. To confirm the simulator built correctly:

```bash
cd ASAPx && python test_sim/test_simple_sim.py --model box/box_stack --steps 2000 && cd ..
```

### 4. Provide your OpenAI API key

Manual and instruction generation read the key from the `OPENAI_API_KEY`
environment variable.

```bash
export OPENAI_API_KEY="sk-..."
```

> The optional `learn` generator needs extra PyTorch and PyTorch Geometric
> dependencies; see the [ASAP repository](https://github.com/yunshengtian/ASAP).

## Usage

All subcommands run through `main.py`, which dispatches on a `test_type`
positional argument.

```bash
python main.py <test_type> --id <assembly_id> [--dir <data_dir>] [options]
```

`--id` accepts a single ID (`00042`) or an inclusive range (`00010-00050`).
`--dir` is resolved under `assets/` and defaults to `data`.

One test assembly, `04489`, ships with the repository in `assets/data/`, and it
is what an omitted `--id` falls back to. A fresh checkout therefore runs the
whole pipeline without downloading a dataset first:

```bash
python main.py test_pipeline
```

which is the same as spelling both out:

```bash
python main.py test_pipeline --id 04489 --dir data
```

Run the full pipeline over a range, in parallel:

```bash
python main.py test_pipeline_batch --id 00000-00100
```

Re-render an already-planned assembly from its saved plan:

```bash
python main.py test_render --id 04489 --storage-dir assets/output/<timestamp>/04489
```

For `test_pipeline`, the `-x` flag selects which stages run: `g` for gravity,
`c` for collisions, `i` for instructions, `t` for tools, `f` for feedback,
`m` for manuals, and `v` for the stitched video. All stages run by default;
prefixing the string with `x` inverts the selection, running everything except
the listed stages.

```bash
python main.py test_pipeline -x gc    # only gravity and collisions
python main.py test_pipeline -x xim   # everything except instructions and manuals
```

To choose a planner and generator:

```bash
python main.py test_pipeline --generator heur-out --planner dfs --max-grippers 2
```

The subcommands are grouped by purpose across three handler modules:

| Group | Module | Subcommands |
|---|---|---|
| Pipeline | [run_pipeline.py](run_pipeline.py) | `test_pipeline`, `test_pipeline_batch`, `test_render` |
| Data generation | [run_data.py](run_data.py) | `data_assembly_time`, `data_sequence_runtime`, `train_heuristic_weights`, `data_heuristic_validation`, `data_manual_validation`, `data_validate_cost`, `data_filter_assemblies`, `test_convex_decomp`, `test_tool_needed`, `collect_tool_data`, `collect_tool_axes` |
| Diagnostics | [run_debug.py](run_debug.py) | `test_divide_optimizer`, `test_param_sweep`, `test_collision`, `test_PCA`, `test_tools`, `test_tool_naming`, `test_gravity`, `test_collision_resolver`, `test_collision_graph_batch`, `test_archive_ASAP` |

Run `python main.py --help` for the full list of subcommands and flags.

Each assembly writes its outputs to a `storage_dir`, by default
`assets/output/<timestamp>/<id>/`:

```
storage_dir/
├── log/
│   ├── tree.pkl        # planning graph
│   ├── stats.json      # sequence, timings, divide split, CLI args
│   └── ...
├── paths/              # per-step motion (npy frames)
├── 0_<part>.gif, …     # per-step disassembly GIFs
└── ...                 # manuals, instructions, feedback
```

## Configuration

`settings.py` is the single source of truth for runtime tuning. It holds the
model selection (`LLM_model`, `VLM_model`), the global render on/off switch,
planner behaviour, the heuristic and divide-optimizer weights, and the
arm-pipeline parameters. Prefer editing it over adding flags when changing
global behaviour.

The OpenAI key is read from the `OPENAI_API_KEY` environment variable and is
required for any step that calls an LLM or VLM. The `--token-limit` flag sets a
cumulative token budget across all assemblies in a run; once the budget is
exceeded, further LLM calls are skipped and the non-LLM stages still complete.
For reproducibility, every run copies its `settings.py` and writes a
`stats.json` summary into the output directory.

## Project Structure

```
AssembleX/
├── main.py                 # CLI entry point and dispatcher
├── setup.sh                # convenience installer (install steps 1-3)
├── run_pipeline.py         # pipeline subcommand handlers
├── run_data.py             # data-generation subcommand handlers
├── run_debug.py            # diagnostic subcommand handlers
├── run_common.py           # shared CLI helpers (ID resolution, summaries)
├── settings.py             # central configuration
├── core/                   # importable package (domain logic)
│   ├── assembly.py         # Eval / Assembly / Object core
│   ├── models.py           # Object, Step
│   ├── sequence_planner.py # SequencePlanner (wraps ASAPx / ATA)
│   ├── simulation.py       # Simulation, ContactTree (collision, gravity)
│   ├── feedback_generator.py  # LLM/VLM manual and instruction generation
│   ├── llm.py              # LLM/VLM client and response caching
│   ├── manual_generator.py    # manual page rendering
│   ├── manual_validator.py    # manual-fidelity validation
│   ├── tool_analyzer.py    # per-part tool decisions and poses
│   ├── tool_eval.py        # tool-decision evaluation against human labels
│   ├── renderer.py         # GIF stitching
│   ├── collision_checker.py   # collision queries
│   ├── perturbation.py     # geometric collision resolver
│   ├── preprocess.py       # OBJ normalisation
│   ├── separate_mesh.py    # split multi-part GLB/OBJ scenes into per-part .obj
│   └── plot_comparison.py  # comparison-summary plotting
├── ASAPx/                  # active planner backend (fork of ASAP) and RedMax sim
├── ATA/                    # legacy planner backend (fork of Assemble-Them-All)
├── assets/                 # data, tools, prompts, outputs, caches
│   ├── data/04489/         # the test assembly shipped with the repository
│   └── tools/              # tracked tool catalog (screwdrivers, allen key)
├── tests/                  # pytest suite
├── environment.yml         # conda environment (name: assemblex, python 3.11)
├── pyproject.toml          # packaging, ruff and mypy configuration
└── CLAUDE.md               # detailed codebase map
```

## Examples

An assembly is a folder of `.obj` part meshes under `assets/<dir>/<id>/`, where
`<dir>` is what `--dir` selects (default `data`). One assembly is tracked in this
repository as the test case: `04489`, a seven-part assembly in `assets/data/`
taken from the ASAP multi-part dataset. It is small enough to plan quickly and
still exercises the subassembly path, since it splits into two legs joined only
by a crossbar that comes off first. Omitting `--id` selects it. Only its part
meshes and its `normalization.json` and `contact_graph.json` are tracked; the
`.sdf` collision caches are generated on demand next to the meshes. For anything
beyond that single example, point `--dir` at your own meshes or at an ASAP
dataset (see the download links in
[ASAPx/README.md](ASAPx/README.md)). The tool catalog under `assets/tools/` is
tracked and ships with a Phillips-head screwdriver, a hex torque screwdriver,
and a hex allen key; it feeds the tool-decision and tool-feasibility steps.
Planning and rendering write per-step disassembly GIFs into each assembly's
output directory, and the `data_assembly_time` benchmark additionally produces
per-assembly and cross-assembly cost charts.

## Contributing

Contributions are welcome through issues and pull requests. The code follows
[ruff](https://docs.astral.sh/ruff/) with `line-length = 88` and the rule sets
configured in `pyproject.toml` (pycodestyle, pyflakes, isort, pep8-naming,
pyupgrade, bugbear, and others); the `ASAPx/` and `ATA/` submodules are
excluded. Run `ruff check .` before submitting, and keep emojis and decorative
formatting out of code and markdown. The development extras (`ruff`, `mypy`,
`pytest`, `pre-commit`) install with `pip install -e ".[dev]"`. Branch off
`main`, keep changes focused, and open a pull request with a clear description.
Read [CLAUDE.md](CLAUDE.md) first: it documents the conventions that are easy to
trip over, such as the ASAPx/ATA module-eviction rule and the use of `tree.pkl`
and `stats.json` as the interchange format.

## License

This project is released under the [MIT License](LICENSE). The bundled `ASAPx/`
and `ATA/` backends derive from prior work (see below) and carry their own
upstream licenses, which the MIT license here does not override.

## Credits and Acknowledgments

This project builds directly on two works by Yunsheng Tian and collaborators.
The ASAPx backend is a fork of
[ASAP](https://github.com/yunshengtian/ASAP) (Tian et al., *Automated Sequence
Planning for Complex Robotic Assembly with Physical Feasibility*, ICRA 2024).
The ATA backend is a fork of
[Assemble-Them-All](https://github.com/yunshengtian/Assemble-Them-All) (Tian et
al., *Assemble Them All: Physics-Based Planning for Generalizable Assembly by
Disassembly*, SIGGRAPH Asia 2022). Both rely on the [RedMax](https://github.com/sueda/redmax) (REDMAX: Efficient & Flexible Approach for Articulated Dynamics, Wang et al., SIGGRAPH 2019) differentiable
rigid-body simulator for the feasibility and stability checks, and the manual
and instruction generation uses the OpenAI API.

I developed this work in the [IDEAL Lab](https://ideal.ethz.ch) at ETH Zürich.

## Contact

Maintained by Faustin von Arx ([@FaustinVonArx](https://github.com/FaustinVonArx)).
Reach me at fvonarx@ethz.ch, or open a
[GitHub issue](https://github.com/IDEALLab/AssembleX/issues) for
questions and bugs.
