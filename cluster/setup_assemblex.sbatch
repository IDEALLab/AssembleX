#!/bin/bash
#SBATCH --job-name=assemblex_setup
#SBATCH --output=/cluster/scratch/fvonarx/logs/assemblex_setup_%j.out
#SBATCH --error=/cluster/scratch/fvonarx/logs/assemblex_setup_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
# No --gpus on purpose: RedMax is a CPU simulator and nothing installed here
# needs a GPU. setup.py builds the C++ extension with a hard-coded `-j8`, so
# 8 CPUs is the useful allocation.
#
# One-time initialisation of AssembleX on the cluster. Idempotent: re-running
# it re-checks every step and only redoes what is missing.
#
#   mkdir -p /cluster/scratch/fvonarx/logs
#   sbatch cluster/setup_assemblex.sbatch
#
# What it does, in order:
#   1. clone the repo + initialise the ASAPx submodule. Both are skipped when
#      already present, so an rsync'd tree works as a starting point too and
#      re-runs never touch an existing checkout.
#   2. pip install --user the Python dependencies (environment.yml + the ones
#      environment.yml is missing: torch, torch_geometric, coacd)
#   3. check redmax_py. The container ships it prebuilt, so this only
#      compiles ASAPx/simulation when the import is missing (or when
#      FORCE_REDMAX_BUILD=1).
#   4. verify the install headlessly (imports + a short simulation)
#
# It does NOT copy assembly data: assets/* is gitignored, so the meshes have
# to be staged from a machine that has them. Step 4 prints the exact rsync
# command for the test assembly.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override by exporting these before sbatch)
# ---------------------------------------------------------------------------
SCRATCH_DIR="${SCRATCH_DIR:-/cluster/scratch/fvonarx}"
REPO_DIR="${REPO_DIR:-${SCRATCH_DIR}/AssembleX}"
CONTAINER_PATH="${CONTAINER_PATH:-${SCRATCH_DIR}/assemble-x-container_latest.sif}"
REPO_URL="${REPO_URL:-https://github.com/IDEALLab/AssembleX.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
USER_BASE="${SCRATCH_DIR}/.local"
WORK_DIR="${SCRATCH_DIR}/.assemblex_setup"
CONSTRAINTS="${WORK_DIR}/constraints.txt"

# Strip trailing slashes so path comparisons and joins below are exact.
SCRATCH_DIR="${SCRATCH_DIR%/}"
REPO_DIR="${REPO_DIR%/}"

# Forward local storage and Euler network proxy settings into Apptainer.
export APPTAINERENV_PYTHONUSERBASE="${USER_BASE}"
export APPTAINERENV_http_proxy="${http_proxy:-}"
export APPTAINERENV_https_proxy="${https_proxy:-}"
# pip's cmake / ninja land in ${USER_BASE}/bin; PATH is special in Apptainer
# and must go through PREPEND_PATH rather than APPTAINERENV_PATH.
export APPTAINERENV_PREPEND_PATH="${USER_BASE}/bin"
export APPTAINER_CACHEDIR="${SCRATCH_DIR}/.apptainer/cache"
export APPTAINER_TMPDIR="${SCRATCH_DIR}/.apptainer/tmp"

mkdir -p "${SCRATCH_DIR}/logs" "${USER_BASE}" "${WORK_DIR}" \
         "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BIND_ARGS=(--bind "${SCRATCH_DIR}:${SCRATCH_DIR}")
case "${REPO_DIR}" in
    "${SCRATCH_DIR}"/*) ;;
    *) BIND_ARGS+=(--bind "${REPO_DIR}:${REPO_DIR}") ;;
esac

# Run a command inside the container, with the repo as working directory.
in_container() {
    apptainer exec --cleanenv "${BIND_ARGS[@]}" --pwd "${REPO_DIR}" \
        "${CONTAINER_PATH}" "$@"
}

# Same, but before the repo exists (working directory = scratch).
in_container_bare() {
    apptainer exec --cleanenv "${BIND_ARGS[@]}" --pwd "${SCRATCH_DIR}" \
        "${CONTAINER_PATH}" "$@"
}

section() { echo; echo "=== $* ==="; }
warn()    { echo "WARNING: $*" >&2; }
die()     { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Step 0: preflight
# ---------------------------------------------------------------------------
section "Step 0: preflight"
command -v apptainer >/dev/null 2>&1 || die "apptainer not found on PATH"
[ -f "${CONTAINER_PATH}" ] || die "container image not found: ${CONTAINER_PATH}"
# git is checked in step 1, and only on the paths that actually need it: an
# rsync'd tree is a valid starting point and needs no git at all.

echo "Scratch:   ${SCRATCH_DIR}"
echo "Repo:      ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "User base: ${USER_BASE}"

# A REPO_DIR outside SCRATCH_DIR is supported (an extra bind is added for it),
# but it is also exactly what a missing slash in the REPO_DIR default looks
# like: ${SCRATCH_DIR}AssembleX resolves to a sibling of the scratch directory,
# not a subdirectory of it, and the clone then lands somewhere unintended.
case "${REPO_DIR}/" in
    "${SCRATCH_DIR}"/*) ;;
    *) warn "REPO_DIR is not under SCRATCH_DIR:
           REPO_DIR    = ${REPO_DIR}
           SCRATCH_DIR = ${SCRATCH_DIR}
         Intentional? Then ignore this. Otherwise check the REPO_DIR line for
         a missing '/' between \${SCRATCH_DIR} and the rest of the path." ;;
esac
in_container_bare python3 -c 'import sys; print("Container python:", sys.version.split()[0])'

# ---------------------------------------------------------------------------
# Step 1: repository + ASAPx submodule
# ---------------------------------------------------------------------------
section "Step 1: repository"

# Three states, each with a different correct action. The earlier version
# tested only for main.py and otherwise cloned, which broke every time the
# directory was already there: `git clone` refuses a non-empty target, and
# re-cloning over a working tree would discard the staged assets/ data that is
# not in git.
[ -e "${REPO_DIR}" ] && [ ! -d "${REPO_DIR}" ] \
    && die "${REPO_DIR} exists but is not a directory"

if [ -f "${REPO_DIR}/main.py" ] && [ -f "${REPO_DIR}/environment.yml" ]; then
    # Already usable, however it got there (clone, rsync, tarball).
    echo "Checkout already present at ${REPO_DIR}; leaving it as it is."
elif { [ -d "${REPO_DIR}/.git" ] || [ -f "${REPO_DIR}/.git" ]; }; then
    # .git but no main.py: a clone that died partway (the usual cause is the
    # proxy dropping out). git will refuse the non-empty target on every later
    # run, so it has to be cleared rather than retried. Not removed
    # automatically: the path may be a typo pointing at real data, and this
    # script cannot tell. Re-run with ALLOW_CLOBBER=1 to let it delete.
    if [ "${ALLOW_CLOBBER:-0}" = "1" ]; then
        warn "removing the partial clone at ${REPO_DIR} (ALLOW_CLOBBER=1)"
        rm -rf "${REPO_DIR}"
        command -v git >/dev/null 2>&1 || die "git not found on PATH; cannot clone"
        echo "Cloning ${REPO_URL} (branch ${REPO_BRANCH}) into ${REPO_DIR}..."
        git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
    else
        die "${REPO_DIR} holds an interrupted clone: it has .git but no main.py.
       git refuses to clone into it, which is the 'already exists and is not an
       empty directory' error. Check the path is the one you meant, then:
         rm -rf ${REPO_DIR}
       and resubmit. Or resubmit with ALLOW_CLOBBER=1 to have this script do it:
         ALLOW_CLOBBER=1 sbatch cluster/setup_assemblex.sbatch"
    fi
elif [ -d "${REPO_DIR}" ] && [ -n "$(ls -A "${REPO_DIR}" 2>/dev/null)" ]; then
    die "${REPO_DIR} exists and is not empty, but holds no AssembleX checkout
       (no main.py + environment.yml). Cloning into it would fail, so this
       needs a decision rather than a guess: point REPO_DIR at the real
       checkout, or move that directory aside.
       It currently contains: $(ls -A "${REPO_DIR}" | head -5 | tr '\n' ' ')
       A wrong REPO_DIR is the usual cause; note that the default is
       \${SCRATCH_DIR}/AssembleX."
else
    command -v git >/dev/null 2>&1 || die "git not found on PATH; cannot clone"
    echo "Cloning ${REPO_URL} (branch ${REPO_BRANCH}) into ${REPO_DIR}..."
    mkdir -p "$(dirname "${REPO_DIR}")" \
        || die "cannot create $(dirname "${REPO_DIR}") - check the path and permissions"
    # -d, because an empty directory is a legitimate starting point that
    # plain `git clone` also accepts.
    rmdir "${REPO_DIR}" 2>/dev/null || true
    git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

# ASAPx is the active planner backend and is required. ATA is only needed for
# --seq-planner ATA, so it stays uninitialised here.
if [ -f "${REPO_DIR}/ASAPx/simulation/setup.py" ]; then
    echo "ASAPx backend already present."
elif [ -d "${REPO_DIR}/.git" ] || [ -f "${REPO_DIR}/.git" ]; then
    # .git can be a file rather than a directory (worktrees, nested submodules).
    command -v git >/dev/null 2>&1 \
        || die "git not found on PATH; cannot initialise the ASAPx submodule"
    echo "Initialising the ASAPx submodule..."
    git submodule update --init --recursive ASAPx
else
    # An rsync'd or extracted tree has no .git, so `git submodule update` would
    # only say 'not a git repository' and take the whole script down with it.
    die "ASAPx is missing and ${REPO_DIR} is not a git checkout, so the
       submodule cannot be fetched here. Either copy ASAPx/ in alongside the
       rest of the tree, or replace the directory with a real git clone."
fi
[ -f "${REPO_DIR}/ASAPx/simulation/setup.py" ] \
    || die "ASAPx is still empty after submodule init"

# ---------------------------------------------------------------------------
# Step 2: Python dependencies
# ---------------------------------------------------------------------------
section "Step 2: Python dependencies"

# numpy 2 breaks the pinned scientific stack, so every pip call below is run
# against this constraints file rather than relying on install order.
cat > "${CONSTRAINTS}" <<'EOF'
numpy<2.0.0
EOF

# Parse environment.yml into a pip-installable package list. Conda-only and
# build/GUI packages are dropped; a few conda names differ on PyPI.
cat > "${WORK_DIR}/parse_environment.py" <<'PYEOF'
"""Turn environment.yml into two pip package lists: required and optional.

Usage: parse_environment.py <required-out> <optional-out>  (cwd = repo root)
"""
import re
import sys

PYPI_MAPPING = {
    "opencv": "opencv-python-headless",
    "rtree": "Rtree",
    "pillow": "Pillow",
}
# python/pip: provided by the container. cmake/setuptools/wheel: installed
# explicitly by the build step. cuda/qt/pyqt: GUI or driver packages that have
# no business in a headless batch job.
SKIP_PKGS = {
    "python", "pip", "cuda", "cudatoolkit", "pyqt", "qt",
    "cmake", "setuptools", "wheel",
}
# Installed one by one afterwards so a single failure does not abort the rest.
OPTIONAL_PKGS = {"pygraphviz"}

required, optional = [], []
in_dependencies = False

with open("environment.yml") as f:
    for line in f:
        raw = line.rstrip()
        if not raw or raw.strip().startswith("#"):
            continue

        if not raw.startswith(" ") and not raw.startswith("\t"):
            in_dependencies = raw.strip().startswith("dependencies:")
            continue

        if not in_dependencies:
            continue

        stripped = raw.strip()
        if stripped.startswith("- pip:") or not stripped.startswith("-"):
            continue

        item = stripped[1:].strip().split("#")[0].strip()
        if not item:
            continue

        # Strip internal whitespace (e.g. "numpy <2.0.0" -> "numpy<2.0.0").
        item = re.sub(r"\s+", "", item)

        pkg_base = re.split(r"[=<>]", item)[0].lower()
        if pkg_base in SKIP_PKGS:
            continue

        if pkg_base in PYPI_MAPPING:
            item = PYPI_MAPPING[pkg_base] + item[len(pkg_base):]

        if "=" in item and not any(op in item for op in ["==", ">=", "<=", "!=", "~="]):
            item = re.sub(r"=", "==", item, count=1)

        (optional if pkg_base in OPTIONAL_PKGS else required).append(item)

with open(sys.argv[1], "w") as f:
    f.write("\n".join(required) + "\n")
with open(sys.argv[2], "w") as f:
    f.write("\n".join(optional) + "\n")
PYEOF

in_container python3 "${WORK_DIR}/parse_environment.py" \
    "${WORK_DIR}/pkgs_required.txt" "${WORK_DIR}/pkgs_optional.txt"
mapfile -t REQUIRED_PKGS < <(grep -v '^[[:space:]]*$' "${WORK_DIR}/pkgs_required.txt")
mapfile -t OPTIONAL_PKGS < <(grep -v '^[[:space:]]*$' "${WORK_DIR}/pkgs_optional.txt")

# Dependencies that are imported at module load but are NOT in environment.yml:
#   torch / torch_geometric — ASAPx/plan_sequence/planner/base.py and
#       generator/learning.py import them unconditionally, so the planner
#       registry cannot even be imported without them. Only the `learn`
#       generator actually uses them, hence the CPU wheels.
#   coacd — core/models.py:7, top-level.
EXTRA_REQUIRED=(coacd)
# Imported lazily, by one feature each (manual PDF rendering, mesh repair,
# weight training, tree plots). A failure here does not block planning.
EXTRA_OPTIONAL=(cairosvg svglib reportlab pymeshfix optuna numpy-stl)

echo "Required from environment.yml:"
printf '  %s\n' "${REQUIRED_PKGS[@]}"
echo "Additional required (missing from environment.yml):"
printf '  %s\n' "${EXTRA_REQUIRED[@]}"

echo
echo "Installing build tools..."
in_container python3 -m pip install --user --no-cache-dir --upgrade \
    -c "${CONSTRAINTS}" setuptools wheel

echo
echo "Installing required packages..."
in_container python3 -m pip install --user --no-cache-dir \
    -c "${CONSTRAINTS}" "${REQUIRED_PKGS[@]}" "${EXTRA_REQUIRED[@]}"

echo
if in_container python3 -c 'import torch' >/dev/null 2>&1; then
    echo "torch already provided by the container; skipping."
else
    echo "Installing torch (CPU wheel: nothing in the planner path uses CUDA)..."
    if ! in_container python3 -m pip install --user --no-cache-dir \
            -c "${CONSTRAINTS}" --index-url https://download.pytorch.org/whl/cpu torch; then
        warn "CPU wheel index unreachable; falling back to the default PyPI build"
        in_container python3 -m pip install --user --no-cache-dir -c "${CONSTRAINTS}" torch
    fi
fi

if in_container python3 -c 'import torch_geometric' >/dev/null 2>&1; then
    echo "torch_geometric already provided by the container; skipping."
else
    echo "Installing torch_geometric..."
    in_container python3 -m pip install --user --no-cache-dir -c "${CONSTRAINTS}" torch_geometric
fi

echo
echo "Installing optional packages (failures are reported, not fatal)..."
for pkg in "${OPTIONAL_PKGS[@]}" "${EXTRA_OPTIONAL[@]}"; do
    if in_container python3 -m pip install --user --no-cache-dir -c "${CONSTRAINTS}" "${pkg}"; then
        echo "  ok: ${pkg}"
    else
        warn "optional package failed to install: ${pkg}"
        case "${pkg}" in
            pygraphviz) echo "         (tree plots fall back to pydot, then to a spring layout)" ;;
            cairosvg|svglib|reportlab) echo "         (manual PDF compilation will be unavailable)" ;;
            optuna)     echo "         (train_heuristic_weights will be unavailable)" ;;
        esac
    fi
done

# ---------------------------------------------------------------------------
# Step 3: build the RedMax physics binding
# ---------------------------------------------------------------------------
section "Step 3: RedMax physics binding (redmax_py)"

# The container image already ships a built redmax_py -- which is why the
# original job script never compiled anything, and why compiling anyway broke
# this step. So the build is opt-in now: it runs only when the import is
# missing, or when FORCE_REDMAX_BUILD=1 says the ASAPx sources have moved ahead
# of the prebuilt binary.
if [ "${FORCE_REDMAX_BUILD:-0}" != "1" ] \
        && in_container python3 -c 'import redmax_py' >/dev/null 2>&1; then
    REDMAX_PATH="$(in_container python3 -c 'import redmax_py; print(redmax_py.__file__)')"
    echo "redmax_py is already provided by the container:"
    echo "  ${REDMAX_PATH}"
    echo "Not rebuilding it. If the ASAPx checkout is newer than that binary,"
    echo "re-run with FORCE_REDMAX_BUILD=1 to compile from ASAPx/simulation."
else
    if [ "${FORCE_REDMAX_BUILD:-0}" = "1" ]; then
        echo "FORCE_REDMAX_BUILD=1: compiling redmax_py from the ASAPx checkout."
    else
        echo "redmax_py is not importable; compiling it from the ASAPx checkout."
    fi

    in_container bash -c 'command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1' \
        || die "no C++ compiler in the container; RedMax cannot be built"

    # cmake 4 removed compatibility with cmake_minimum_required(VERSION < 3.5),
    # and the vendored externals still declare 3.1
    # (simulation/externals/CMakeLists.txt:2), 2.8.12 (glfw) and 2.8 (imgui), so
    # a cmake-4 configure fails outright with 'Compatibility with CMake < 3.5
    # has been removed'. setup.py hard-codes its cmake argv and takes no extra
    # -D flags, so the fix is a pinned cmake<4 in ${USER_BASE}/bin, which
    # APPTAINERENV_PREPEND_PATH already puts ahead of the container's cmake.
    # CMAKE_POLICY_VERSION_MINIMUM covers the fallback path, where the pin could
    # not be installed and the container's own cmake has to do the configure.
    export APPTAINERENV_CMAKE_POLICY_VERSION_MINIMUM="3.5"
    echo "Installing cmake<4 for the build (the vendored externals predate cmake 4)..."
    if in_container python3 -m pip install --user --no-cache-dir "cmake<4"; then
        echo "Build will use: $(in_container bash -c 'cmake --version | head -1')"
        echo "               $(in_container bash -c 'command -v cmake')"
    else
        warn "could not install cmake<4; falling back to the container's cmake
         with CMAKE_POLICY_VERSION_MINIMUM=3.5, which may not be honoured by
         every cmake version"
    fi

    # --no-build-isolation: use the setuptools and the cmake installed above
    #   rather than a fresh isolated build env, which would re-download them.
    # --force-reinstall --no-deps: the version in setup.py never changes
    #   (0.0.1), so without this a re-run can be a no-op against a stale
    #   binary. redmax_py declares no install_requires, so --no-deps is free.
    # This compiles RedMax and takes 10-20 minutes on 8 cores.
    if ! in_container python3 -m pip install --user --no-cache-dir --no-build-isolation \
            --force-reinstall --no-deps \
            -c "${CONSTRAINTS}" "${REPO_DIR}/ASAPx/simulation"; then
        echo >&2
        echo "ERROR: the RedMax build failed. The usual causes are:" >&2
        echo "  - cmake 4.x configuring the vendored externals, which declare" >&2
        echo "    cmake_minimum_required(VERSION 3.1) and older. The giveaway is" >&2
        echo "    'Compatibility with CMake < 3.5 has been removed'. The fix is a" >&2
        echo "    cmake<4 first on PATH; check whether the install above worked." >&2
        echo "  - no OpenGL development files in the container. The CMake project" >&2
        echo "    calls find_package(OpenGL REQUIRED) even for the headless build," >&2
        echo "    so the image needs libgl1-mesa-dev (or equivalent)." >&2
        echo "  - no C++ toolchain in the container." >&2
        echo "Look for the first 'CMake Error' line in the log above." >&2
        echo >&2
        echo "Note: the container normally ships a prebuilt redmax_py, so a build" >&2
        echo "is not usually needed at all. Unset FORCE_REDMAX_BUILD to use it." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 4: verification
# ---------------------------------------------------------------------------
section "Step 4: verification"

SITE_PACKAGES="$(in_container python3 -c 'import site; print(site.getusersitepackages())')"
export APPTAINERENV_PYTHONPATH="${SITE_PACKAGES}:${REPO_DIR}"
# Headless: no X server on a compute node, and VTK/matplotlib both default to
# interactive backends.
export APPTAINERENV_MPLBACKEND="Agg"
export APPTAINERENV_PYVISTA_OFF_SCREEN="true"

cat > "${WORK_DIR}/verify_install.py" <<'PYEOF'
"""Headless post-install check: imports, then a short RedMax simulation.

Deliberately avoids SimRenderer.replay (which the README's test_simple_sim.py
ends with) because it opens a GL window and cannot run on a compute node.
"""
import os
import sys

# The container runs with --pwd <repo>, so cwd is the fallback.
REPO_DIR = os.environ.get("ASSEMBLEX_REPO") or os.getcwd()
sys.path.insert(0, REPO_DIR)

failures = []


def check(label, fn):
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - report, don't abort
        failures.append(label)
        print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
    else:
        print(f"  ok    {label}" + (f" ({detail})" if detail else ""))


print("Core dependencies:")
check("numpy", lambda: __import__("numpy").__version__)
check("scipy", lambda: __import__("scipy").__version__)
check("trimesh", lambda: __import__("trimesh").__version__)
check("networkx", lambda: __import__("networkx").__version__)
check("pyvista", lambda: __import__("pyvista").__version__)
check("coacd", lambda: "")
check("torch", lambda: __import__("torch").__version__)
check("torch_geometric", lambda: __import__("torch_geometric").__version__)
check("redmax_py", lambda: __import__("redmax_py").__file__)

print("\nMatplotlib backend (must be Agg on a compute node):")


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    return matplotlib.get_backend()


check("matplotlib", _mpl)

print("\nAssembleX entry point:")
check("core.assembly", lambda: __import__("core.assembly", fromlist=["Eval"]).Eval.__name__)

print("\nASAPx registries (these pull in redmax_py, torch and torch_geometric):")
sys.path.insert(0, os.path.join(REPO_DIR, "ASAPx"))
check(
    "plan_sequence.generator",
    lambda: ",".join(sorted(__import__("plan_sequence.generator", fromlist=["generators"]).generators)),
)
check(
    "plan_sequence.planner",
    lambda: ",".join(sorted(__import__("plan_sequence.planner", fromlist=["planners"]).planners)),
)

print("\nRedMax simulation (50 steps of assets/box/box_stack.xml, no viewer):")


def _sim():
    import numpy as np
    import redmax_py as redmax

    model = os.path.join(REPO_DIR, "ASAPx", "assets", "box", "box_stack.xml")
    sim = redmax.Simulation(model)
    sim.reset(backward_flag=False)
    for _ in range(50):
        sim.set_u(np.zeros(sim.ndof_u))
        sim.forward(1, verbose=False, test_derivatives=False)
    q = sim.get_q()
    return f"ndof_q={len(q)}, |q|={float(np.linalg.norm(q)):.4f}"


check("redmax simulation", _sim)

if failures:
    print(f"\nFAILED: {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("\nAll checks passed.")
PYEOF

export APPTAINERENV_ASSEMBLEX_REPO="${REPO_DIR}"
in_container python3 "${WORK_DIR}/verify_install.py"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section "Setup complete"
cat <<EOF
PYTHONPATH for later jobs:
  ${SITE_PACKAGES}:${REPO_DIR}

Still to do before the test run: stage the assembly data. assets/* is
gitignored (only assets/tools/ ships with the repo), so the meshes are not in
the clone. From a machine that has them:

  rsync -av <local>/AssembleX/assets/data/asap/01234/ \\
      ${REPO_DIR}/assets/data/asap/01234/

  # optional, the trained heuristic weights; without it the planner falls
  # back to settings.heuristic_weights and prints a warning
  rsync -av <local>/AssembleX/assets/heuristic_weights_optuna.json \\
      ${REPO_DIR}/assets/

Then verify the install on a real planning run:

  sbatch cluster/test_sequence_planner.sbatch
EOF
