import shutil
import subprocess
import sys
from pathlib import Path


def preprocess(args):
    # raise NotImplementedError("This is a placeholder function. Implement your processing logic here.")
    print("Running process_mesh and saving output...")

    # Extract id and dir from args
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)

    # Create command to pass to ATA
    command = [
        "python",
        "ATA/assets/process_mesh",
        "--input-dir",
        str(source_dir),
        "--output-dir",
        str(target_dir),
        "--subdivide",
    ]

    # Find path to the assembly Conda environment
    conda_path = shutil.which("conda")
    if conda_path:
        base_conda_dir = Path(conda_path).resolve().parents[1]

        if sys.platform == "win32":
            assembly_exe = base_conda_dir / "envs" / "assembly" / "python.exe"
        else:
            assembly_exe = base_conda_dir / "envs" / "assembly" / "bin" / "python"

        print(f"Dynamic Python Engine: {assembly_exe}")
    else:
        raise OSError(
            "Conda executable not found. Please ensure Conda is installed and added to your system PATH."
        )

    print("Running preprocessing...")
    result = subprocess.run(
        [str(assembly_exe), *command[1:]], capture_output=True, text=True
    )

    if result.returncode == 0:
        print("Preprocessing completed successfully!")
        print(result.stdout)
    else:
        print("Preprocessing failed with error:")
        print(result.stderr)

    return result.returncode == 0


def preprocess_direct(args):
    # This function can be used to call the processing logic directly without subprocess, if the processing code is implemented in Python and can be imported as a module.
    from ATA.assets import process_mesh

    source_dir = args.source_dir
    # Decide whether the source must first be split into per-part .obj files.
    # GLB files are scene containers that process_mesh cannot read directly, so
    # their presence forces splitting automatically; an explicit --separate (or
    # args.separate=True) also forces it.
    separate = getattr(args, "separate", False) or any(
        p.suffix.lower() == ".glb" for p in Path(source_dir).iterdir()
    )
    if separate:
        from core.separate_mesh import process_directory

        # separate_mesh writes one subdir per input GLB/OBJ; expect a single assembly file in source_dir.
        separated_root = Path(source_dir) / "separated"
        process_directory(source_dir, str(separated_root))
        subdirs = sorted(d for d in separated_root.iterdir() if d.is_dir())
        if len(subdirs) != 1:
            raise ValueError(
                f"--separate expects a single input file in {source_dir}, found {len(subdirs)} separated subdirs."
            )
        source_dir = str(subdirs[0])

    return process_mesh.process_mesh(
        source_dir, args.target_dir, args.subdivide, scale=args.scale
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess mesh data for AssembleX."
    )
    parser.add_argument(
        "--target-dir", type=str, required=True, help="Directory to save processed data"
    )
    parser.add_argument(
        "--source-dir", type=str, required=True, help="Directory containing raw data"
    )
    parser.add_argument(
        "--subdivide",
        action="store_true",
        help="Whether to subdivide mesh edges during processing",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Run separate_mesh.py to split multi-part GLB/OBJ scenes into per-part .obj files before preprocessing",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform scale factor applied to source meshes before normalization (use to correct source-unit mistakes)",
    )

    args = parser.parse_args()

    if preprocess_direct(args):
        print("Preprocessing succeeded. You can now run the evaluation tests.")
    else:
        print(
            "Preprocessing failed. Please check the error messages above and try again."
        )
