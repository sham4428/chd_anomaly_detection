import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

FILES_TO_COPY = (
    "app.py",
    "streamlit_app.py",
    "model.py",
    "inference.py",
    "config.py",
    "data_preprocessing.py",
    "train.py",
    "optimize_thresholds.py",
    "evaluate_model.py",
    "export_onnx.py",
    "requirements.txt",
    "requirements-deploy.txt",
    "render.yaml",
    "README.md",
    "DEPLOY.md",
    "PROJECT_GUIDE.md",
    "PROJECT_INDEX.md",
    "ecg_ae.pth",
    "pcg_ae.pth",
    "cxr_ae.pth",
    "thresholds.npz",
)

DIRECTORIES_TO_COPY = ("docs",)


def is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def copy_bundle(output_dir, include_data=False):
    output_dir = output_dir.resolve()
    project_root = PROJECT_ROOT.resolve()

    if output_dir == project_root or is_relative_to(output_dir, project_root):
        raise ValueError("The output directory must be outside the project directory.")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Choose a new empty path."
        )

    output_dir.mkdir(parents=True)
    copied = []
    skipped = []

    for relative_path in FILES_TO_COPY:
        source = project_root / relative_path
        destination = output_dir / relative_path
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(relative_path)
        else:
            skipped.append(relative_path)

    directories = list(DIRECTORIES_TO_COPY)
    if include_data:
        directories.append("data")

    for relative_path in directories:
        source = project_root / relative_path
        destination = output_dir / relative_path
        if source.is_dir():
            shutil.copytree(source, destination)
            copied.append(f"{relative_path}/")
        else:
            skipped.append(f"{relative_path}/")

    print(f"Bundle created: {output_dir}")
    print(f"Copied entries: {len(copied)}")
    if skipped:
        print("Skipped missing entries:")
        for relative_path in skipped:
            print(f"  - {relative_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a clean upload bundle without modifying the source project."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT.parent / f"{PROJECT_ROOT.name}_upload",
        help="Output directory. It must not already exist or be inside the project.",
    )
    parser.add_argument(
        "--include-data",
        action="store_true",
        help="Include data/. This can make the bundle very large.",
    )
    args = parser.parse_args()
    copy_bundle(args.output, include_data=args.include_data)


if __name__ == "__main__":
    main()
