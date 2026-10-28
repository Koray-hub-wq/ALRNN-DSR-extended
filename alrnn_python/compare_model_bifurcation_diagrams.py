"""Run the existing bifurcation analysis with this repository's AL-RNN loader.

All analysis options are passed through to the existing script. The backend can
be selected with --backend-script. Constant training parameters retain their
vertical lines; time-varying training parameters get a pale blue range.
"""

import argparse
import importlib.util
from pathlib import Path
import sys

from alrnn_model import load_saved_model
from bifurcation_training_markers import install_training_markers


def find_backend(explicit=None):
    candidates = [explicit] if explicit is not None else [
        Path.home() / "diffrentbutconst_prams/compare_model_bifurcation_diagrams.py",
        Path(__file__).resolve().parents[2] / "generate_trainingdata/diffrentbutconst_prams/compare_model_bifurcation_diagrams.py",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() != Path(__file__).resolve():
            return candidate.resolve()
    raise FileNotFoundError("Pass --backend-script /path/to/diffrentbutconst_prams/compare_model_bifurcation_diagrams.py")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend-script", type=Path)
    args, remaining = parser.parse_known_args(argv)
    path = find_backend(args.backend_script)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("alrnn_extended_bifurcation_backend", path)
    backend = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = backend
    spec.loader.exec_module(backend)
    backend.load_alrnn_model = load_saved_model
    install_training_markers(backend)
    original_argv = sys.argv
    try:
        sys.argv = [str(path), *remaining]
        backend.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
