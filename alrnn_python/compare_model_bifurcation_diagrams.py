"""Run the existing bifurcation analysis with this repository's AL-RNN loader.

All analysis options are passed through to the existing script. The backend can
be selected with --backend-script. Constant training parameters retain their
vertical lines; time-varying training parameters get a pale blue range.
--use-training-normalization supports globally standardized mixed datasets.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from alrnn_model import load_saved_model
from bifurcation_training_markers import install_training_markers


@dataclass
class TrainingCoordinates:
    mean: np.ndarray
    std: np.ndarray
    phi_boundary: float
    phi_scale: float

    def metadata(self):
        return dict(mean=self.mean.tolist(), std=self.std.tolist(),
                    phi_boundary=self.phi_boundary, phi_scale=self.phi_scale)


def load_training_coordinates(data_dir):
    data_dir = Path(data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8-sig"))
    if metadata.get("config", {}).get("standardization") != "global":
        raise ValueError(f"Training-normalized bifurcations require global standardization: {data_dir}")
    mean = np.load(data_dir / "trajectory_mean.npy", allow_pickle=False)
    std = np.load(data_dir / "trajectory_std.npy", allow_pickle=False)
    if mean.shape != std.shape or mean.ndim != 3 or mean.shape[1] != 1 or not mean.shape[0] or not mean.shape[2]:
        raise ValueError(f"Expected matching (trajectory, 1, channel) mean/std arrays: {data_dir}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(f"Invalid training mean/std: {data_dir}")
    if not np.allclose(mean, mean[0], atol=1e-6, rtol=1e-6) or not np.allclose(std, std[0], atol=1e-6, rtol=1e-6):
        raise ValueError(f"Training statistics are not global across trajectories: {data_dir}")
    reference = metadata.get("phi_reference", {})
    try:
        boundary, scale = float(reference["boundary"]), float(reference["scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid phi_reference: {data_dir}") from exc
    if not np.isfinite([boundary, scale]).all() or scale <= 0:
        raise ValueError(f"Invalid phi_reference: {data_dir}")
    return TrainingCoordinates(mean[0, 0].astype(np.float64), std[0, 0].astype(np.float64), boundary, scale)


def install_training_coordinates(backend, args):
    if args.model_type != "alrnn" or not args.model_runs:
        raise ValueError("--use-training-normalization requires --model-type alrnn and --model-runs.")
    if args.training_data_dirs is not None and len(args.training_data_dirs) != len(args.model_runs):
        raise ValueError("Pass one --training-data-dirs entry per model run.")
    coordinates = {}
    for index, run in enumerate(args.model_runs):
        run = Path(run).expanduser().resolve()
        if args.training_data_dirs is not None:
            data_dir = Path(args.training_data_dirs[index]).expanduser().resolve()
        else:
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8-sig"))
            saved_dir = metrics.get("args", {}).get("data_dir")
            if not saved_dir:
                raise ValueError(f"No saved data_dir; pass --training-data-dirs for {run}")
            data_dir = backend.find_training_data_dir(saved_dir, run, args.repo_root)
        coordinates[run] = load_training_coordinates(data_dir)

    # Plot every model and ground truth in the first run's fixed coordinates.
    reference_run = Path(args.model_runs[0]).expanduser().resolve()
    reference = coordinates[reference_run]
    if any(value.mean.shape != reference.mean.shape for value in coordinates.values()):
        raise ValueError("All compared models must have the same observed dimension.")
    regime = backend.regime_by_name(args.regime)
    original_load = backend.load_alrnn_model
    original_forecast = backend.forecast_alrnn_model
    original_cache_metadata = backend.cache_metadata
    loaded_coordinates = {}

    def load_model(run_dir, device):
        model, metrics, checkpoint = original_load(run_dir, device)
        scaling = coordinates[Path(run_dir).expanduser().resolve()]
        if model.N != scaling.mean.size:
            raise ValueError("Checkpoint observed dimension differs from its training statistics.")
        loaded_coordinates[id(model)] = scaling
        return model, metrics, checkpoint

    def standardize(data):
        data = np.asarray(data)
        if data.ndim != 3 or data.shape[-1] != reference.mean.size:
            raise ValueError("Simulated observations do not match training coordinates.")
        shape = (data.shape[0], 1, data.shape[-1])
        return (
            ((data - reference.mean) / reference.std).astype(np.float32),
            np.broadcast_to(reference.mean, shape).copy(),
            np.broadcast_to(reference.std, shape).copy(),
        )

    def forecast(model, trajectory, phi_value, context_steps, forecast_steps, device):
        scaling = loaded_coordinates[id(model)]
        raw = np.asarray(trajectory) * reference.std + reference.mean
        model_input = ((raw - scaling.mean) / scaling.std).astype(np.float32)
        raw_parameter = float(backend.raw_from_phi(regime, phi_value))
        model_phi = (raw_parameter - scaling.phi_boundary) / scaling.phi_scale
        prediction = original_forecast(model, model_input, model_phi, context_steps, forecast_steps, device)
        return ((prediction * scaling.std + scaling.mean - reference.mean) / reference.std).astype(np.float32)

    def cache_metadata(settings, selected_regime):
        metadata = original_cache_metadata(settings, selected_regime)
        metadata["training_coordinates"] = {
            "reference_run": str(reference_run),
            "models": {str(run): value.metadata() for run, value in coordinates.items()},
        }
        return metadata

    backend.load_alrnn_model = load_model
    backend.standardize_trajectories = standardize
    backend.forecast_alrnn_model = forecast
    backend.cache_metadata = cache_metadata
    print(f"Bifurcation coordinates: fixed training scaling of {reference_run}", flush=True)


def install_provenance(backend, backend_path, normalization="per_simulated_trajectory"):
    recorded_sources = []
    cache_reused = False

    def json_value(value):
        if isinstance(value, Path):
            return str(value.expanduser().resolve())
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    def wrap_loader(original, model_type, checkpoint_index, info_index):
        def load(run_dir, *args, **kwargs):
            result = original(run_dir, *args, **kwargs)
            checkpoint = Path(result[checkpoint_index]).expanduser().resolve()
            digest = hashlib.sha256()
            with checkpoint.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            info = result[info_index]
            recorded_sources.append({
                "run_dir": str(Path(run_dir).expanduser().resolve()),
                "model_type": model_type,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": digest.hexdigest(),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "training_args": info.get("args") if model_type == "alrnn" else info,
                "model_config": info.get("model_config") if model_type == "alrnn" else None,
                "parameter_counts": info.get("parameter_counts") if model_type == "alrnn" else None,
                "provenance_status": "recorded_at_computation",
            })
            return result
        return load

    if hasattr(backend, "load_alrnn_model"):
        backend.load_alrnn_model = wrap_loader(backend.load_alrnn_model, "alrnn", 2, 1)
    if hasattr(backend, "load_model"):
        backend.load_model = wrap_loader(backend.load_model, "dynamix", 4, 2)

    original_save = backend.save_points_cache
    original_load = backend.load_points_cache
    original_write = backend.write_plots

    def save_cache(path, metadata, *args, **kwargs):
        # Provenance travels with the cache, not with today's checkpoint files.
        metadata = {**metadata, "model_sources": recorded_sources}
        return original_save(path, metadata, *args, **kwargs)

    def load_cache(*args, **kwargs):
        nonlocal cache_reused
        result = original_load(*args, **kwargs)
        cache_reused = True
        return result

    def cache_matches(path, expected_metadata):
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
            metadata.pop("model_sources", None)
            return metadata == expected_metadata
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return False

    def write_plots(*args, **kwargs):
        written = original_write(*args, **kwargs)
        settings = backend.parse_args()
        cache = settings.cache if settings.cache is not None else settings.output.with_suffix(".npz")
        with np.load(cache, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
        sources = metadata.get("model_sources", [])
        models = []
        for index, run_dir in enumerate(settings.model_runs):
            resolved = str(Path(run_dir).expanduser().resolve())
            source = next((item for item in sources if item.get("run_dir") == resolved), None)
            models.append({
                **(source or {"run_dir": resolved, "checkpoint_path": None,
                              "checkpoint_sha256": None,
                              "provenance_status": "not_recorded_in_cache"}),
                "index": index, "label": kwargs["labels"][index],
            })
        for name in settings.pretrained_model_names:
            index = len(models)
            models.append({"index": index, "label": kwargs["labels"][index],
                           "model_type": "pretrained", "name": name,
                           "repo_root": settings.pretrained_repo_root,
                           "provenance_status": "pretrained_checkpoint_not_recorded"})
        payload = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "backend_script": str(Path(backend_path).resolve()),
            "command": list(sys.argv),
            "normalization": (
                "fixed_training" if metadata.get("training_coordinates")
                else "per_simulated_trajectory" if cache_reused else normalization
            ),
            "models": models,
            "settings": vars(settings),
            "cache": {"path": str(Path(cache).resolve()), "reused": cache_reused,
                      "metadata": metadata},
            "plots": [str(Path(path).resolve()) for path in written],
        }
        for path in written:
            path = Path(path)
            included = list(range(len(models)))
            component_name = None
            for component in kwargs["components"]:
                count = len(kwargs["components"])
                if path == backend.output_for_component(kwargs["output"], component, count):
                    component_name = component
                for index, label in enumerate(["Ground Truth", *kwargs["labels"]]):
                    if path == backend.individual_output_path(kwargs["output"], component, label, count):
                        included = [] if index == 0 else [index - 1]
                        component_name = component
            sidecar = path.with_suffix(".sources.json")
            document = {**payload, "plot": str(path.resolve()),
                        "component": component_name, "plot_model_indices": included}
            sidecar.write_text(json.dumps(document, indent=2, default=json_value) + "\n", encoding="utf-8")
            print(f"Saved model provenance: {sidecar.resolve()}", flush=True)
        return written

    backend.save_points_cache = save_cache
    backend.load_points_cache = load_cache
    backend.cache_matches = cache_matches
    backend.write_plots = write_plots


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
    parser.add_argument("--use-training-normalization", action="store_true")
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
        if args.use_training_normalization:
            install_training_coordinates(backend, backend.parse_args())
        install_provenance(
            backend, path,
            normalization="fixed_training" if args.use_training_normalization else "per_simulated_trajectory",
        )
        backend.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
