"""Check/update a server simulator, or synchronize both local simulator copies."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

SIMULATOR_NAME = "generate_dynamic_regime_trajectories.py"
RELATIVE_FILES = (
    "time_varying_bifurcation_data/generate_dynamic_regime_trajectories.py",
    "diffrentbutconst_prams/generate_dynamic_regime_trajectories.py",
)
SIMULATOR_HELP = "generate_trainingdata root, diffrentbutconst_prams directory, or simulator .py file."


def simulator_paths(root: Path) -> list[Path]:
    """Use the available copies; two copies, when present, must agree."""
    root = Path(root).expanduser().resolve()
    if root.is_file():
        return [root]
    nested = [root / name for name in RELATIVE_FILES if (root / name).is_file()]
    if nested:
        return nested
    direct = root / SIMULATOR_NAME
    if direct.is_file():
        return [direct]
    raise FileNotFoundError(
        f"No {SIMULATOR_NAME} found at {root}. Pass --simulator-root pointing to "
        "diffrentbutconst_prams, the simulator .py file, or generate_trainingdata."
    )


def default_simulator_root() -> Path:
    configured = os.environ.get("GENERATE_TRAININGDATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    parent = Path(__file__).resolve().parents[3]
    candidates = (parent / "generate_trainingdata", parent / "diffrentbutconst_prams",
                  Path.home() / "diffrentbutconst_prams")
    for candidate in candidates:
        try:
            simulator_paths(candidate)
            return candidate
        except FileNotFoundError:
            continue
    return candidates[0]


DEFAULT_ROOT = default_simulator_root()
OLD_SIGNATURE = "def halvorsen_rhs(t: float, x: Array, p: float, *, drive: float) -> Array:"
NEW_SIGNATURE = """def halvorsen_rhs(
    t: float, x: Array, p: float, *, drive: float = 0.0,
    b: float = 4.0, c: float = 4.0,
) -> Array:
    \"\"\"Halvorsen with p=a; b=c=4 preserves all existing one-parameter runs.\"\"\""""


def updated_source(source: bytes) -> bytes:
    newline = "\r\n" if b"\r\n" in source else "\n"
    text = source.decode("utf-8").replace("\r\n", "\n")
    if NEW_SIGNATURE in text:
        return source
    if text.count(OLD_SIGNATURE) != 1:
        raise ValueError("Unrecognized Halvorsen function; refusing an ambiguous replacement.")
    start = text.index(OLD_SIGNATURE)
    end = text.index("\n\n\nRHS:", start)
    function = text[start:end].replace(OLD_SIGNATURE, NEW_SIGNATURE)
    for first, second in ((1, 2), (2, 0), (0, 1)):
        old = f"- 4.0 * x[{first}] - 4.0 * x[{second}]"
        if function.count(old) != 1:
            raise ValueError(f"Unrecognized coupling expression: {old}")
        function = function.replace(old, f"- b * x[{first}] - c * x[{second}]")
    return (text[:start] + function + text[end:]).replace("\n", newline).encode("utf-8")


def synchronize(root: Path, apply: bool = False) -> str:
    paths = simulator_paths(root)
    sources = [path.read_bytes() for path in paths]
    if any(source != sources[0] for source in sources[1:]):
        raise ValueError("The simulator copies differ. Review their differences before synchronizing.")
    updated = updated_source(sources[0])
    if updated != sources[0]:
        if not apply:
            raise ValueError("Halvorsen b/c extension is missing; run this script with --apply.")
        # Both transformations are validated before either file is written.
        written = []
        try:
            for path in paths:
                path.write_bytes(updated)
                written.append(path)
        except OSError:
            for path in written:
                path.write_bytes(sources[0])
            raise
    if any(path.read_bytes() != updated for path in paths):
        raise RuntimeError("Simulator synchronization failed.")
    return hashlib.sha256(updated).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-root", type=Path, default=DEFAULT_ROOT, help=SIMULATOR_HELP)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    digest = synchronize(args.simulator_root.expanduser().resolve(), args.apply)
    print(f"Checked {len(simulator_paths(args.simulator_root))} simulator file(s) with a/b/c support. SHA256: {digest}")


if __name__ == "__main__":
    main()
