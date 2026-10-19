"""Environment and reference-repo guards."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
REFERENCE_DIR = REPO / "reference" / "quantum-oracle-sketching"
REFERENCE_COMMIT = "a3fe8d6fe777750749f5e90131180e016088bdaf"
VENV_NAME = ".venv-qos-ref"


def require_qos_ref_venv(prefix=None):
    """Refuse to run outside .venv-qos-ref (the main pinned env must never run JAX code)."""
    prefix = Path(prefix if prefix is not None else sys.prefix)
    if prefix.name != VENV_NAME:
        raise RuntimeError(f"qostight must run under {VENV_NAME}, not {prefix}")


def reference_commit():
    out = subprocess.run(["git", "-C", str(REFERENCE_DIR), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def load_reference(expected_commit=REFERENCE_COMMIT):
    """Import the reference modules at the pinned commit (enables JAX x64 via utils)."""
    require_qos_ref_venv()
    commit = reference_commit()
    if commit != expected_commit:
        raise RuntimeError(f"Reference commit {commit} != pinned {expected_commit}")
    if str(REFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(REFERENCE_DIR))
    import primitives, qos, qos_sampling, qsvt, utils  # noqa: E401
    return SimpleNamespace(commit=commit, qos=qos, qos_sampling=qos_sampling, qsvt=qsvt,
                           utils=utils, primitives=primitives)
