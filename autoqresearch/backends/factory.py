"""Backend factory for the primitive-based execution flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator


IDEAL_MPS_MODES = {"ideal_mps", "mps", "statevector"}
NOISY_MODES = {"noisy_simulator", "noisy_heron"}
HARDWARE_MODES = {"hardware"}


@dataclass
class BackendConfig:
    """Configuration for backend creation."""

    mode: str = "ideal_mps"
    shots: int = 1000
    sampler_shots: int | None = None
    seed: int | None = 7
    transpile_optimization_level: int = 1
    noise_model: Any | None = None
    hardware_backend_name: str | None = None


@dataclass
class BackendBundle:
    """Execution bundle shared by the solvers."""

    mode: str
    backend: Any
    estimator: Any
    sampler: Any
    shots: int
    sampler_shots: int
    pass_manager: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_backend_mode(mode: str) -> str:
    """Map user-facing aliases onto the internal backend modes."""

    lowered = (mode or "ideal_mps").lower()
    if lowered in IDEAL_MPS_MODES:
        return "ideal_mps"
    if lowered in NOISY_MODES:
        return "noisy_simulator"
    if lowered in HARDWARE_MODES:
        return "hardware"
    raise ValueError(f"Unknown backend mode: {mode}")


def create_backend(config: BackendConfig) -> Any:
    """Backward-compatible helper returning only the raw backend."""

    return create_execution_context(config).backend


def create_execution_context(config: BackendConfig) -> BackendBundle:
    """Create the backend plus the estimator/sampler primitives."""

    mode = normalize_backend_mode(config.mode)
    shots = int(config.shots)
    sampler_shots = int(config.sampler_shots or config.shots)

    if mode == "ideal_mps":
        backend = AerSimulator(
            method="matrix_product_state",
            seed_simulator=config.seed,
        )
        metadata = {
            "mode": mode,
            "backend_name": "aer_mps",
        }
    elif mode == "noisy_simulator":
        if config.noise_model is None:
            raise NotImplementedError(
                "Noisy simulation is intentionally left as an extension point. "
                "Pass a configured noise_model when you are ready to add it."
            )
        backend = AerSimulator(
            method="density_matrix",
            noise_model=config.noise_model,
            seed_simulator=config.seed,
        )
        metadata = {
            "mode": mode,
            "backend_name": "aer_density_matrix",
        }
    else:
        raise NotImplementedError(
            "Hardware execution is not wired yet. "
            "Add the IBM Runtime primitives in create_execution_context() when needed."
        )

    estimator = BackendEstimatorV2(backend=backend)
    sampler = BackendSamplerV2(
        backend=backend,
        options={"default_shots": sampler_shots},
    )
    pass_manager = generate_preset_pass_manager(
        optimization_level=config.transpile_optimization_level,
        backend=backend,
    )

    return BackendBundle(
        mode=mode,
        backend=backend,
        estimator=estimator,
        sampler=sampler,
        shots=shots,
        sampler_shots=sampler_shots,
        pass_manager=pass_manager,
        metadata=metadata,
    )


def get_noise_scale_factor(config: BackendConfig) -> float:
    """Return a simple noise scale for reporting hooks."""

    mode = normalize_backend_mode(config.mode)
    if mode == "ideal_mps":
        return 0.0
    if mode == "noisy_simulator" and config.noise_model is not None:
        return 1.0
    return 0.0
