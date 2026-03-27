#!/usr/bin/env python3
"""IBM hardware adapters for the autoqresearch solver stack.

This module keeps all hardware-specific code isolated under ``hardware_runs/``.
It reuses the benchmark repo's IBM credential rotation and backend selection
logic, then exposes Runtime V2 sampler/estimator adapters that match the small
subset of the ``BackendSamplerV2`` / ``BackendEstimatorV2`` interface used by
``autoqresearch``.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoqresearch.backends.factory import BackendBundle


LOGGER = logging.getLogger("hardware_runs.autoq_hardware_backend")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
QOBENCH_SRC = (
    SCRIPT_DIR
    / "quantum-optimization-benchmarks"
    / "research_benchmark"
    / "src"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(QOBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(QOBENCH_SRC))

from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2
from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2
from qobench.hardware_manager import QuantumHardwareManager
from qobench.serialization import to_jsonable
try:
    from qiskit import qpy
except Exception:  # pragma: no cover - fallback path only
    qpy = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_job_id(job: Any) -> str | None:
    job_id = getattr(job, "job_id", None)
    if callable(job_id):
        try:
            job_id = job_id()
        except Exception:
            job_id = None
    if job_id is None:
        return None
    return str(job_id)


def _set_nested_attr(target: Any, dotted_name: str, value: Any) -> bool:
    current = target
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        if not hasattr(current, part):
            return False
        try:
            current = getattr(current, part)
        except Exception:
            return False
    leaf = parts[-1]
    if not hasattr(current, leaf):
        return False
    try:
        setattr(current, leaf, value)
    except Exception:
        return False
    return True


def _configure_sampler(target: Any, shots: int) -> None:
    if target is None:
        return
    if hasattr(target, "set_options"):
        try:
            target.set_options(shots=int(shots))
        except Exception:
            pass
    _set_nested_attr(target, "options.shots", int(shots))
    _set_nested_attr(target, "options.default_shots", int(shots))
    try:
        setattr(target, "default_shots", int(shots))
    except Exception:
        pass


def _configure_estimator(target: Any, shots: int) -> None:
    if target is None:
        return
    _set_nested_attr(target, "options.default_shots", int(shots))
    _set_nested_attr(target, "options.default_precision", 0.05)
    try:
        setattr(target, "default_shots", int(shots))
    except Exception:
        pass


def _configure_runtime_mitigation(target: Any) -> None:
    if target is None:
        return
    _set_nested_attr(target, "options.resilience_level", 1)
    _set_nested_attr(target, "options.dynamical_decoupling.enable", True)
    _set_nested_attr(target, "options.dynamical_decoupling.sequence_type", "XY4")
    _set_nested_attr(target, "options.twirling.enable_gates", True)
    _set_nested_attr(target, "options.twirling.num_randomizations", "auto")


def _infer_circuit(entry: Any) -> Any | None:
    if hasattr(entry, "num_qubits"):
        return entry
    if isinstance(entry, (tuple, list)) and entry:
        first = entry[0]
        if hasattr(first, "num_qubits"):
            return first
    return None


def _infer_required_qubits(entries: list[Any] | tuple[Any, ...] | Any) -> int:
    if entries is None:
        return 0
    if not isinstance(entries, (list, tuple)):
        entries = [entries]
    max_qubits = 0
    for entry in entries:
        circuit = _infer_circuit(entry)
        if circuit is not None:
            max_qubits = max(max_qubits, int(getattr(circuit, "num_qubits", 0)))
    return max_qubits


def _circuit_fingerprint(circuit: Any) -> str:
    digest = hashlib.sha1()
    digest.update(str(getattr(circuit, "num_qubits", 0)).encode("utf-8"))
    digest.update(str(getattr(circuit, "num_parameters", 0)).encode("utf-8"))
    if qpy is not None:
        try:
            buffer = io.BytesIO()
            qpy.dump(circuit, buffer)
            digest.update(buffer.getvalue())
            return digest.hexdigest()
        except Exception:
            pass
    try:
        digest.update(repr(circuit).encode("utf-8"))
    except Exception:
        digest.update(str(type(circuit)).encode("utf-8"))
    try:
        digest.update(str(circuit.count_ops()).encode("utf-8"))
    except Exception:
        pass
    try:
        digest.update(
            ",".join(sorted(param.name for param in circuit.parameters)).encode("utf-8")
        )
    except Exception:
        pass
    return digest.hexdigest()


def _collect_transpiled_metrics(compiled: Any, optimization_level: int) -> dict[str, Any]:
    gate_counts = {}
    try:
        gate_counts = {
            str(name): int(count)
            for name, count in compiled.count_ops().items()
            if name not in {"measure", "barrier"}
        }
    except Exception:
        gate_counts = {}

    one_q = 0
    two_q = 0
    meas = 0
    try:
        for inst, qargs, _ in compiled.data:
            if inst.name == "measure":
                meas += 1
            elif len(qargs) == 1:
                one_q += 1
            elif len(qargs) == 2:
                two_q += 1
    except Exception:
        pass

    try:
        depth = int(compiled.depth())
    except Exception:
        depth = 0

    return {
        "optimization_level": int(optimization_level),
        "transpiled_depth": int(depth),
        "transpiled_1q_gates": int(one_q),
        "transpiled_2q_gates": int(two_q),
        "transpiled_measurements": int(meas),
        "transpiled_num_qubits": int(getattr(compiled, "num_qubits", 0) or 0),
        "transpiled_num_parameters": int(getattr(compiled, "num_parameters", 0) or 0),
        "transpiled_gate_counts": gate_counts,
    }


def _apply_layout_to_observable(observable: Any, layout: Any) -> Any:
    if layout is None or observable is None:
        return observable
    if isinstance(observable, list):
        return [_apply_layout_to_observable(item, layout) for item in observable]
    if isinstance(observable, tuple):
        return tuple(_apply_layout_to_observable(item, layout) for item in observable)
    if hasattr(observable, "apply_layout"):
        try:
            return observable.apply_layout(layout)
        except Exception:
            return observable
    return observable


def _values_to_list(values: Any) -> Any:
    if hasattr(values, "tolist"):
        try:
            return values.tolist()
        except Exception:
            pass
    return values


def _remap_parameter_values(
    original_circuit: Any,
    transpiled_circuit: Any,
    parameter_values: Any,
) -> Any:
    values = _values_to_list(parameter_values)
    original_parameters = list(getattr(original_circuit, "parameters", []))
    transpiled_parameters = list(getattr(transpiled_circuit, "parameters", []))
    if not transpiled_parameters:
        return [] if isinstance(values, list) else values

    def _remap_single(sequence: Any) -> list[Any]:
        sequence = _values_to_list(sequence)
        if not isinstance(sequence, (list, tuple)):
            return [sequence]
        name_to_value = {}
        for index, parameter in enumerate(original_parameters):
            if index >= len(sequence):
                break
            name_to_value[parameter.name] = sequence[index]
        remapped: list[Any] = []
        fallback_index = 0
        for parameter in transpiled_parameters:
            if parameter.name in name_to_value:
                remapped.append(name_to_value[parameter.name])
            elif fallback_index < len(sequence):
                remapped.append(sequence[fallback_index])
                fallback_index += 1
            else:
                remapped.append(0.0)
        return remapped

    if isinstance(values, (list, tuple)) and values and isinstance(
        _values_to_list(values[0]),
        (list, tuple),
    ):
        return [_remap_single(item) for item in values]
    return _remap_single(values)


def _bind_circuit_if_possible(circuit: Any, parameter_values: Any) -> tuple[Any, bool]:
    values = _values_to_list(parameter_values)
    if not list(getattr(circuit, "parameters", [])):
        return circuit, False
    if values is None:
        return circuit, False
    if isinstance(values, (list, tuple)) and values and isinstance(
        _values_to_list(values[0]),
        (list, tuple),
    ):
        if len(values) != 1:
            return circuit, False
        values = _values_to_list(values[0])
    if not isinstance(values, (list, tuple)):
        values = [values]

    bind_map = {}
    for index, parameter in enumerate(list(circuit.parameters)):
        if index >= len(values):
            break
        bind_map[parameter] = values[index]
    if not bind_map:
        return circuit, False
    try:
        return circuit.assign_parameters(bind_map), True
    except Exception:
        return circuit, False


class HardwareJobRecorder:
    """Collect per-job metadata across one run."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    @property
    def count(self) -> int:
        return len(self.records)

    def record_submit(self, payload: dict[str, Any]) -> int:
        self.records.append(copy.deepcopy(to_jsonable(payload)))
        return len(self.records) - 1

    def mark_complete(
        self,
        index: int,
        *,
        final_status: str,
        complete_time_utc: str,
        error: str | None = None,
    ) -> None:
        if index < 0 or index >= len(self.records):
            return
        record = self.records[index]
        record["final_status"] = str(final_status)
        record["complete_time_utc"] = str(complete_time_utc)
        if error:
            record["error"] = str(error)

    def slice_from(self, start: int) -> list[dict[str, Any]]:
        return copy.deepcopy(to_jsonable(self.records[start:]))


class ManagedIBMBackendSelector:
    """Resolve a usable IBM backend just-in-time before each primitive call."""

    def __init__(
        self,
        *,
        manager: QuantumHardwareManager,
        recorder: HardwareJobRecorder,
        job_timeout_sec: float | None,
        min_log_interval_sec: float,
    ) -> None:
        self.manager = manager
        self.recorder = recorder
        self.job_timeout_sec = job_timeout_sec
        self.min_log_interval_sec = float(min_log_interval_sec)

    def select_backend_info(self, required_qubits: int) -> dict[str, Any]:
        self.manager.refresh_credentials_if_needed()
        info = self.manager._select_ibm_backend_info(int(required_qubits))
        if info is None:
            raise RuntimeError(
                f"No IBM backend supports {required_qubits} required qubits."
            )
        return info

    def transpile_circuit(
        self,
        *,
        backend_info: dict[str, Any],
        circuit: Any,
    ) -> tuple[Any, dict[str, Any]]:
        backend = backend_info["backend"]
        backend_name = str(
            backend_info.get("name", getattr(backend, "name", "unknown_backend"))
        )
        ansatz_id = _circuit_fingerprint(circuit)
        compiled = self.manager._get_transpiled_template(
            backend=backend,
            ansatz_id=ansatz_id,
            qiskit_template=circuit,
        )
        metrics = _collect_transpiled_metrics(
            compiled,
            optimization_level=self.manager.qiskit_optimization_level,
        )
        metrics["backend_name"] = backend_name
        metrics["ansatz_id"] = ansatz_id
        return compiled, metrics


class TrackedPrimitiveJob:
    """Wrap an IBM Runtime job and feed status/result metadata to the recorder."""

    def __init__(
        self,
        *,
        job: Any,
        manager: QuantumHardwareManager,
        recorder: HardwareJobRecorder,
        record_index: int,
        backend_name: str,
        primitive_label: str,
        timeout_sec: float | None,
        min_log_interval_sec: float,
    ) -> None:
        self._job = job
        self._manager = manager
        self._recorder = recorder
        self._record_index = record_index
        self._backend_name = backend_name
        self._primitive_label = primitive_label
        self._timeout_sec = timeout_sec
        self._min_log_interval_sec = float(min_log_interval_sec)
        self._cached_result = None
        self._finished = False

    def result(self, *args, **kwargs):
        if self._finished:
            return self._cached_result

        job_id = _extract_job_id(self._job)
        label = (
            f"IBM {self._primitive_label} job {job_id or 'unknown'} "
            f"({self._backend_name})"
        )
        try:
            self._manager._wait_for_terminal_status(
                runtime_handle=self._job,
                label=label,
                timeout_sec=self._timeout_sec,
                terminal_states={"DONE", "ERROR", "CANCELLED"},
                success_states={"DONE"},
                min_log_interval_sec=self._min_log_interval_sec,
            )
            result = self._job.result(*args, **kwargs)
        except Exception as exc:
            self._recorder.mark_complete(
                self._record_index,
                final_status="ERROR",
                complete_time_utc=_utcnow_iso(),
                error=str(exc),
            )
            try:
                self._manager.refresh_credentials_if_needed()
            except Exception:
                pass
            raise

        self._recorder.mark_complete(
            self._record_index,
            final_status="DONE",
            complete_time_utc=_utcnow_iso(),
        )
        try:
            self._manager.refresh_credentials_if_needed()
        except Exception:
            pass
        self._cached_result = result
        self._finished = True
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._job, name)


class IBMRuntimeSamplerAdapter:
    """Drop-in replacement for the sampler subset used by autoqresearch."""

    def __init__(self, *, backend, options: dict | None = None):
        if not isinstance(backend, ManagedIBMBackendSelector):
            raise TypeError(
                "IBMRuntimeSamplerAdapter expects a ManagedIBMBackendSelector."
            )
        self._selector = backend
        self._options = dict(options or {})

    def run(self, pubs, shots: int | None = None):
        required_qubits = _infer_required_qubits(pubs)
        backend_info = self._selector.select_backend_info(required_qubits)
        backend = backend_info["backend"]
        backend_name = str(
            backend_info.get("name", getattr(backend, "name", "unknown_backend"))
        )
        effective_shots = int(
            shots
            if shots is not None
            else self._options.get("default_shots", 1024)
        )
        normalized_pubs = pubs if isinstance(pubs, (list, tuple)) else [pubs]
        transpiled_pubs = []
        transpile_metrics = []
        for pub in normalized_pubs:
            if hasattr(pub, "num_qubits"):
                compiled, metrics = self._selector.transpile_circuit(
                    backend_info=backend_info,
                    circuit=pub,
                )
                transpiled_pubs.append(compiled)
                transpile_metrics.append(metrics)
            elif isinstance(pub, (tuple, list)) and pub and hasattr(pub[0], "num_qubits"):
                bound_circuit, consumed = _bind_circuit_if_possible(pub[0], pub[1] if len(pub) >= 2 else None)
                compiled, metrics = self._selector.transpile_circuit(
                    backend_info=backend_info,
                    circuit=bound_circuit,
                )
                if consumed:
                    transpiled_pubs.append(compiled)
                else:
                    payload = list(pub)
                    payload[0] = compiled
                    if len(payload) >= 2:
                        payload[1] = _remap_parameter_values(pub[0], compiled, payload[1])
                    transpiled_pubs.append(tuple(payload))
                transpile_metrics.append(metrics)
            else:
                transpiled_pubs.append(pub)
                transpile_metrics.append({})

        sampler = RuntimeSamplerV2(mode=backend)
        _configure_sampler(sampler, effective_shots)
        _configure_runtime_mitigation(sampler)
        record_index = self._selector.recorder.record_submit(
            {
                "primitive": "sampler",
                "backend_name": backend_name,
                "backend_qubits": int(backend_info.get("num_qubits", 0) or 0),
                "pending_jobs_at_submit": backend_info.get("pending_jobs"),
                "required_qubits": int(required_qubits),
                "shots": int(effective_shots),
                "job_id": None,
                "submit_time_utc": _utcnow_iso(),
                "provider": "ibm",
                "transpile_metrics": transpile_metrics,
            }
        )
        try:
            job = sampler.run(transpiled_pubs, shots=effective_shots)
        except Exception as exc:
            self._selector.recorder.mark_complete(
                record_index,
                final_status="ERROR",
                complete_time_utc=_utcnow_iso(),
                error=str(exc),
            )
            raise

        job_id = _extract_job_id(job)
        if job_id:
            self._selector.recorder.records[record_index]["job_id"] = job_id
        LOGGER.info(
            "IBM sampler submitted | backend=%s job_id=%s qubits=%s shots=%s",
            backend_name,
            job_id,
            required_qubits,
            effective_shots,
        )
        return TrackedPrimitiveJob(
            job=job,
            manager=self._selector.manager,
            recorder=self._selector.recorder,
            record_index=record_index,
            backend_name=backend_name,
            primitive_label="sampler",
            timeout_sec=self._selector.job_timeout_sec,
            min_log_interval_sec=self._selector.min_log_interval_sec,
        )


class IBMRuntimeEstimatorAdapter:
    """Drop-in replacement for the estimator subset used by autoqresearch."""

    def __init__(self, *, backend, options: dict | None = None):
        if not isinstance(backend, ManagedIBMBackendSelector):
            raise TypeError(
                "IBMRuntimeEstimatorAdapter expects a ManagedIBMBackendSelector."
            )
        self._selector = backend
        self._options = dict(options or {})

    def run(self, pubs=None, **kwargs):
        if pubs is None:
            pubs = kwargs.get("pubs")
        required_qubits = _infer_required_qubits(pubs)
        backend_info = self._selector.select_backend_info(required_qubits)
        backend = backend_info["backend"]
        backend_name = str(
            backend_info.get("name", getattr(backend, "name", "unknown_backend"))
        )
        effective_shots = int(self._options.get("default_shots", 1024))
        normalized_pubs = pubs if isinstance(pubs, (list, tuple)) else [pubs]
        transpiled_pubs = []
        transpile_metrics = []
        for pub in normalized_pubs:
            if not isinstance(pub, (tuple, list)) or not pub:
                transpiled_pubs.append(pub)
                transpile_metrics.append({})
                continue
            circuit = pub[0]
            if not hasattr(circuit, "num_qubits"):
                transpiled_pubs.append(pub)
                transpile_metrics.append({})
                continue

            bound_circuit = circuit
            params_consumed = False
            if len(pub) >= 3:
                bound_circuit, params_consumed = _bind_circuit_if_possible(
                    circuit,
                    pub[2],
                )
            compiled, metrics = self._selector.transpile_circuit(
                backend_info=backend_info,
                circuit=bound_circuit,
            )
            payload = list(pub)
            payload[0] = compiled
            if len(payload) >= 2:
                payload[1] = _apply_layout_to_observable(
                    payload[1],
                    getattr(compiled, "layout", None),
                )
            if len(payload) >= 3 and not params_consumed:
                payload[2] = _remap_parameter_values(circuit, compiled, payload[2])
            elif len(payload) >= 3 and params_consumed:
                payload = payload[:2]
            transpiled_pubs.append(tuple(payload))
            transpile_metrics.append(metrics)

        estimator = RuntimeEstimatorV2(mode=backend)
        _configure_estimator(estimator, effective_shots)
        _configure_runtime_mitigation(estimator)
        record_index = self._selector.recorder.record_submit(
            {
                "primitive": "estimator",
                "backend_name": backend_name,
                "backend_qubits": int(backend_info.get("num_qubits", 0) or 0),
                "pending_jobs_at_submit": backend_info.get("pending_jobs"),
                "required_qubits": int(required_qubits),
                "shots": int(effective_shots),
                "job_id": None,
                "submit_time_utc": _utcnow_iso(),
                "provider": "ibm",
                "transpile_metrics": transpile_metrics,
            }
        )
        try:
            job = estimator.run(pubs=transpiled_pubs)
        except Exception as exc:
            self._selector.recorder.mark_complete(
                record_index,
                final_status="ERROR",
                complete_time_utc=_utcnow_iso(),
                error=str(exc),
            )
            raise

        job_id = _extract_job_id(job)
        if job_id:
            self._selector.recorder.records[record_index]["job_id"] = job_id
        LOGGER.info(
            "IBM estimator submitted | backend=%s job_id=%s qubits=%s shots=%s",
            backend_name,
            job_id,
            required_qubits,
            effective_shots,
        )
        return TrackedPrimitiveJob(
            job=job,
            manager=self._selector.manager,
            recorder=self._selector.recorder,
            record_index=record_index,
            backend_name=backend_name,
            primitive_label="estimator",
            timeout_sec=self._selector.job_timeout_sec,
            min_log_interval_sec=self._selector.min_log_interval_sec,
        )


class AutoQHardwareBackendFactory:
    """Create hardware-backed BackendBundle objects for autoqresearch."""

    def __init__(
        self,
        *,
        ibm_credentials_json: Path,
        ibm_min_runtime_seconds: float = 60.0,
        qiskit_optimization_level: int = 3,
        job_status_log_interval: float = 120.0,
        job_timeout_sec: float | None = None,
        capture_calibration: bool = False,
    ) -> None:
        self.ibm_credentials_json = Path(ibm_credentials_json).expanduser().resolve()
        self.recorder = HardwareJobRecorder()
        self.job_timeout_sec = job_timeout_sec
        self.job_status_log_interval = float(job_status_log_interval)
        self.capture_calibration = bool(capture_calibration)
        self._calibration_snapshot = None

        self.manager = QuantumHardwareManager(
            ibm_credentials_json=str(self.ibm_credentials_json),
            use_aws=False,
            use_ibm=True,
            allow_simulators=False,
            enabled_qpu_ids={"ibm_quantum"},
            qiskit_optimization_level=int(qiskit_optimization_level),
        )
        self.manager.ibm_min_runtime_seconds = float(ibm_min_runtime_seconds)
        self.manager.job_status_log_interval = float(job_status_log_interval)

        init_status = self.manager.initialize()
        if not init_status.get("ibm_quantum", False):
            snapshot = self.manager.status_snapshot().get("ibm_quantum", {})
            reason = snapshot.get("last_error", "IBM hardware initialization failed.")
            raise RuntimeError(str(reason))

    @property
    def job_count(self) -> int:
        return self.recorder.count

    def job_records(self, start: int = 0) -> list[dict[str, Any]]:
        return self.recorder.slice_from(int(start))

    def status_snapshot(self) -> dict[str, dict[str, Any]]:
        return to_jsonable(self.manager.status_snapshot())

    def calibration_snapshot(self) -> dict[str, Any] | None:
        if not self.capture_calibration:
            return None
        if self._calibration_snapshot is None:
            self._calibration_snapshot = to_jsonable(
                self.manager.get_calibration_snapshot()
            )
        return copy.deepcopy(self._calibration_snapshot)

    def create_bundle(
        self,
        *,
        shots: int,
        sampler_shots: int,
        seed: int | None = None,
    ) -> BackendBundle:
        selector = ManagedIBMBackendSelector(
            manager=self.manager,
            recorder=self.recorder,
            job_timeout_sec=self.job_timeout_sec,
            min_log_interval_sec=self.job_status_log_interval,
        )
        return BackendBundle(
            mode="hardware",
            backend=selector,
            estimator=IBMRuntimeEstimatorAdapter(
                backend=selector,
                options={"default_shots": int(shots)},
            ),
            sampler=IBMRuntimeSamplerAdapter(
                backend=selector,
                options={"default_shots": int(sampler_shots)},
            ),
            shots=int(shots),
            sampler_shots=int(sampler_shots),
            pass_manager=None,
            metadata={
                "mode": "hardware",
                "provider": "ibm",
                "selection_policy": "least_busy",
                "ibm_min_runtime_seconds": float(self.manager.ibm_min_runtime_seconds),
                "seed": None if seed is None else int(seed),
            },
        )


@contextlib.contextmanager
def patch_autoq_primitives():
    """Patch the solver-local primitive references to use IBM Runtime adapters."""

    import qiskit.primitives as qiskit_primitives
    import autoqresearch.solvers.qubo_primitives as qubo_primitives

    original_estimator = getattr(qiskit_primitives, "BackendEstimatorV2", None)
    original_sampler = getattr(qiskit_primitives, "BackendSamplerV2", None)
    original_module_sampler = getattr(qubo_primitives, "BackendSamplerV2", None)

    qiskit_primitives.BackendEstimatorV2 = IBMRuntimeEstimatorAdapter
    qiskit_primitives.BackendSamplerV2 = IBMRuntimeSamplerAdapter
    qubo_primitives.BackendSamplerV2 = IBMRuntimeSamplerAdapter
    try:
        yield
    finally:
        if original_estimator is not None:
            qiskit_primitives.BackendEstimatorV2 = original_estimator
        if original_sampler is not None:
            qiskit_primitives.BackendSamplerV2 = original_sampler
        if original_module_sampler is not None:
            qubo_primitives.BackendSamplerV2 = original_module_sampler
