"""Training observations only: workers publish, one parent renders and journals.

No training decisions are made here. Every I/O boundary is best effort so a
broken terminal, queue or disk cannot replace a training result or exception.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from collections import deque
import contextvars
from datetime import datetime, timezone
from functools import wraps
import inspect
import json
import logging
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
import uuid

from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

_CONTEXT = contextvars.ContextVar("gpse_execution", default={})
_CLIENT = None
_SESSION = None
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class EventChannel:
    """Parent queue + per-process append journals, safe under forced termination.

    A killed writer can corrupt a multiprocessing pipe/lock. Independent JSONL
    producers isolate partial writes; the parent consumes complete lines only.
    The transport carries observations and never drives training execution.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.owner_pid = os.getpid()
        self.local = queue.Queue()
        self.offsets = {}
        self.pending = deque()

    def __getstate__(self):
        return {"directory": self.directory, "owner_pid": self.owner_pid}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.local = None

    def put(self, event, timeout=None):
        if os.getpid() == self.owner_pid:
            self.local.put(event)
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(event, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        descriptor = os.open(
            self.directory / f"producer_{os.getpid()}.jsonl",
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("Event producer write made no progress")
                remaining = remaining[written:]
        finally:
            os.close(descriptor)

    def get(self, timeout=0.2):
        deadline = time.monotonic() + timeout
        while True:
            if self.pending:
                return self.pending.popleft()
            batch = []
            terminal = False
            while True:
                try:
                    event = self.local.get_nowait()
                except queue.Empty:
                    break
                if event is None:
                    terminal = True
                else:
                    batch.append(event)
            for path in sorted(self.directory.glob("producer_*.jsonl")):
                try:
                    with path.open("rb") as handle:
                        handle.seek(self.offsets.get(path, 0))
                        while True:
                            line = handle.readline()
                            if not line.endswith(b"\n"):
                                break
                            self.offsets[path] = handle.tell()
                            try:
                                batch.append(json.loads(line))
                            except (ValueError, UnicodeDecodeError):
                                continue
                except OSError:
                    continue
            batch.sort(key=lambda item: item["timestamp"])
            self.pending.extend(batch)
            if terminal:
                self.pending.append(None)
            if self.pending:
                return self.pending.popleft()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty
            time.sleep(min(0.025, remaining))


def plain(value):
    return _ANSI.sub("", str(value)).replace("\r", "")


def terminal_active():
    return _SESSION is not None and _SESSION.pid == os.getpid()


def get_client_config():
    return _CLIENT[1] if _CLIENT is not None else None


def init_client(config):
    global _CLIENT
    _CLIENT = (os.getpid(), config) if config else None


def configure_optuna():
    # Import before removing its default handler: Optuna creates it on import.
    import optuna

    optuna.logging.disable_default_handler()
    optuna.logging.enable_propagation()
    optuna.logging.set_verbosity(logging.DEBUG)


def publish(stage, status, **fields):
    """Submit an event, without ever propagating monitoring failures."""
    try:
        if _CLIENT is None or _CLIENT[0] != os.getpid():
            return
        config = _CLIENT[1]
        event = {
            "schema_version": 1,
            "event_id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": config["run_id"],
            "run_execution_id": config.get("execution_id", config["run_id"]),
            "model": None,
            "repeat": None,
            "fold": None,
            "trial": None,
            **_CONTEXT.get(),
            "stage": stage,
            "status": status,
            "pid": os.getpid(),
            **fields,
        }
        if stage == "run":
            event["execution_id"] = config.get("execution_id", config["run_id"])
        config["queue"].put(event, timeout=0.2)
    except Exception:
        pass


def _log_sink(message):
    record = message.record
    publish(
        "log",
        "record",
        level=record["level"].name,
        message=plain(record["message"]),
        source=record["name"],
        exception=plain(message) if record["exception"] else None,
        **{k: v for k, v in record["extra"].items() if k in {"model", "repeat", "fold", "trial"}},
    )


class LogStream:
    """Line-buffer Python stdout/stderr into the same logging event route."""

    def __init__(self, level):
        self.level = level
        self.buffer = ""
        self.lock = threading.RLock()
        self.encoding = "utf-8"

    def isatty(self):
        return False

    def write(self, value):
        with self.lock:
            self.buffer += value
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                if line.strip():
                    logger.log(self.level, plain(line))
        return len(value)

    def flush(self):
        with self.lock:
            if self.buffer.strip():
                logger.log(self.level, plain(self.buffer))
            self.buffer = ""


@contextmanager
def routed_output():
    stdout, stderr = LogStream("INFO"), LogStream("WARNING")
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield
    finally:
        stdout.flush()
        stderr.flush()


def configure_worker(config, log_file, log_level="DEBUG"):
    """Discard inherited terminal/file sinks; worker files are independent."""
    global _CLIENT, _SESSION
    _SESSION = None
    _CLIENT = (os.getpid(), config) if config else None
    from gpse.utils.log_utils import _capture_std_logging, _capture_warnings

    try:
        logger.remove()
        logger.add(_log_sink, level="DEBUG", catch=True, format="{message}")
        _capture_std_logging()
        _capture_warnings()
        configure_optuna()
    except Exception:
        pass
    try:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level=log_level,
            enqueue=False,
            colorize=False,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra} | {message}",
            diagnose=False,
            catch=True,
        )
    except Exception as exc:
        publish("log", "record", level="WARNING", message=f"Worker log file unavailable: {exc}")
    sys.stdout = LogStream("INFO")
    sys.stderr = LogStream("WARNING")


def result_metrics(stage, result):
    if stage == "trial" and isinstance(result, (int, float)):
        return {"robust_objective": float(result)}
    if not isinstance(result, dict):
        return {}
    if stage == "model":
        agg = result.get("cv_aggregation", {})
        return {
            **{
                k: v
                for k, v in agg.items()
                if k.startswith("cv_") and isinstance(v, (int, float, str))
            },
            "cv_std_metric": agg.get("stability_metric"),
            **{
                k.replace("avg_ensemble_", "holdout_ensemble_"): v
                for k, v in result.items()
                if k.startswith("avg_ensemble_")
            },
        }
    if stage in {"fold", "stacking"}:
        return {
            "train": result.get("train_metrics", {}),
            "validation": result.get("val_metrics", {}),
            "holdout": result.get("test_metrics", {}),
        }
    if stage == "optimization":
        return {
            "robust_objective": result.get("best_value"),
            "trials": result.get("n_trials"),
            "reused_from_repeat": result.get("reused_from_repeat"),
        }
    return {}


def failure_status(exc):
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted"
    if type(exc).__name__ == "TrialPruned":
        return "pruned"
    return "failed"


def observed(stage):
    """Observe an existing method, preserving its return/exception semantics."""

    def decorate(fn):
        signature = inspect.signature(fn)

        @wraps(fn)
        def wrapped(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs).arguments
            context = dict(_CONTEXT.get())
            for key, source in (
                ("model", "model_name"),
                ("repeat", "repeat_idx"),
                ("fold", "fold_idx"),
            ):
                if source in bound:
                    value = bound[source]
                    context[key] = value + 1 if key != "model" else value
            if "trial" in bound:
                context["trial"] = bound["trial"].number + 1
            token = _CONTEXT.set(context)
            execution_id = uuid.uuid4().hex
            started = time.monotonic()
            publish(stage, "started", execution_id=execution_id)
            try:
                with logger.contextualize(**context):
                    result = fn(*args, **kwargs)
            except BaseException as exc:
                publish(
                    stage,
                    failure_status(exc),
                    execution_id=execution_id,
                    elapsed_s=time.monotonic() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            else:
                try:
                    metrics = result_metrics(stage, result)
                except Exception:
                    metrics = {}
                publish(
                    stage,
                    "failed" if stage == "fold" and result is None else "completed",
                    execution_id=execution_id,
                    elapsed_s=time.monotonic() - started,
                    metrics=metrics,
                )
                return result
            finally:
                _CONTEXT.reset(token)

        return wrapped

    return decorate


@contextmanager
def observation(stage, **context):
    token = _CONTEXT.set({**_CONTEXT.get(), **context})
    execution_id = uuid.uuid4().hex
    started = time.monotonic()
    publish(stage, "started", execution_id=execution_id)
    try:
        with logger.contextualize(**_CONTEXT.get()):
            yield
    except BaseException as exc:
        publish(
            stage,
            failure_status(exc),
            execution_id=execution_id,
            elapsed_s=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    else:
        publish(stage, "completed", execution_id=execution_id, elapsed_s=time.monotonic() - started)
    finally:
        _CONTEXT.reset(token)


class ConsoleRenderer:
    """Only this object writes training output to the terminal."""

    def __init__(
        self, models, repeats, folds, verbosity="NORMAL", progress="auto", level="INFO", stream=None
    ):
        stream = stream or sys.stderr
        interactive = bool(getattr(stream, "isatty", lambda: False)())
        interactive = interactive and os.environ.get("TERM") != "dumb" and not os.environ.get("CI")
        self.console = Console(
            file=stream, force_terminal=interactive, color_system="auto" if interactive else None
        )
        self.dynamic = interactive and progress != "off"
        self.verbosity = verbosity
        self.level = logging._nameToLevel.get(level, logging.INFO)
        self.repeats, self.folds = repeats, folds
        self.models = {
            name: {"status": "pending", "active": {}, "folds": 0, "repeats": 0} for name in models
        }
        self.phase = "Initializing"
        self.started = time.monotonic()
        self.live = None
        self.last_refresh = 0.0

    def refresh(self, force=False):
        if self.live:
            refresh_now = force or time.monotonic() - self.last_refresh >= 0.25
            self.live.update(self.table(), refresh=refresh_now)
            if refresh_now:
                self.last_refresh = time.monotonic()

    def table(self):
        finished = sum(
            s["status"] in {"completed", "failed", "interrupted"} for s in self.models.values()
        )
        table = Table(
            title=f"GPSE · {self.phase} · models {finished}/{len(self.models)} · {time.monotonic() - self.started:.0f}s",
            expand=True,
        )
        table.add_column("Model")
        table.add_column("Status")
        table.add_column("Repeats")
        table.add_column("CV folds")
        table.add_column("Active stages")
        for name, state in self.models.items():
            active = ", ".join(state["active"].values())
            table.add_row(
                Text(name),
                state["status"],
                f"{state['repeats']}/{self.repeats}",
                f"{state['folds']}/{self.repeats * self.folds}",
                Text(active),
            )
        return table

    def start(self):
        if self.dynamic:
            self.live = Live(
                self.table(),
                console=self.console,
                refresh_per_second=4,
                redirect_stdout=False,
                redirect_stderr=False,
                auto_refresh=False,
            )
            self.live.start()

    def accept(self, event):
        stage, status, model = event["stage"], event["status"], event.get("model")
        if stage == "log":
            severity = logging._nameToLevel.get(event["level"], logging.INFO)
            if severity < self.level:
                return
            if severity >= logging.WARNING or self.verbosity == "DEBUG":
                tag = " ".join(
                    str(event[k])
                    for k in ("model", "repeat", "fold", "trial")
                    if event.get(k) is not None
                )
                self.refresh()
                self.console.print(Text(f"{event['level']} {tag} {event['message']}".strip()))
            return
        self.phase = (
            stage
            if stage in {"run", "data", "selection", "stacking", "holdout", "reports"}
            else self.phase
        )
        if stage == "run":
            self.phase = "Initializing" if status == "started" else status.title()
        elif stage == "model" and status == "started":
            self.phase = "Training models"
        if model in self.models:
            state = self.models[model]
            key = event.get("execution_id", event["event_id"])
            if stage == "model":
                state["status"] = status
            if status == "started":
                label = stage
                for field, prefix in (("repeat", "R"), ("fold", "F"), ("trial", "T")):
                    if event.get(field) is not None:
                        label += f" {prefix}{event[field]}"
                state["active"][key] = label
            else:
                state["active"].pop(key, None)
                if stage == "repeat" and status == "completed":
                    state["repeats"] += 1
                if stage == "fold" and status == "completed":
                    state["folds"] += 1
                if stage == "model":
                    state["active"].clear()
        self.refresh()
        visible = stage in {"run", "model", "data", "selection", "stacking", "holdout", "reports"}
        visible = (
            visible or self.verbosity in {"VERBOSE", "DEBUG"} or status in {"failed", "interrupted"}
        )
        if (visible and self.level <= logging.INFO) or status in {"failed", "interrupted"}:
            # Live carries routine progress; persistent lines keep warnings and final metrics visible.
            if (
                not self.dynamic
                or status in {"failed", "interrupted"}
                or (stage in {"run", "model"} and status == "completed")
            ):
                ids = " ".join(
                    f"{key}={event[key]}"
                    for key in ("model", "repeat", "fold", "trial")
                    if event.get(key) is not None
                )
                detail = event.get("error", "")
                if event.get("elapsed_s") is not None:
                    detail += f" elapsed={event['elapsed_s']:.2f}s"
                if stage == "run" and status == "completed":
                    detail += f" outputs={len(event.get('completed_models', []))}"

                metrics = event.get("metrics") or {}
                if metrics:
                    if self.verbosity == "DEBUG":
                        detail += " " + json.dumps(metrics, ensure_ascii=False, default=str)
                    else:
                        parts = []
                        if metrics.get("cv_mean") is not None:
                            parts.append(f"CV Mean={metrics['cv_mean']:.6f}")
                        if metrics.get("cv_std") is not None:
                            parts.append(
                                f"CV Std={metrics['cv_std']:.6f} ({metrics.get('cv_std_metric') or 'configured stability'})"
                            )
                        for key in (
                            "holdout_ensemble_pearson",
                            "holdout_ensemble_accuracy",
                            "robust_objective",
                        ):
                            if metrics.get(key) is not None:
                                label = key.replace("_", " ").title()
                                parts.append(f"{label}={metrics[key]:.6f}")
                        for split, label in (("validation", "Validation"), ("holdout", "Holdout")):
                            values = metrics.get(split) or {}
                            for key in ("pearson", "accuracy"):
                                if values.get(key) is not None:
                                    parts.append(f"{label} {key}={values[key]:.6f}")
                        detail += " " + " ".join(parts)
                self.console.print(Text(f"{stage} {status} {ids} {detail}".strip()))

    def close(self):
        if self.live:
            self.refresh(force=True)
            self.live.stop()


class TrainingMonitor:
    def __init__(self, predictor, models):
        self.predictor = predictor
        self.models = list(dict.fromkeys(models))
        self.pid = os.getpid()
        self.journal = None
        self.thread = None
        self.sink = None
        self.renderer = None
        self.config = None
        self.stop = threading.Event()

    def start(self):
        global _CLIENT, _SESSION
        try:
            logger.complete()
            execution_id = uuid.uuid4().hex
            self.events = EventChannel(self.predictor.logs_dir / "events" / execution_id)
            self.config = {
                "queue": self.events,
                "run_id": self.predictor.paths.run_id,
                "execution_id": execution_id,
            }
            _CLIENT = (self.pid, self.config)
            _SESSION = self
            verbosity = getattr(self.predictor, "console_verbosity", "NORMAL")
            from gpse.utils.log_utils import console_log_level

            level = console_log_level()
            if level == "DEBUG":
                verbosity = "DEBUG"
            self.renderer = ConsoleRenderer(
                self.models,
                self.predictor.n_repeats,
                self.predictor.n_splits,
                verbosity,
                getattr(self.predictor, "progress", "auto"),
                level,
            )
            try:
                self.journal = (self.predictor.logs_dir / "events.jsonl").open(
                    "a", encoding="utf-8"
                )
            except Exception as exc:
                self.renderer.console.print(Text(f"WARNING Event journal unavailable: {exc}"))
            self.renderer.start()
            self.thread = threading.Thread(target=self.consume, name="gpse-monitor", daemon=True)
            self.thread.start()
            self.sink = logger.add(_log_sink, level="DEBUG", catch=True, format="{message}")
            configure_optuna()
            publish(
                "run",
                "started",
                models=self.models,
                config={
                    k: v
                    for k, v in vars(self.predictor).items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                },
                cli_config=getattr(self.predictor, "cli_config", None),
                result_dir=str(self.predictor.results_dir),
                manifest="run_manifest.json",
                model_statistics="models/<model>/summary_results.json",
            )
        except Exception as exc:
            self.close()
            logger.warning(f"Training monitor unavailable: {exc}")

    def consume(self):
        while True:
            try:
                event = self.events.get(timeout=0.2)
            except queue.Empty:
                if self.stop.is_set():
                    break
                if self.renderer and self.renderer.live:
                    try:
                        self.renderer.refresh()
                    except Exception:
                        pass
                continue
            except Exception:
                break
            if event is None:
                break
            if self.journal:
                try:
                    self.journal.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                    self.journal.flush()
                except Exception as exc:
                    try:
                        self.journal.close()
                    except Exception:
                        pass
                    self.journal = None
                    try:
                        self.renderer.console.print(Text(f"WARNING Event journal disabled: {exc}"))
                    except Exception:
                        pass
            try:
                self.renderer.accept(event)
            except Exception:
                # Disable Live after a terminal failure; retain the journal.
                try:
                    self.renderer.close()
                except Exception:
                    pass
                self.renderer.live = None
                self.renderer.dynamic = False

    def close(self):
        global _CLIENT, _SESSION
        try:
            logger.complete()
            if self.sink is not None:
                logger.remove(self.sink)
            if self.thread:
                try:
                    self.events.put(None, timeout=0.2)
                except Exception:
                    self.stop.set()
                self.thread.join(timeout=5)
            if self.renderer:
                self.renderer.close()
            if self.journal:
                self.journal.close()
        except Exception:
            pass
        finally:
            _SESSION = None
            _CLIENT = None


def monitored_run(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        # Model selection remains in the existing pipeline; this is a display inventory.
        models = kwargs.get("models")
        if models is None and len(args) > 3:
            models = args[3]
        monitor = TrainingMonitor(self, models if models is not None else self.available_models)
        monitor.start()
        started = time.monotonic()
        try:
            if terminal_active():
                with routed_output():
                    result = fn(self, *args, **kwargs)
            else:
                result = fn(self, *args, **kwargs)
        except BaseException as exc:
            publish(
                "run",
                failure_status(exc),
                elapsed_s=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        else:
            publish(
                "run",
                "completed",
                elapsed_s=time.monotonic() - started,
                completed_models=list(result),
                result_dir=str(self.results_dir),
            )
            return result
        finally:
            monitor.close()
            # Register logs after the journal is flushed so sizes/checksums are final.
            try:
                from gpse.output.manifest import read_run_manifest, atomic_write_json
                from gpse.output.registry import collect_artifacts

                manifest = read_run_manifest(self.paths)
                if manifest:
                    manifest["artifact_index"] = collect_artifacts(self.paths)
                    atomic_write_json(self.paths.path("run_manifest"), manifest)
            except Exception:
                pass

    return wrapped
