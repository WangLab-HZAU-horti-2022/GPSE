"""Cancellation of GPSE-owned process trees, including native/joblib children."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import threading
import time


def process_identity(pid):
    """Linux PID identity: start time prevents signalling a reused PID."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(pid), fields[19], int(fields[2]), fields[0], int(fields[1]))
    except (OSError, ValueError, IndexError):
        return None


def owned_processes(roots):
    """Snapshot only the descendants of explicitly owned root PIDs."""
    identities = {}
    if not Path("/proc").exists():
        return identities
    table = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            info = process_identity(int(path.name))
            if info:
                table[info[0]] = info
    pending = list(roots)
    visited = set()
    while pending:
        pid = pending.pop()
        if pid in visited:
            continue
        visited.add(pid)
        if pid == os.getpid():
            pending.extend(p for p, entry in table.items() if entry[4] == pid)
            continue
        info = table.get(pid)
        if info:
            identities[pid] = info
            pending.extend(p for p, entry in table.items() if entry[4] == pid)
    return identities


def _alive(identity):
    current = process_identity(identity[0])
    return current is not None and current[1] == identity[1] and current[3] != "Z"


@contextmanager
def uninterrupted_cleanup():
    """A second Ctrl+C must not interrupt the kill/reap phase."""
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (
            signal.SIGINT,
            signal.SIGTERM,
            *([signal.SIGHUP] if hasattr(signal, "SIGHUP") else []),
        ):
            previous[sig] = signal.signal(sig, signal.SIG_IGN)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def terminate_owned(identities, grace=2.0):
    """TERM then KILL owned descendants and isolated groups; never the CLI group."""
    if not identities:
        return
    with uninterrupted_cleanup():
        own_group = os.getpgrp() if hasattr(os, "getpgrp") else None
        for sig in (signal.SIGTERM, signal.SIGKILL):
            live = [info for info in identities.values() if _alive(info)]
            groups = {info[2] for info in live if info[2] != own_group}
            # Only groups whose leader is itself an owned process are ours.
            groups &= set(identities)
            for group in groups:
                try:
                    os.killpg(group, sig)
                except (OSError, AttributeError):
                    pass
            for info in reversed(live):
                if _alive(info):
                    try:
                        os.kill(info[0], sig)
                    except OSError:
                        pass
            deadline = time.monotonic() + (grace if sig == signal.SIGTERM else 1.0)
            while time.monotonic() < deadline and any(_alive(i) for i in identities.values()):
                time.sleep(0.025)


def initialize_pool_worker(initializer, initargs):
    """Each worker gets its own group, isolating cancellation from the user's shell."""
    if hasattr(os, "setsid"):
        os.setsid()
    if initializer:
        initializer(*initargs)


def stop_subprocess(process, identities=None):
    identities = identities if identities is not None else owned_processes([process.pid])
    with uninterrupted_cleanup():
        if identities:
            terminate_owned(identities)
        elif process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=2)
        except Exception:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)


def interruptible_entry(fn):
    """Route scheduler SIGTERM through the same cleanup path as Ctrl+C."""
    from functools import wraps

    @wraps(fn)
    def wrapped(*args, **kwargs):
        previous = {}
        if threading.current_thread() is threading.main_thread():

            def stop(signum, frame):
                raise KeyboardInterrupt

            for sig in (signal.SIGTERM, *([signal.SIGHUP] if hasattr(signal, "SIGHUP") else [])):
                previous[sig] = signal.signal(sig, stop)
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            return 130
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    return wrapped


def run_subprocess(args, *, check=False, capture_output=False, **kwargs):
    """subprocess.run semantics with an isolated, cancellable process tree."""
    import subprocess

    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process = subprocess.Popen(args, start_new_session=(os.name == "posix"), **kwargs)
    try:
        stdout, stderr = process.communicate()
    except BaseException:
        stop_subprocess(process)
        raise
    result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, args, output=stdout, stderr=stderr)
    return result
