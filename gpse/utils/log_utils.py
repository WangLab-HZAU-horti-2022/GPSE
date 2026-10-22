#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loguru-based logger initialization with RichHandler console output.
"""

import datetime
import io
import logging
import os
import sys
import warnings
from pathlib import Path


try:
    from loguru import logger
    _LOGURU_AVAILABLE = True
except ImportError:
    _LOGURU_AVAILABLE = False
    logger = None


def _ensure_loguru() -> None:
    if not _LOGURU_AVAILABLE:
        raise ImportError("loguru is required for logging. Install it with: pip install loguru")


def _capture_warnings() -> None:
    """Redirect Python :mod:`warnings` output to the Loguru logger."""
    if not _LOGURU_AVAILABLE:
        return

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        logger.warning(
            f"{category.__name__}: {message} ({filename}:{lineno})"
        )

    warnings.showwarning = _showwarning


class _InterceptHandler(logging.Handler):
    """Send records from the standard :mod:`logging` module to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _capture_std_logging() -> None:
    """Redirect standard-library logging (e.g., Optuna) to the Loguru logger."""
    if not _LOGURU_AVAILABLE:
        return

    # Replace the root logger configuration so all std-logging records flow
    # through Loguru. Existing handlers are cleared to avoid duplicate output.
    logging.basicConfig(
        handlers=[_InterceptHandler()],
        level=logging.DEBUG,
        force=True,
    )


try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

def _interactive_terminal():
    return bool(sys.stderr.isatty() and not os.environ.get("CI") and os.environ.get("TERM") != "dumb")


_CONSOLE = Console(stderr=True, force_terminal=_interactive_terminal()) if _RICH_AVAILABLE else None
_CONSOLE_LEVEL = "INFO"
_TRAINING_VERBOSITY = None

def console_log_level():
    return _CONSOLE_LEVEL

def set_training_verbosity(value):
    global _TRAINING_VERBOSITY
    _TRAINING_VERBOSITY = value

def _console_filter(record):
    from gpse.utils.training_monitor import terminal_active
    if terminal_active():
        return False
    if _TRAINING_VERBOSITY in {"NORMAL", "VERBOSE"} and record["level"].no < 30:
        return bool(record["extra"].get("console_summary"))
    return True

_ANALYSIS_LOGGER = None
_ANALYSIS_LOG_FILE_PATH = None


def _add_console_handler(
    log_level: str,
    more_info: bool = False,
    style: str = "default",
) -> None:
    if not _RICH_AVAILABLE:
        fmt = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        logger.add(sys.stderr, format=fmt, level=log_level, colorize=_interactive_terminal(), enqueue=True, filter=_console_filter)
        return

    if style == "minimal":
        handler = RichHandler(
            console=_CONSOLE,
            show_time=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        fmt = "{message}"
    elif style == "detailed" or more_info:
        handler = RichHandler(
            console=_CONSOLE,
            show_time=True,
            omit_repeated_times=False,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
            log_time_format="[%X]",
        )
        fmt = "{name}:{function}:{line} - {message}"
    elif style == "plain":
        fmt = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        logger.add(sys.stderr, format=fmt, level=log_level, colorize=_interactive_terminal(), enqueue=True, filter=_console_filter)
        return
    else:
        handler = RichHandler(
            console=_CONSOLE,
            show_time=True,
            omit_repeated_times=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            log_time_format="[%X]",
        )
        fmt = "{message}"

    logger.add(handler, format=fmt, level=log_level, enqueue=True, filter=_console_filter)

def _print_run_header(
    sw: dict,
    logger_name: str,
    output_dir: str,
    project_configs: list | None = None,
) -> None:
    """Print a compact software-metadata header via Rich."""
    if not _RICH_AVAILABLE:
        return

    author = sw.get("author", "unknown")
    version = sw.get("version", "unknown")

    header_text = Text()
    header_text.append(
        f"GPSE v{version} | Genomic Prediction with Stacking Ensemble\n",
        style="bold cyan",
    )
    header_text.append(f"Author: {author}\n\n", style="dim")
    header_text.append(f"Log    : {logger_name}\n", style="white")
    header_text.append(f"Output : {output_dir}\n", style="white")

    cfg_str = ", ".join(project_configs) if project_configs else "(no project config found)"
    header_text.append(f"Config : {cfg_str}", style="white")

    _CONSOLE.print(Panel(header_text, border_style="dim", expand=False))


def setup_subprocess_logging(
    model_name: str,
    repeat_idx: int,
    log_dir: Path,
    log_level: str = "DEBUG",
):
    """Compatibility entry point: worker file plus centralized event queue."""
    from gpse.utils.training_monitor import configure_worker, get_client_config

    log_file = Path(log_dir) / "workers" / f"{model_name}_repeat_{repeat_idx + 1:03d}.log"
    configure_worker(get_client_config(), log_file, log_level=log_level)
    return logger


def collect_subprocess_logs(log_dir: Path, main_logger) -> None:
    """Keep worker files in place; replaying them would duplicate observations."""
    main_logger.debug(f"Worker logs retained in {Path(log_dir) / 'workers'}")


def shorten_path(path, max_len: int = 80) -> str:
    """
    Abbreviate a filesystem path for one-line console logs.

    Replaces the home directory with ``~`` and, if still too long, collapses
    leading components with ``…`` so the filename always stays visible.
    """
    p = str(path)
    home = str(Path.home())
    if home != "/" and p.startswith(home + "/"):
        p = "~" + p[len(home):]
    if len(p) <= max_len:
        return p
    parts = p.split("/")
    tail = parts[-1]
    # Keep as many trailing components as fit, prefix with an ellipsis
    kept = tail
    for part in reversed(parts[:-1]):
        candidate = part + "/" + kept
        if len("…/" + candidate) > max_len:
            break
        kept = candidate
    if len("…/" + kept) <= max_len:
        return "…/" + kept
    return "…" + p[-(max_len - 1):]


def log_header(title: str) -> None:
    logger.info("")
    logger.info(f"[bold cyan]{'=' * 10} {title.upper()} {'=' * 10}[/bold cyan]")


def log_section(title: str) -> None:
    logger.info(f"\n[bold blue]─── {title} ───[/bold blue]")


def log_success(msg: str) -> None:
    logger.info(f"[bold green]✔ {msg}[/bold green]")


def log_warning(msg: str) -> None:
    logger.warning(f"[bold yellow]⚠ {msg}[/bold yellow]")


def log_error(msg: str) -> None:
    logger.error(f"[bold red]✘ {msg}[/bold red]")


def log_info(msg: str) -> None:
    logger.info(msg)


def log_step(step: int, total: int, msg: str) -> None:
    logger.info(f"[bold magenta][Step {step}/{total}][/bold magenta] {msg}")


def log_config(config_dict: dict, title: str = "Configuration") -> None:
    if not _RICH_AVAILABLE:
        for k, v in config_dict.items():
            logger.info(f"{title} - {k}: {v}")
        return

    table = Table(title=title, show_header=True, header_style="bold magenta", box=None)
    table.add_column("Parameter", style="dim")
    table.add_column("Value", style="bold")

    for k, v in config_dict.items():
        table.add_row(str(k), str(v))

    console = Console(file=io.StringIO(), color_system=None, width=80)
    console.print(table)
    logger.info("\n" + console.file.getvalue())


def log_config_panel(config_dict: dict, title: str = "GPSE Configuration") -> None:
    """
    Print a Rich panel for configuration and write plain text to the log file.
    """
    if not _RICH_AVAILABLE:
        for k, v in config_dict.items():
            logger.info(f"{title} - {k}: {v}")
        return

    text = Text()
    for section, params in config_dict.items():
        text.append(f"\n[ {section} ]\n", style="bold cyan")
        if isinstance(params, dict):
            for k, v in params.items():
                text.append(f"  • {k: <20}: ", style="dim")
                text.append(f"{v}\n", style="white")
        else:
            text.append(f"  {params}\n", style="white")

    _CONSOLE.print(
        Panel(text, title=f"[bold]{title}[/bold]", border_style="cyan", expand=False)
    )

    logger.debug(f"=== {title} ===")
    for section, params in config_dict.items():
        if isinstance(params, dict):
            for k, v in params.items():
                logger.debug(f"[{section}] {k}: {v}")
        else:
            logger.debug(f"[{section}] {params}")


def get_loading_status(msg: str):
    """
    Return a Rich status context manager, or a logging fallback.
    """
    if _RICH_AVAILABLE:
        return _CONSOLE.status(f"[bold green]{msg}[/bold green]", spinner="dots")

    class DummyStatus:
        def __enter__(self):
            logger.info(msg)

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    return DummyStatus()


# ── Public API ─────────────────────────────────────────────────────────────

def logger_init(
    logger_name: str = None,
    log_level: str = None,
    more_info: bool = False,
    style: str = "default",
) -> "logger":
    """
    Configure Loguru logger with Rich-styled console output.
    """
    _ensure_loguru()

    from .configuration import load_default_config, get_loaded_project_configs
    default = load_default_config()
    cfg_logs = default.get("logs", {})

    global _CONSOLE_LEVEL
    log_level = log_level or _CONSOLE_LEVEL or cfg_logs.get("log_level", "INFO")
    _CONSOLE_LEVEL = log_level
    more_info = more_info if more_info is not None else cfg_logs.get("more_info", False)
    style = style or cfg_logs.get("style", "default")

    logger.remove()

    _add_console_handler(log_level, more_info=more_info, style=style)
    _capture_warnings()
    _capture_std_logging()

    # File handler — plain text, no colours, enqueue for multi-process safety
    if logger_name:
        file_fmt = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {extra} | {message}"
        )
        try:
            Path(logger_name).parent.mkdir(parents=True, exist_ok=True)
            logger.add(
                logger_name,
                format=file_fmt,
                level="DEBUG",
                colorize=False,
                enqueue=True,
                backtrace=True,
                diagnose=False,
            )
        except Exception as exc:
            logger.warning(f"Detailed log file unavailable: {exc}")

    # Print project config info as the first log messages
    project_configs = get_loaded_project_configs()
    if project_configs:
        for cfg_path in project_configs:
            logger.debug(f"GPSE config loaded: {cfg_path}")
    else:
        logger.debug("GPSE config: none found (using package defaults)")

    return logger


def logger_generator(
    output_dir: str,
    log_level: str = "INFO",
    more_info: bool = False,
    style: str = "default",
) -> tuple:
    """
    Create a logger, print software metadata via Rich, and return ``(logger, output_dir)``.
    """
    _ensure_loguru()

    from .configuration import (
        load_software_config,
        load_default_config,
        get_loaded_project_configs,
    )

    software = load_software_config()
    default = load_default_config()

    sw = software.get("software", {})
    label = default.get("logs", {}).get("Label", "GPSE")

    times = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger_name = f"{output_dir}/{label}_{times}.log"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Detect project-level config files
    project_configs = get_loaded_project_configs()

    # Pretty console header
    _print_run_header(sw, logger_name, output_dir, project_configs=project_configs)

    # Init logger (also prints project config info)
    logger = logger_init(
        logger_name,
        log_level=log_level,
        more_info=more_info,
        style=style,
    )

    # Also persist metadata to the log file
    logger.info(f"GPSE Author : {sw.get('author', 'unknown')}")
    logger.info(f"GPSE Version : {sw.get('version', 'unknown')}")
    logger.info(f"GPSE Email : {sw.get('email', '')}")
    logger.info(f"Logger initialized, log file : {logger_name}")
    logger.info(f"GPSE Analysis Result : {output_dir}")
    logger.debug(f"Software Full config : {software}")
    logger.debug(f"Default Full config : {default}")

    return logger, output_dir


def setup_analysis_logging(
    log_dir="logs",
    log_file_prefix="analysis",
    max_file_size="100 MB",
    console_level="INFO",
    file_level="DEBUG",
    style="default",
):
    """
    Setup logging for analysis scripts with RichHandler console output.
    """
    _ensure_loguru()

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = log_dir_path / f"{log_file_prefix}_{timestamp}.log"

    logger.remove()
    logger.add(
        log_file_path,
        rotation=max_file_size,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        level=file_level,
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )
    _add_console_handler(console_level, style=style)
    _capture_warnings()
    _capture_std_logging()

    logger.info(
        f"[bold green]{log_file_prefix.capitalize()} Script Initialized[/bold green] "
        f"(Style: {style})"
    )
    logger.info(f"Log file: {log_file_path}")

    return logger, log_file_path


def get_logger():
    return logger


def initialize_analysis_logger(**kwargs):
    global _ANALYSIS_LOGGER, _ANALYSIS_LOG_FILE_PATH

    if _ANALYSIS_LOGGER is None:
        _ANALYSIS_LOGGER, _ANALYSIS_LOG_FILE_PATH = setup_analysis_logging(**kwargs)

    return _ANALYSIS_LOGGER, _ANALYSIS_LOG_FILE_PATH


def get_analysis_logger():
    global _ANALYSIS_LOGGER

    if _ANALYSIS_LOGGER is None:
        _ANALYSIS_LOGGER, _ = initialize_analysis_logger()

    return _ANALYSIS_LOGGER


def get_analysis_log_file_path():
    return _ANALYSIS_LOG_FILE_PATH
