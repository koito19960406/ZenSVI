import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union


class Logger:
    """A singleton logger class that handles logging to files."""

    _loggers: Dict[str, "Logger"] = {}

    def __new__(cls, log_file_path: Union[str, Path], level: int = logging.INFO) -> "Logger":
        """Create or retrieve a logger instance.

        This method implements the singleton pattern, ensuring only one logger exists
        per log file path.

        Args:
            log_file_path: Path to the log file
            level: Logging level to use, defaults to logging.INFO

        Returns:
            Logger: A logger instance for the specified log file
        """
        if log_file_path not in cls._loggers:
            logger = super(Logger, cls).__new__(cls)
            cls._initialize_logger(logger, log_file_path, level)
            cls._loggers[log_file_path] = logger
        return cls._loggers[log_file_path]

    @classmethod
    def _initialize_logger(cls, instance: "Logger", log_file_path: Union[str, Path], level: int) -> None:
        """Initialize a new logger instance.

        Args:
            instance: Logger instance to initialize
            log_file_path: Path to the log file
            level: Logging level to use
        """
        instance.logger = logging.getLogger(str(log_file_path))
        if not instance.logger.handlers:
            instance.logger.setLevel(level)
            # Ensure log directory exists
            Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
            # Create file handler which logs even debug messages
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(level)
            # Create formatter and add it to the handlers
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fh.setFormatter(formatter)
            # Add the handler to the logger
            instance.logger.addHandler(fh)

    def log_info(self, message: str) -> None:
        """Log an info message.

        Args:
            message: Message to log
        """
        self.logger.info(message)

    def log_error(self, message: str) -> None:
        """Log an error message.

        Args:
            message: Message to log
        """
        self.logger.error(message)

    def log_warning(self, message: str) -> None:
        """Log a warning message.

        Args:
            message: Message to log
        """
        self.logger.warning(message)

    def log_debug(self, message: str) -> None:
        """Log a debug message.

        Args:
            message: Message to log
        """
        self.logger.debug(message)

    def log_args(self, func_name: str, *args: Any, **kwargs: Any) -> None:
        """Logs the arguments of a function call along with the function's name.

        Args:
            func_name: Name of the function being logged
            *args: Positional arguments passed to the function
            **kwargs: Keyword arguments passed to the function
        """
        formatted_args = []
        for i, arg in enumerate(args):
            formatted_args.append(f"arg{i}={repr(arg)}")

        for key, value in kwargs.items():
            formatted_args.append(f"{key}={repr(value)}")
        formatted_message = f"Called function '{func_name}' with: {', '.join(formatted_args)}"
        self.log_info(formatted_message)

    def log_failed_tile(self, failed_tile_name: str) -> None:
        """Logs the failed tiles to a log file.

        Args:
            failed_tile_name: Name of the tile that failed processing
        """
        self.log_error(f"Failed to process tile: {failed_tile_name}")

    def log_failed_pid(self, failed_pid: Union[str, int]) -> None:
        """Log failed panorama ID.

        Args:
            failed_pid: The failed panorama ID.
        """
        self.log_error(f"Failed to process pid: {failed_pid}")


def verbosity_tqdm(
    iterable: Iterable,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    disable: bool = False,
    verbosity: int = 1,
    level: int = 1,
    **kwargs: Any,
) -> Iterable:
    """A wrapper around tqdm that respects verbosity levels.

    Args:
        iterable: Iterable to decorate with a progressbar
        desc: Description to show in the progressbar
        total: Total number of items in the iterable
        disable: Whether to disable the progressbar
        verbosity: Current verbosity level (0 = no output, 1 = outer loops only, 2 = all loops)
        level: The nested level of this loop (1 = outermost, 2 = inner, etc.)
        **kwargs: Additional arguments to pass to tqdm

    Returns:
        tqdm: A tqdm progress bar or a simple iterable if disabled
    """
    from tqdm import tqdm

    # If verbosity is 0, or level is greater than verbosity, disable the progress bar
    should_disable = disable or verbosity == 0 or level > verbosity

    if should_disable:
        return iterable
    else:
        return tqdm(iterable, desc=desc, total=total, **kwargs)
