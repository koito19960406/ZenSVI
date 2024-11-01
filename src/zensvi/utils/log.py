import logging
from pathlib import Path


class Logger:
    """A singleton logger class that handles logging to files."""

    _loggers = {}

    def __new__(cls, log_file_path, level=logging.INFO):
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
    def _initialize_logger(cls, instance, log_file_path, level):
        """Initialize a new logger instance.

        Args:
          instance: Logger instance to initialize
          log_file_path: Path to the log file
          level: Logging level to use

        Returns:
          : None

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

    def log_info(self, message):
        """Log an info message.

        Args:
          message: Message to log

        Returns:
          : None

        """
        self.logger.info(message)

    def log_error(self, message):
        """Log an error message.

        Args:
          message: Message to log

        Returns:
          : None

        """
        self.logger.error(message)

    def log_warning(self, message):
        """Log a warning message.

        Args:
          message: Message to log

        Returns:
          : None

        """
        self.logger.warning(message)

    def log_debug(self, message):
        """Log a debug message.

        Args:
          message: Message to log

        Returns:
          : None

        """
        self.logger.debug(message)

    def log_args(self, func_name, *args, **kwargs):
        """Logs the arguments of a function call along with the function's name.

        Args:
          func_name: Name of the function being logged
          *args: Positional arguments passed to the function
          **kwargs: Keyword arguments passed to the function

        Returns:
          : None

        """
        formatted_args = []
        for i, arg in enumerate(args):
            formatted_args.append(f"arg{i}={repr(arg)}")

        for key, value in kwargs.items():
            formatted_args.append(f"{key}={repr(value)}")
        formatted_message = f"Called function '{func_name}' with: {', '.join(formatted_args)}"
        self.log_info(formatted_message)

    def log_failed_tiles(self, failed_tile_name):
        """Logs the failed tiles to a log file.

        Args:
          failed_tile_name: Name of the tile that failed processing

        Returns:
          : None

        """
        self.log_error(f"Failed to process tile: {failed_tile_name}")

    def log_failed_pids(self, failed_pid):
        """Logs the failed pids to a log file.

        Args:
          failed_pid: ID of the process that failed

        Returns:
          : None

        """
        self.log_error(f"Failed to process pid: {failed_pid}")
