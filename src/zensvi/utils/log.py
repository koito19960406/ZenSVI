import logging
from pathlib import Path

class Logger:
    _loggers = {}

    def __new__(cls, log_file_path, level=logging.INFO):
        if log_file_path not in cls._loggers:
            cls._loggers[log_file_path] = super(Logger, cls).__new__(cls)
            cls._initialize_logger(cls._loggers[log_file_path], log_file_path, level)
        return cls._loggers[log_file_path]

    @staticmethod
    def _initialize_logger(self, log_file_path, level):
        self.logger = logging.getLogger(log_file_path)
        if not self.logger.handlers:
            self.logger.setLevel(level)

            # Ensure log directory exists
            Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

            # Create file handler which logs even debug messages
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(level)

            # Create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)

            # Add the handler to the logger
            self.logger.addHandler(fh)

        # Function argument names mapping
        self.function_arg_names = {
            "MLYDownloader download_svi": ["dir_output", "path_pid", "lat", "lon", "input_csv_file", "input_shp_file", "input_place_name", "buffer", "update_pids", "resolution", "cropped", "batch_size", "start_date", "end_date", "metadata_only", "use_cache"],
            "GSVDownloader download_svi": ["dir_output", "path_pid", "zoom", "cropped", "full", "lat", "lon", "input_csv_file", "input_shp_file", "input_place_name", "id_columns", "buffer", "augment_metadata", "batch_size", "update_pids", "start_date", "end_date", "metadata_only"]
            # Add other functions and their arguments here
        }
        
    def log_info(self, message):
        self.logger.info(message)

    def log_error(self, message):
        self.logger.error(message)

    def log_warning(self, message):
        self.logger.warning(message)

    def log_debug(self, message):
        self.logger.debug(message)

    def log_args(self, function_name, *args, **kwargs):
        """
        Logs the arguments of a function call along with the function's name.
        """
        if function_name in self.function_arg_names:
            arg_names = self.function_arg_names[function_name]
            args_str = ", ".join(f"{name}: {arg}" for name, arg in zip(arg_names, args))
        else:
            args_str = ", ".join(f"{arg}" for arg in args)
        
        kwargs_str = ", ".join(f"{key} = {value}" for key, value in kwargs.items())

        formatted_message = f"Called function '{function_name}' with: "
        if args_str:
            formatted_message += f"args: {args_str}"
        if kwargs_str:
            if args_str:
                formatted_message += "; "
            formatted_message += f"kwargs: {kwargs_str}"
        self.log_info(formatted_message)

    
    def log_failed_tiles(self, failed_tile_name):
        """
        Logs the failed tiles to a log file.
        """
        self.log_error(f"Failed to process tile: {failed_tile_name}")
            
    def log_failed_pids(self, failed_pid):
        """
        Logs the failed pids to a log file.
        """
        self.log_error(f"Failed to process pid: {failed_pid}")