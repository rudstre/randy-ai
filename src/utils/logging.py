"""
Logging utilities for the interview system.
"""
import os
import logging


def setup_logging(log_file_path: str) -> str:
    """
    Set up logging to file with minimal console output.
    
    Args:
        log_file_path: Full path to the log file
        
    Returns:
        Path to the log file
    """
    # Extract directory from log file path and create it
    workdir = os.path.dirname(log_file_path)
    if workdir:  # Only create directory if path has a directory component
        os.makedirs(workdir, exist_ok=True)
    
    log_file = log_file_path
    
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler for minimal output only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)  # Only show critical messages on console
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file
