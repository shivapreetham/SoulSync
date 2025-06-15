# Provides a centralized logging system.
# Logs to a file for debugging and monitoring.
# Can be extended to log to console or other outputs.
import logging
import os

def setup_logger(name: str) -> logging.Logger:
    """Set up a logger with file and console output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler('soulsync.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    return logger