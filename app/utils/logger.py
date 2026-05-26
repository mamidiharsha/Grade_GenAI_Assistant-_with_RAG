"""
Structured logging configuration for the application.
Provides consistent log formatting across all modules.
"""

import logging
import sys
from datetime import datetime


def setup_logger(name: str = "genai_assistant") -> logging.Logger:
    """
    Set up and return a configured logger instance.
    
    Args:
        name: Logger name for identification in log output.
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler with structured formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Structured format: timestamp | level | module | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Global logger instance
logger = setup_logger()
