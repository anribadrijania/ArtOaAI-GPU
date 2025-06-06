import logging
import os
from datetime import datetime
import pytz

# Retrieve logging mode and timezone from environment variables
LOG_MODE = os.getenv("LOG_MODE", "information")
TIMEZONE = os.getenv("TIMEZONE", "Asia/Tbilisi")


def get_current_time():
    """
    Get the current time formatted as a string based on the configured timezone.

    :return: Formatted timestamp string.
    """
    timezone = pytz.timezone(TIMEZONE)
    return datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")


# Logger Configuration
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s: %(message)s | Timestamp: [%(asctime)s]")

# Console Handler (Only in Debug Mode)
if LOG_MODE == "debug":
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)  # Log everything to console
    logger.addHandler(console_handler)

# File Handler (Log Everything to File)
file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)  # Log everything to the file
logger.addHandler(file_handler)


# Logging Functions
def log_info(message):
    """ Log an informational message. """
    logger.info(message)


def log_warning(message):
    """ Log a warning message. """
    logger.warning(message)


def log_error(message):
    """ Log an error message. """
    logger.error(message)


def log_debug(message):
    """ Log a debug message. """
    logger.debug(message)
