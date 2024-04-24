# utils/logger.py
import logging

# Configure the root logger
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create a global logger instance
logger = logging.getLogger(__name__)
