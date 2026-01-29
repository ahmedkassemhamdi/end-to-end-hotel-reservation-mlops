import logging
import os
from datetime import datetime


LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOGS_FILE = os.path.join(LOGS_DIR, f'LOGS_{datetime.now().strftime("%Y-%m-%d")}.log')

logging.basicConfig(
                    filename=LOGS_FILE,
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s"
                    )

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

