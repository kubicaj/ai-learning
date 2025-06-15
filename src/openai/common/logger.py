import logging
import sys


def init_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger()
