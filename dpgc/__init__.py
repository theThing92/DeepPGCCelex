import logging.config
import torch

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": "%(asctime)s - %(name)s - %(module)s - %(lineno)d - %(levelname)s - %(message)s"}},
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "dpgc": {"handlers": ["console"], "level": "INFO", "propagate": True}
        },
    }
)

logger = logging.getLogger("dpgc")


def set_device():
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        logger.info("GPU is available.")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available, CPU used.")

    return device

device = set_device()