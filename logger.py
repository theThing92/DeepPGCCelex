import logging.config

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
            "pgc": {"handlers": ["console"], "level": "INFO", "propagate": True}
        },
    }
)

logger = logging.getLogger("pgc")
