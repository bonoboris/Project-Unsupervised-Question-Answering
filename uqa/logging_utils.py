import logging

FORMATTER = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
CONSOLE_FORMATER = logging.Formatter('%(message)s')


def init_root_logger(quiet=False, no_log=False, log_file="uqa.log", verbosity=logging.INFO, file_verbosity=logging.DEBUG):
    handlers = []
    if not no_log:    
        file_handler = logging.FileHandler(log_file, 'a', 'utf8')
        file_handler.setLevel(file_verbosity)
        file_handler.setFormatter(FORMATTER)
        handlers.append(file_handler)

    if not quiet:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(verbosity)
        stream_handler.setFormatter(CONSOLE_FORMATER)
        handlers.append(stream_handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers
    )

    logging.debug("Logger initialized")
