import logging
import datetime

class MyLogger:
    def __init__(self, filename, level=logging.INFO):
        self.filename = filename
        self.level = level
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(level)
        self.file_handler = logging.FileHandler(filename)
        self.file_handler.setLevel(level)
        self.file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(self.file_handler)

    def log(self, message, level='info'):
        if level == 'debug':
            self.logger.debug(message)
        elif level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'critical':
            self.logger.critical(message)

    def debug(self, message):
        self.log(message, 'debug')

    def info(self, message):
        self.log(message, 'info')

    def warning(self, message):
        self.log(message, 'warning')

    def error(self, message):
        self.log(message, 'error')

    def critical(self, message):
        self.log(message, 'critical')