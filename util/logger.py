import logging


class Logger:
    def __init__(self, name):
        self._handler = logging.StreamHandler()
        self._handler.setLevel(logging.INFO)
        self._handler.setFormatter(
            logging.Formatter(
                '%(asctime)s.%(msecs)06d: %(levelname).1s [%(filename)s:%(lineno)d] %(message)s',
                '%Y-%m-%d %H:%M:%S'))
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self._handler)
        self.logger.propagate = False