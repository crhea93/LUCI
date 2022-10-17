"""
    This file contains the logger functionality for LUCI. We use the standard logging functionality in python.
"""
import logging 
from datetime import datetime


class LUCILog():
    def __init__(self):
        """
            Initialize LUCI Logger
        """
        self.logger = logging.getLogger('LUCI.log')
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        # Initial comment
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.logger.debug('Starting LUCI Log at %s'% dt_string)
    
    def warn(self, warning_message):
        """
            Add warning message to the logger. At the moment, it doesn't do anything other than the normal warn log message.
        """
        self.logger.warn(warning_message)

    def info(self, info_message):
        """
            Add information message to the logger
        """
        self.logger.info(info_message)
