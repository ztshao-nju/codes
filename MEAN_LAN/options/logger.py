import logging
import time
import os


class logFrame:

    def getlogger(self, log_path=None):
        self.logger = logging.getLogger("logger")
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)

            # 控制台输出
            control = logging.StreamHandler()
            control_formatter = logging.Formatter(fmt="%(asctime)s,%(filename)s,行号:%(lineno)d,"
                                                      "描述:%(message)s", datefmt="%Y/%m/%d %H:%M:%S")
            self.add_output(self.logger, control, control_formatter, logging.INFO)
            # filename=log_path + "./log/{}_log.txt".format(time.strftime("%Y_%m_%d %H_%M_%S", time.localtime()))

            # 文件输出
            if log_path != None:
                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                file = logging.FileHandler(
                    filename=log_path,
                    encoding="utf8")
                file_formatter = logging.Formatter(fmt="%(asctime)s,%(filename)s,行号:%(lineno)d,"
                                                       "描述:%(message)s", datefmt="%Y/%m/%d %H:%M:%S")
                self.add_output(self.logger, file, file_formatter, logging.DEBUG)

        return self.logger

    # 添加一个输出
    def add_output(self, logger, handler, formatter, level):
        logger.addHandler(handler)
        handler.setFormatter(formatter)
        handler.setLevel(level)
