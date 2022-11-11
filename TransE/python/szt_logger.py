import logging

# 创建一个日志器logger
import time


class logFrame:

    def getlogger(self, log_path):
        self.logger = logging.getLogger("logger")
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            con = logging.StreamHandler()
            con_formatter = logging.Formatter(fmt="%(asctime)s,%(filename)s,行号:%(lineno)d，级别:%(levelname)s，"
                                              "描述:%(message)s", datefmt="%Y/%m/%d %H:%M:%S")
            # filename=log_path + "./log/{}_log.txt".format(time.strftime("%Y_%m_%d %H_%M_%S", time.localtime()))
            out = logging.FileHandler(
                filename=log_path,
                encoding="utf8")
            out_formatter = logging.Formatter(fmt="%(asctime)s,%(filename)s,行号:%(lineno)d，级别:%(levelname)s，"
                                                "描述:%(message)s", datefmt="%Y/%m/%d %H:%M:%S")

            self.logger.addHandler(con)
            con.setFormatter(con_formatter)
            con.setLevel(logging.DEBUG)

            self.logger.addHandler(out)
            out.setFormatter(out_formatter)
            out.setLevel(logging.INFO)

        return self.logger


# if __name__ == '__main__':
#     log = logFrame()
#     logger = log.getlogger()
#     # 输出日志
#     try:
#         b = int(input("请输入一个除数:"))
#         num = 4 / b
#         logger.info(f"4/{b}={num},计算完成")
#         logger.debug("这条对了")
#     except Exception as error:
#         print(str(error))
#         logger.error(str(error))