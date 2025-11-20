import logging
import os
from datetime import datetime

def setup_file_logger(log_file="kv_cache.log"):
    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.INFO)

    logger.handlers.clear()   # 清除控制台默认 handler

    # 创建目录
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 文件输出 handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(file_handler)

    logging.info("===== NEW RUN @ %s =====", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return logger



