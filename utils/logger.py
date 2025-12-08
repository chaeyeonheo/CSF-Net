# utils/logger.py

import logging
import os
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO, rank=0, world_size=1):
    """
    분산 환경을 고려한 로거 설정. rank 0 프로세스에서만 파일에 기록하고 콘솔에 출력.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False # 중복 로깅 방지

    formatter = logging.Formatter('%(asctime)s - %(name)s - R%(rank)d - %(levelname)s - %(message)s')

    if rank == 0:
        # 파일 핸들러 (디렉토리 생성)
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        # 다른 rank에서는 아무것도 출력하지 않도록 NullHandler 사용
        logger.addHandler(logging.NullHandler())
        
    # 모든 핸들러에 rank 정보 추가 (formatter에서 사용 위함)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) or isinstance(handler, logging.StreamHandler):
            class RankFilter(logging.Filter):
                def filter(self, record):
                    record.rank = rank
                    return True
            handler.addFilter(RankFilter())
            
    return logger

# 예시:
# if __name__ == '__main__':
#     # 단일 프로세스 테스트
#     logger_rank0 = setup_logger("my_app_rank0", "app_rank0.log", rank=0)
#     logger_rank0.info("This is an info message from rank 0.")
#     logger_rank0.warning("This is a warning message from rank 0.")

#     # 다른 rank 테스트 (실제 DDP 환경에서는 자동으로 rank 값 할당됨)
#     logger_rank1 = setup_logger("my_app_rank1", "app_rank1.log", rank=1, world_size=2) # 파일은 생성 안됨
#     logger_rank1.info("This message from rank 1 should not appear in console or file.")