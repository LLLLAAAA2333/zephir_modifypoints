from contextlib import contextmanager
import time

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.cyk_logging as logging

@contextmanager
def logged_operation(display_context="operation", ignore_error=False,):
    """
    带日志记录的上下文管理器
    
    :param context: 操作名称
    :param context_getter: 用于获取上下文对象的函数或参数名
    """
    start_time = time.time()
    try:
        logging.info(f"> Start {display_context}")
        yield  # 在此处执行操作
        elapsed = time.time() - start_time
        logging.info(f"\033[92m[Time Cost: {elapsed:.3f}s] Finish {display_context}\033[0m")
    except Exception as e:
        elapsed = time.time() - start_time
        logging.exception(f"\033[91m[Time Cost: {elapsed:.3f}s] Error occurred while processing {display_context}\033[0m")
        if not ignore_error:
            raise  # 重新抛出异常，保持原始堆栈跟踪

# 使用示例
if __name__ == "__main__":
    file_path = "example.txt"  # 假设这是你的文件路径
    with logged_operation(display_context=f"File Processing({file_path})", ignore_error=False):
        # 这里是你的实际处理代码
        print(f"Processing file: {file_path}")
        # 故意制造一个错误
        1/0
