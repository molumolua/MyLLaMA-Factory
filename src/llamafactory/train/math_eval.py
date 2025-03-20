from .math_evaluation.grader import math_equal
from .math_evaluation.parser import extract_answer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import multiprocessing
def reject_sample(response,solution,timeout=True):
    '''
    input为两个完整答案的string
    '''
    temp_ans = extract_answer(response,data_name="math")
    ans = extract_answer(solution,data_name="math")
    return math_equal(temp_ans,ans,timeout=timeout)

def process_reject_sample(problem, section,response, logger,timeout=10):
    """
    在单独的进程中执行reject_sample相关的操作，
    如果超过设定的超时时间（默认为10秒），直接杀死子进程并返回False
    """

    # 这个内部函数里放需要执行的逻辑，比如调用 reject_sample 及其子函数
    def _worker_func(return_dict, problem, response):
        try:
            if problem and problem.get(section) and response:
                # 如果你还需要传 logger 或其它参数，也可一并加入
                result = reject_sample(response, problem[section],timeout=False)
                return_dict['result'] = result
            else:
                logger.warning("Missing data for reject sample.")
                return_dict['result'] = False
        except Exception as e:
            logger.error(f"Error in reject_sample: {e}")
            return_dict['result'] = False

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # 创建子进程
    p = multiprocessing.Process(
        target=_worker_func,
        args=(return_dict, problem, response)
    )
    if timeout ==0:
        try:
            if problem and problem.get(section) and response:
                result = reject_sample(response, problem[section],timeout=False)
                return result
            else:
                logger.warning("Missing data for reject sample.")
                return False
        except Exception as e:
            logger.error(f"Error in reject_sample: {e}")
            return False
    else:
        try:
            # 启动子进程
            p.start()
            # 设置最大等待时间20秒
            p.join(timeout=timeout)

            # 如果子进程还存活，说明超时
            if p.is_alive():
                logger.warning(problem)
                logger.warning(response)
                logger.warning(f"process_reject_sample exceeded the timeout limit of {timeout} seconds.")
                p.terminate()   # 终止子进程
                p.join()        # 回收子进程
                return False

            # 如果没超时就获取执行结果
            result = return_dict.get('result', False)
            return result

        except Exception as e:
            logger.error(f"Exception in process_reject_sample: {e}")
            if p.is_alive():
                p.terminate()
                p.join()
            return False

