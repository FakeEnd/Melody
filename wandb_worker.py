import atexit
import threading
from argparse import Namespace
from queue import Queue

from loguru import logger
import wandb

class WandbWorker:
    def __init__(self, args: Namespace = Namespace(use_swanlab=False, project='Melody')):
        self.wandb_inited = False
        self.log_queue = Queue()
        self.args = args
        self.wandb = wandb
        if getattr(args, "use_swanlab", False):
            try:
                import swanlab
            except Exception as e:
                raise RuntimeError("use_swanlab=True but swanlab is not installed") from e
            self.wandb = swanlab

        self.project_name = getattr(args, "project", "Melody")
        self.lab_name = getattr(args, 'lab_name', 'default_lab')
        self.logging_thread = threading.Thread(target=self.wandb_logger_worker, args=(self.log_queue,), daemon=True)
        self.logging_thread.start()
        atexit.register(self.shutdown_wandb_logger)

    # noinspection PyTypeChecker
    def wandb_logger_worker(self, q: Queue):
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break
            try:
                if not self.wandb_inited:
                    self.wandb.init(project=self.project_name, name=self.lab_name, config=self.args)
                    self.wandb_inited = True

                log_dict = item.get('log_dict', {})
                kwargs = item.get('kwargs', {})
                self.wandb.log(log_dict, **kwargs)

            except Exception as e:
                logger.error(f"Wandb Thread Error: {e}", exc_info=True)
            finally:
                q.task_done()

    def shutdown_wandb_logger(self):
        if self.wandb_inited:
            self.log_queue.join()
            self.wandb.finish()
            print("Wandb run finished.")

        self.log_queue.put(None)
        self.logging_thread.join(timeout=10)

    def log(self, log_dict, i):
        self.log_queue.put({'log_dict': log_dict, 'kwargs': {'step': i}})
