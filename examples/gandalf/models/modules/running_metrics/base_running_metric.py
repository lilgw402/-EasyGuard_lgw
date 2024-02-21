import torch


class BaseRunningMetric:
    def batch_eval(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    @staticmethod
    def unwrap_to_tensors(*tensors: torch.Tensor, move_to_cpu=False):
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures that you're using tensors directly, and they can be on
        the CPU.
        """
        return (
            (x.detach().cpu() if move_to_cpu else x.detach()) if isinstance(x, torch.Tensor) else x for x in tensors
        )
