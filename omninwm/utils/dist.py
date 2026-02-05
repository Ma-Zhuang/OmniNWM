import os
import torch.distributed as dist


def is_distributed() -> bool:
    """
    Check if the code is running in a distributed setting.

    Returns:
        bool: True if running in a distributed setting, False otherwise
    """
    return os.environ.get("WORLD_SIZE", None) is not None


def is_main_process() -> bool:
    """
    Check if the current process is the main process.

    Returns:
        bool: True if the current process is the main process, False otherwise.
    """
    return not is_distributed() or dist.get_rank() == 0


def get_world_size() -> int:
    """
    Get the number of processes in the distributed setting.

    Returns:
        int: The number of processes.
    """
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1