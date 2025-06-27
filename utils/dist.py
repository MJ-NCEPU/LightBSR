import os
import pickle
import subprocess
import torch
import torch.distributed as dist


def get_rank():
    return int(os.environ.get('SLURM_PROCID', 0))

def get_local_rank():
    return get_rank() % torch.cuda.device_count()

def get_world_size():
    return int(os.environ.get('SLURM_NTASKS', 1))


def dist_init(method='slurm', port='5671'):
    if method == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)

        world_size = get_world_size()
        rank = get_rank()

        addr = subprocess.getoutput(
            "scontrol show hostname {} | head -n1".format(os.environ["SLURM_NODELIST"])
        )
        os.environ["MASTER_PORT"] = port
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
    else:
        raise NotImplementedError

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )

    return rank, world_size


def _serialize_to_tensor(data, group=None):
    device = torch.cuda.current_device()

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        import logging
        logger = logging.getLogger('global')
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def broadcast_object(obj, group=None):
    """make suare obj is picklable
    """
    if get_world_size() == 1:
        return obj

    serialized_tensor = _serialize_to_tensor(obj).cuda()
    numel = torch.IntTensor([serialized_tensor.numel()]).cuda()
    dist.broadcast(numel, 0)
    # serialized_tensor from storage is not resizable
    serialized_tensor = serialized_tensor.clone()
    serialized_tensor.resize_(numel)
    dist.broadcast(serialized_tensor, 0)
    serialized_bytes = serialized_tensor.cpu().numpy().tobytes()
    deserialized_obj = pickle.loads(serialized_bytes)
    return deserialized_obj
