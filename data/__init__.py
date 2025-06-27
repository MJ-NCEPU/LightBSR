from importlib import import_module
from dataloader import MSDataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import os


class Data:
    def __init__(self, args):
        if args.n_GPUs > 1:
            self.local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.rank = torch.distributed.get_rank()

        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())  ## load the right dataset loader module
            trainset = getattr(module_train, args.data_train)(
                args)  ## load the dataset, args.data_train is the  dataset name
            if args.n_GPUs > 1:
                self.sampler = DistributedSampler(trainset)
                self.loader_train = MSDataLoader(
                    args,
                    trainset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    pin_memory=False,
                    sampler=self.sampler
                )
            else:
                self.loader_train = MSDataLoader(
                    args,
                    trainset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    pin_memory=False,
                )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, name=args.data_test, train=False)
        else:
            module_test = import_module('data.' + args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)

        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=False
        )
