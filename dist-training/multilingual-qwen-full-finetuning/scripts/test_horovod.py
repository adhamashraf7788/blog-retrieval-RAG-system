import horovod.torch as hvd

hvd.init()

print(f"I am rank {hvd.rank()} of {hvd.size()}, local rank {hvd.local_rank()}")
