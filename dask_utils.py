from dask.distributed import Client, LocalCluster, wait, progress
import dask.config
import logging
import sys


def initialize_dask(w, t, mem, disk):
    virtual_mem = (mem + disk) / w
    spill = (mem / w) / virtual_mem

    dask.config.set({
        'distributed.worker.memory.target': None,
        'distributed.worker.memory.spill': spill,
        'distributed.worker.memory.pause': spill + 0.1,
        # 'distributed.worker.memory.terminate': False
    })
    cluster = LocalCluster(
        n_workers=w,
        threads_per_worker=t,
        memory_limit=f'{virtual_mem}GB',
        processes=True,
        silence_logs=logging.ERROR,
    )
    client = Client(cluster)

    return client


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def print_mem(locals):
    for name, size in sorted(
        ((name, sys.getsizeof(value)) for name, value in locals.items()),
            key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
