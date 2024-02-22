from mpi4py import MPI
from enum import Enum, auto

import config_parser


class MPI_TAGS(Enum):
    # First job request from a worker. Can be issued at the beginning of
    # the iteration or after a best feature is found and received.
    WORKER_FIRST_JOB = auto()

    # Training results sent by a worker, also requesting a new job.
    WORKER_JOB_RESULT = auto()

    # New feature to be evaluated by a worker, send by the manager
    MANAGER_NEW_JOB = auto()

    # Best feature from the current f_it, sent by the manager. This message
    # implies that more features should be tested, thus, should be followed
    # by a WORKER_FIRST_JOB message from all workers.
    MANAGER_SELECTED_FEATURE = auto()

    # Best features set of the current iteration. This means that the feature
    # selection stage is done for the current iteration.
    MANAGER_BEST_FEATURES = auto()

    # These two messages are related to the case on which propagation reached
    # a point that only one well is currently propagating. This means that
    # leave-one-well-out cross-validation won't work. On this case, the whole
    # execution should be halted.
    # A worker that identifies this case sends WORKER_ABORT_PROP. The manager
    # then replies all remaining workers with MANAGER_ABORT_PROP.
    WORKER_ABORT_PROP = auto()
    MANAGER_ABORT_PROP = auto()


def _get_local_node_comm(comm):
    '''
    Return a split communicator for processes within the same node.
    For instance, running 4 nodes and 8 processes would yield 4 different
    communicators with 2 processes.

    This is required for using distributed hdf5. The MPI_File_open routine
    needs the same file for the input MPI communicator, and for running
    locally the files although having the same name and path, are different.
    '''

    # Get all node names
    local_host_name = MPI.Get_processor_name()
    node_names = comm.allgather(local_host_name)

    # Get unique sorted
    node_names = list(set(node_names))
    node_names.sort()

    rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    manager_rank = mpi_size - 1

    # Perform split
    if rank == manager_rank:
        # Manager should be at a separate communicator since it won't be
        # opening any h5 file.
        local_color = len(local_host_name)
    else:
        local_color = node_names.index(local_host_name)
    local_comm = comm.Split(local_color)

    return local_comm


def _should_update_local(mpi_size, rank, comm):
    '''
    Check whether the current process should update the local
    h5 porosity file.
    Only one process per node should do this.
    Although multiple updates works on h5, it is inefficient.
    Manager process don't perform propagation.
    '''

    ret = False
    assigned_nodes = []
    for r in range(mpi_size - 1):
        # Get node name of rank r
        if r == rank:
            cur_node = MPI.Get_processor_name()
            comm.bcast(cur_node, root=r)
        else:
            cur_node = comm.bcast(None, root=r)

        # Update list of seen nodes
        if cur_node not in assigned_nodes:
            assigned_nodes.append(cur_node)

            # If this is the first unique rank of a node, it should update
            if r == rank:
                # print(f'rank {r} should update on node {cur_node}')
                ret = True

    return ret

def _get_mapping(rank, mpi_size, manager_rank, comm):
    
    if rank == manager_rank:
        tmp_rank_mapping = dict()    
        for i in range(mpi_size-1):
            r, node = comm.recv()
            if node in tmp_rank_mapping:
                tmp_rank_mapping[node].append(r)
            else:
                tmp_rank_mapping[node] = [r]
        
        # Change node name to an uid
        rank_mapping = dict()
        for nid, ranks in enumerate(tmp_rank_mapping.values()):
            rank_mapping[nid] = ranks
        
        comm.bcast(rank_mapping, root=manager_rank)
    else:
        comm.send((rank, MPI.Get_processor_name()),dest=manager_rank)
        rank_mapping = comm.bcast(None, root=manager_rank)

    return rank_mapping

def initialize(config: config_parser.Config):
    # Get base MPI variables
    global_comm = MPI.COMM_WORLD
    rank = global_comm.Get_rank()
    mpi_size = global_comm.Get_size()
    manager_rank = mpi_size - 1

    local_comm = _get_local_node_comm(global_comm)
    should_update_local = _should_update_local(mpi_size, rank, global_comm)
    rank_mapping = _get_mapping(rank, mpi_size, manager_rank, global_comm)

    # TODO: later add a 'mpi' TOP_LEVEL_BASE_CONFIG to config
    config.add_param('mpi_global_comm', global_comm)
    config.add_param('mpi_rank', rank)
    config.add_param('mpi_size', mpi_size)
    config.add_param('mpi_manager_rank', manager_rank)
    config.add_param('mpi_local_comm', local_comm)
    config.add_param('mpi_should_update_local', should_update_local)
    config.add_param('mpi_rank_mapping', rank_mapping)
