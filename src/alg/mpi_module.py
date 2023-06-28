from mpi4py import MPI

import config_parser


def _get_local_node_comm(comm):
    """
    Return a split communicator for processes within the same node.
    For instance, running 4 nodes and 8 processes would yield 4 different
    communicators with 2 processes.

    This is required for using distributed hdf5. The MPI_File_open routine
    needs the same file for the input MPI communicator, and for running
    locally the files although having the same name and path, are different.
    """

    # Get all node names
    local_host_name = MPI.Get_processor_name()
    node_names = comm.allgather(local_host_name)

    # Get unique sorted
    node_names = list(set(node_names))
    node_names.sort()

    # Perform split
    local_color = node_names.index(local_host_name)
    local_comm = comm.Split(local_color)

    return local_comm


def _should_update_local(mpi_size, rank, comm):
    """
    Check whether the current process should update the local
    h5 porosity file.
    Only one process per node should do this.
    Although multiple updates works on h5, it is inefficient.
    """

    ret = False
    assigned_nodes = []
    for r in range(mpi_size):
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


def initialize(config: config_parser.Config):
    """ """

    # Get base MPI variables
    global_comm = MPI.COMM_WORLD
    rank = global_comm.Get_rank()
    mpi_size = global_comm.Get_size()
    manager_rank = mpi_size - 1

    local_comm = _get_local_node_comm(global_comm)
    should_update_local = _should_update_local(mpi_size, rank, global_comm)

    # TODO: later add a 'mpi' TOP_LEVEL_BASE_CONFIG to config
    config.add_param("mpi_global_comm", global_comm)
    config.add_param("mpi_rank", rank)
    config.add_param("mpi_size", mpi_size)
    config.add_param("mpi_manager_rank", manager_rank)
    config.add_param("mpi_local_comm", local_comm)
    config.add_param("mpi_should_update_local", should_update_local)
