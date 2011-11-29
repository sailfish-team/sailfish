Distributed simulations on a cluster
====================================

Sailfish supports running simulations across multiple hosts.  The following
requirements have to be fulfilled in order for this to work:

* the user needs an SSH account on each machine,
* all node machines need to be able to connect to each other using TCP connections.

A distributed Sailfish simulation is started just like a local simulation,
the only difference being the presence of the ``cluster_spec`` config parameter, whose
value is a path to a cluster definition file.  If the path is relative, the controller
will look for it in the current working directory as well as in the ``.sailfish``
directory in the user's home.

Optionally, the ``cluster_sync``
argument can also be specified to automatically sync files from the controller to
the nodes.  A common value to use here is ``$PWD:.``, which means "sync the contents
of the current directory (``$PWD``) to the working directory on every node (``.``)".

Cluster definition files
------------------------
A cluster definition file is a Python script that contains a global ``nodes`` list
of :class:`MachineSpec` instances.  Each :class:`MachineSpec` defines a cluster node.
Here is sample file defining two nodes::

    from sailfish.config import MachineSpec

    nodes = [
            MachineSpec('ssh=user1@hosta//chdir=/home/user1/sailfish-cluster',
                'hosta', cuda_nvcc='/usr/local/cuda/bin/nvcc',
                gpus=[0, 1]),
            MachineSpec('ssh=user2@hostb//chdir=/home/user2/sailfish-cluster',
                'hostb', cuda_nvcc='/usr/local/cuda/bin/nvcc',
                gpus=[0, 2], block_size=256)
    ]

The two nodes are located on ``hosta`` and ``hostb`` respectively, and different
user accounts are used to get access to both hosts.  The ``cuda_nvcc`` and ``block_size``
parameters are standard Sailfish config parameters than can be specified from the
command line for single-host simulations.  Here we use them to provide a location
of the NVCC compiler and to use a larger CUDA block size on ``hostb`` (e.g. because
it has Fermi-generation GPUs).  The first argument of :class:`MachineSpec` is an
execnet address string.  See the execnet documentation for more information about all
supported features.  For common configurations, just modifying the above example should
work fine.

Note that each node definition contains a list of available GPUs.  This information
is used for load-balancing purposes.  By default, only the first available GPU (``[0]``)
will be used, even on a multi-GPU machine.
