Distributed simulations on a cluster
====================================

Sailfish supports running simulations across multiple hosts.  Distributed simulations
currently require either SSH access to each machine or a cluster with a GPU-aware
PBS scheduler (e.g. Torque).  In both cases, all node machines need to be able to connect to
each other using TCP connections.

Using direct SSH connections
----------------------------
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
^^^^^^^^^^^^^^^^^^^^^^^^
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

Using a PBS cluster
-------------------
Running a Sailfish simulation on a PBS cluster requires you to create two script files
-- one to set up the environment on each node, and a standard PBS job description script.
The first script will be used internally by the Sailfish controller, and for consistency
it is probably a good idea to use it in the job description script as well.  The location
of the script can be specified using the ``--cluster_pbs_initscript`` command line option,
which has a default value of ``sailfish-init.sh``.  Below we give example scripts for the
user ``myuser`` on the ZEUS cluster which is part of the PLGRID infrastructure.  Some details
such as paths will probably differ on your cluster, so you will need to adjust them accordingly.

Environment setup script (``$HOME/sailfish-init.sh``)::

    #!/bin/bash
    
    # Add paths for Python modules that are not present by default on the cluster and
    # were installed manually for this user.
    export PATH="$PATH:/storage/myuser/bin/"
    export PYTHONPATH="/storage/myuser/lib/python2.7/site-packages:$PWD:$PBS_O_WORKDIR:$PYTHONPATH"
    
    # Assume that the job was submitted from a directory containing a Sailfish installation.
    cd $PBS_O_WORKDIR
    
    # The ZEUS cluster uses the Modules system to install additional software.  The
    # settings below will normally be set automatically by PBS for the _first_ task started
    # within a job.  We need them to be available also on later tasks started by Sailfish,
    # so we copy add the settings into this script.  You can see the environment set up
    # for your job by starting an interactive job with: qsub -I -q <queue> -l nodes=1:ppn:1
    # and then running the env command.
    export MODULE_VERSION=3.2.7
    export MODULE_VERSION_STACK=3.2.7
    export MODULEPATH=/software/local/Modules/versions:/software/local/Modules/$MODULE_VERSION/modulefiles:/software/local/Modules/modulefiles
    export MODULESHOME=/software/local/Modules/3.2.7
    function module {
            eval `/software/local/Modules/$MODULE_VERSION/bin/modulecmd bash $*`
    }
    
    # Load additional modules, make sure python points to python2.7 for which the
    # Python modules installed by myuser were previously configured.
    module add python/current
    module add numpy/1.6.1
    module add cuda/4.0.17
    alias python=python2.7

PBS job script (``$HOME/sailfish-test.pbs``)::

    #!/bin/sh
    
    #PBS -l nodes=2:ppn=1:gpus=1
    #PBS -N test_sailfish
    #PBS -q gpgpu
    
    . $HOME/sailfish-init.sh
    python ./examples/lbm_cylinder_multi.py --lat_nx=2046 --lat_ny=30000 --block_size=256 --mode=benchmark --vertical \
            --every=500 --max_iters=2000 --blocks=2 --log=/mnt/lustre/scratch/people/myuser/test.log --verbose

Once you have both scripts in place and a Sailfish installation in ``$HOME/mysailfish``, you can submit the job
by running::

    cd $HOME/mysailfish
    qsub ../sailfish-test.pbs

Using InfiniBand
^^^^^^^^^^^^^^^^
Sailfish currently does not support InfiniBand (IB) explicitly.  It is however possible
to utilize IB interconnects by using the ``libsdp`` library which makes it possible to automatically
replace TCP connections with SDP ones.  To do so, you need to:

  * make sure the ``ib_sdp`` kernel module is loaded on the computational nodes,
  * add ``export LD_PRELOAD=libsdp.so`` to your ``sailfish-init.sh`` script,
  * run your simulation with the ``--cluster_pbs_interface=ib0`` parameter (this assumes
    the IB interface on the computational nodes is ``ib0``).

How it works behind the scenes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the ``--cluster_pbs`` option is set to true (default) and the ``$PBS_GPUFILE``
environment variable is set, Sailfish will assume it is running on a PBS cluster.
A cluster specification will be dynamically built using the contents of the
``$PBS_GPUFILE``.  For each machine listed in this file, ``pbsdsh`` will be used
to execute ``--cluster_pbs_initscript`` (``sailfish-init.sh`` in the previous
section), followed by ``python sailfish/socketserver.py`` (with a random port).
The socket server will then be used to establish an execnet channel to the
node and to start a local machine master.

