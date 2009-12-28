Setting up simulations
======================

Mapping physical quantities to simulation parameters
----------------------------------------------------

Having a good grasp of how choosing LBM parameters such as grid size, numerical
viscosity and maximum velocity affects the results is essential for running simulations
efficiently and for interpreting their results.  The mapping between physical quantities
describing a system and their correspoding counterparts in the simulation is often
a source of unnnecessary confusion.  In this section, we will show how to do it and
how different choices of various simulation parameters impact the precision of the
results and simulation time.

Throughout this section, we will be working with a specific physical system that
we will be using as an example.  We choose a 2D system with the following parameters:

* width: 2m
* height: 1m
* flow speed :math:`u_{phys}`: 10m/s
* Reynolds number: 1000

We will now want to determine the lattice size :math:`2N \times N`,
numerical viscosity :math:`\nu_{lb}` and the lattice time step size
:math:`\Delta t = 1 \,\mathrm{lt}`, which by definition is equal to 1 lattice time unit (lt).

The following simple relations will prove to be very useful in the next subsections:

* the lattice spacing: :math:`\Delta x = 1 \,\mathrm{lu} = 1 \,\mathrm{m} / (N-1)`, which by definition is 1 lattice unit (lu).
* the lattice flow speed: :math:`u_{lb} = \frac{\Delta t}{\Delta x} u_{phys}`
* the lattice viscosity: :math:`\nu_{lb} = \frac{\Delta t}{\Delta x^2} \frac{1}{\mathrm{Re}}`

Assuming a known lattice size and Reynolds number
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Let's choose a lattice of 101x201 nodes.  This sets the lattice
spacing to :math:`\Delta x = 0.01 \,\mathrm{m}`.  We know the Reynolds number, which is 1000, but
we need one more constraint to calculate the size of the time step and the numerical
viscosity.  The constraint we need is the maximum lattice flow speed.  The flow speed
can never exceed the lattice speed of sound :math:`c_s`, which is :math:`1/\sqrt{3}` lu/ls
for the D2Q9 lattice.  It will in fact have to be much lower than that, since the
Lattice Boltzmann model only works in the limit of low Mach numbers.  The highest
relatively safe value used in practice is 0.1 lu/lt, and this is what we are going
to use for calculation.  Setting:

.. math:: 10 \,\mathrm{m/s} = \frac{\Delta x}{\Delta t} 0.1 \,\mathrm{lu/ls}

we can easily calculate the time step size, which is :math:`\Delta t = 10^{-4} \,s`
and the lattice viscosity :math:`\nu_{lb} = 10^{-3} = 1/\mathrm{Re}`.

If we wanted to simulate a flow of a duration of 1 s, we would need 10000 iterations,
which corresponds to approximately
:math:`100 \times 200 \times 10000 = 2 \cdot 10^{8}` lattice node updates.

It is also easy to see that the size of the time step scales linearly with the maximum
velocity, i.e. if we decrease the maximum flow speed 10 times, we will need to run the
simulation 10 times longer to reach the physical time of 1s.  We will also need to decrease
the numerical viscosity :math:`\nu_{lb}` 10 times in order to make sure we're simulating the same
physical system.

Assuming a known numerical viscosity and Reynolds number
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Starting with a known numerical viscosity will some times make sense, as all
LB models have a limited range of viscosities for which they are stable.  Let's start
with a value of :math:`\nu_{lb} = 0.1` and Re = 1000 as above.  We will need to determine
the lattice spacing and time step size.  Using the lattice viscosity and lattice flow
speed equations we get, respectively: :math:`0.1 = \frac{\Delta t}{\Delta x^2} 0.001`
and :math:`0.1 = \frac{\Delta t}{\Delta x} 10`, which we can easily solve to get
:math:`\Delta x = 10^{-4} \,\mathrm{m}` and :math:`\Delta t = 10^{-6} \,\mathrm{s}`.

To get a physical duration of 1 s, we thus need :math:`10^6` iterations on a lattice of
:math:`10001 \times 20001` or :math:`2 \cdot 10^{14}` lattice node updates.  The price
to pay for the increased stability and precision of the simulation is a larger lattice
and much longer simulation time.

By decreasing the viscosity by a factor of 10, we could increase both the step size
and the time step size by a factor of 10, and thus cut the overall simulation time
by a factor of :math:`10^3` (or :math:`10^4` for 3D simulations).


