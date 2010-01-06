Poiseuille flow (3D)
--------------------

Body force driving
^^^^^^^^^^^^^^^^^^

Results for the D3Q13 grid.

.. plot::

    from pyplots import poiseuille
    poiseuille.make_plot('../regtest/results/poiseuille3d',
         ('D3Q13/mrt/force/single/fullbb.dat', ),
         ('D3Q13 / MRT / single / fullbb', ))

Results for the D3Q15 grid.

.. plot::

    from pyplots import poiseuille
    poiseuille.make_plot('../regtest/results/poiseuille3d',
         ('D3Q15/bgk/force/single/fullbb.dat', 'D3Q15/mrt/force/single/fullbb.dat'),
         ('D3Q15 / BGK / single / fullbb', 'D3Q15 / MRT / single / fullbb'))

Results for the D3Q19 grid.

.. plot::

    from pyplots import poiseuille
    poiseuille.make_plot('../regtest/results/poiseuille3d',
         ('D3Q19/bgk/force/single/fullbb.dat', 'D3Q19/mrt/force/single/fullbb.dat'),
         ('D3Q19 / BGK / single / fullbb', 'D3Q19 / MRT / single / fullbb'))


