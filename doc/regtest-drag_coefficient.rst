Drag coefficient of a sphere (3D)
---------------------------------

Results for the D3Q13 grid.

.. plot::

    from pyplots import drag_coefficient
    drag_coefficient.make_plot(
        '../regtest/results/drag_coefficient',
        ('D3Q13/mrt/single/fullbb.dat', ),
        ('D3Q13 / MRT / single / fullbb', ))

Results for the D3Q15 grid.

.. plot::

    from pyplots import drag_coefficient
    drag_coefficient.make_plot(
        '../regtest/results/drag_coefficient',
        ('D3Q15/bgk/single/fullbb.dat', 'D3Q15/mrt/single/fullbb.dat'),
        ('D3Q15 / BGK / single / fullbb', 'D3Q15 / MRT / single / fullbb'))

Results for the D3Q19 grid.

.. plot::

    from pyplots import drag_coefficient
    drag_coefficient.make_plot(
        '../regtest/results/drag_coefficient',
        ('D3Q19/bgk/single/fullbb.dat', 'D3Q19/mrt/single/fullbb.dat'),
        ('D3Q19 / BGK / single / fullbb', 'D3Q19 / MRT / single / fullbb'))


