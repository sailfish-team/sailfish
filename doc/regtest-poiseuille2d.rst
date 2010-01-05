Poiseuille flow (2D)
--------------------

Body force driving
^^^^^^^^^^^^^^^^^^

Single precision:

.. plot::

    from pyplots import poiseuille
    poiseuille.make_plot('../regtest/results/poiseuille',
        ('D2Q9/bgk/force/single/fullbb.dat',
         'D2Q9/mrt/force/single/fullbb.dat',
         'D2Q9/bgk/force/single/zouhe.dat',
         'D2Q9/mrt/force/single/zouhe.dat'),
        ('BGK / single / fullbb', 'MRT / single / fullbb',
         'BGK / single / zouhe', 'MRT / single / zouhe'))


Double precision:

.. plot::

    from pyplots import poiseuille
    poiseuille.make_plot('../regtest/results/poiseuille',
        ('D2Q9/bgk/force/double/fullbb.dat',
         'D2Q9/mrt/force/double/fullbb.dat',
         'D2Q9/bgk/force/double/zouhe.dat',
         'D2Q9/mrt/force/double/zouhe.dat'),
        ('BGK / double / fullbb', 'MRT / double / fullbb',
         'BGK / double / zouhe', 'MRT / double / zouhe'))


Pressure driving
^^^^^^^^^^^^^^^^

Single precision:

.. plot::

    from pyplots import poiseuille
    poiseuille.make_plot('../regtest/results/poiseuille',
        ('D2Q9/bgk/pressure/single/equilibrium.dat',
         'D2Q9/mrt/pressure/single/equilibrium.dat',
         'D2Q9/bgk/pressure/single/zouhe.dat',
         'D2Q9/mrt/pressure/single/zouhe.dat'),
        ('BGK / single / equilibrium', 'MRT / single / equilibrium',
         'BGK / single / zouhe', 'MRT / single / zouhe'))


Double precision:

.. plot::

    from pyplots import poiseuille
    poiseuille.make_plot('../regtest/results/poiseuille',
        ('D2Q9/bgk/pressure/double/equilibrium.dat',
         'D2Q9/mrt/pressure/double/equilibrium.dat',
         'D2Q9/bgk/pressure/double/zouhe.dat',
         'D2Q9/mrt/pressure/double/zouhe.dat'),
        ('BGK / double / equilibrium', 'MRT / double / equilibrium',
         'BGK / double / zouhe', 'MRT / double / zouhe'))

