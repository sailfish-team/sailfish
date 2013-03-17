import unittest
from sailfish import config, sym, sym_codegen, sym_equilibrium
from sympy import Poly

class TestEntropicEquilibria(unittest.TestCase):

    def test_d315_2nd_order_bgk(self):
        grid = sym.D3Q15
        c = config.LBConfig()
        c.minimize_roundoff = False
        c.incompressible = False

        bgk = sym_equilibrium.bgk_equilibrium(grid, c)
        elbm = sym_equilibrium.elbm_d3q15_equilibrium(grid, order=2)

        for i, (elbm_ex, bgk_ex) in enumerate(zip(elbm.expression, bgk.expression)):
            for x in elbm.local_vars:
                elbm_ex = elbm_ex.subs({x.lhs: x.rhs})

            elbm_trunc = sym_codegen.truncate_velocity(Poly(elbm_ex), order=2).subs({
                'vsq': sym.S.vx**2 + sym.S.vy**2 + sym.S.vz**2})

            self.assertEqual(elbm_trunc.expand(), bgk_ex.expand(),
                             '%d component' % i)

    def test_d319_2nd_order_bgk(self):
        grid = sym.D3Q19
        c = config.LBConfig()
        c.minimize_roundoff = False
        c.incompressible = False

        bgk = sym_equilibrium.bgk_equilibrium(grid, c)
        elbm = sym_equilibrium.elbm_d3q19_equilibrium(grid, order=2)

        for i, (elbm_ex, bgk_ex) in enumerate(zip(elbm.expression, bgk.expression)):
            for x in elbm.local_vars:
                elbm_ex = elbm_ex.subs({x.lhs: x.rhs})

            elbm_trunc = sym_codegen.truncate_velocity(Poly(elbm_ex), order=2).subs({
                'vsq': sym.S.vx**2 + sym.S.vy**2 + sym.S.vz**2})

            self.assertEqual(elbm_trunc.expand(), bgk_ex.expand(),
                            '%d component' % i)


if __name__ == '__main__':
    unittest.main()
