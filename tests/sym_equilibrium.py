import unittest
from sailfish import config, sym, sym_codegen, sym_equilibrium
import sympy
from sympy import Poly

class TestEntropicEquilibria(unittest.TestCase):

    def test_d3q15_2nd_order_bgk(self):
        """Verifies that the entropic equilibrium for D3Q15 is the same
        as the LBGK equilibrium up to 2nd order in velocity."""
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

    def test_d3q19_2nd_order_bgk(self):
        """Verifies that the entropic equilibrium for D3Q19 is the same
        as the LBGK equilibrium up to 2nd order in velocity."""
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


    def _verify_moments(self, grid, elbm):
        c = config.LBConfig()
        c.minimize_roundoff = False
        c.incompressible = False

        bgk = sym_equilibrium.bgk_equilibrium(grid, c)

        rho_ex = 0
        momentum = [0, 0, 0]

        for elbm_ex, ei in zip(elbm.expression, grid.basis):
            for x in elbm.local_vars:
                elbm_ex = elbm_ex.subs({x.lhs: x.rhs})
            rho_ex += elbm_ex

            for i, ei_c in enumerate(ei):
                momentum[i] += ei_c * elbm_ex

        def _v(c, i):
            if i == 0:
                return {sym.S.vx: c, sym.S.vy: 0, sym.S.vz: 0, 'vsq': c**2}
            elif i == 1:
                return {sym.S.vx: 0, sym.S.vy: c, sym.S.vz: 0, 'vsq': c**2}
            else:
                return {sym.S.vx: 0, sym.S.vy: 0, sym.S.vz: c, 'vsq': c**2}

        self.assertEqual(rho_ex.subs(_v(0, 0)), sym.S.rho)

        # The 0th and 1st moment equalities only hold approximately. We test
        # them for realistic values of velocity.
        for v in 0.0, 0.05, 0.1, 0.15:
            for i in 0, 1, 2:
                self.assertLess(abs(rho_ex.subs(_v(v, i)) / sym.S.rho - 1),
                                1e-7, '%f for %d' % (v, i))
                self.assertLess(abs(rho_ex.subs(_v(-v, i)) / sym.S.rho - 1),
                                1e-7, '%f for %d' % (-v, i))

                self.assertAlmostEqual(momentum[i].subs(_v(v, i)) / sym.S.rho, v)
                self.assertAlmostEqual(momentum[i].subs(_v(-v, i)) / sym.S.rho, -v)

    def test_d3q15_moments(self):
        grid = sym.D3Q15
        elbm = sym_equilibrium.elbm_d3q15_equilibrium(grid, order=8)
        self._verify_moments(grid, elbm)

    def test_d3q19_moments(self):
        grid = sym.D3Q19
        elbm = sym_equilibrium.elbm_d3q19_equilibrium(grid, order=8)
        self._verify_moments(grid, elbm)

if __name__ == '__main__':
    unittest.main()
