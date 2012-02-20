
.PHONY: clean regtest2d_single regtest2d_double regtest3d_single regtest3d_double regtest regtest_small_block presubmit

regtest2d_single:
	python -u regtest/poiseuille.py --dim=2
	python -u regtest/poiseuille.py --dim=2 --drive=pressure
	python -u regtest/poiseuille.py --dim=2 --model=mrt
	python -u regtest/poiseuille.py --dim=2 --model=mrt --drive=pressure

regtest2d_double:
	python -u regtest/poiseuille.py --dim=2 --precision=double
	python -u regtest/poiseuille.py --dim=2 --precision=double --drive=pressure
	python -u regtest/poiseuille.py --dim=2 --model=mrt --precision=double
	python -u regtest/poiseuille.py --dim=2 --model=mrt --precision=double --drive=pressure

regtest3d_single:
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q13 --model=mrt --bc=fullbb
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q15 --bc=fullbb
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q15 --model=mrt --bc=fullbb
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q19 --bc=fullbb
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q19 --model=mrt --bc=fullbb

regtest3d_double:
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q13 --model=mrt --precision=double --bc=fullbb
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q15 --precision=double --bc=fullbb
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q15 --model=mrt --precision=double --bc=fullbb
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q19 --precision=double --bc=fullbb
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q19 --model=mrt --precision=double --bc=fullbb

regtest3d_drag:
	python -u regtest/drag_coefficient.py --grid=D3Q13 --model=mrt
	python -u regtest/drag_coefficient.py --grid=D3Q15 --model=bgk
	python -u regtest/drag_coefficient.py --grid=D3Q15 --model=mrt
	python -u regtest/drag_coefficient.py --grid=D3Q19 --model=bgk
	python -u regtest/drag_coefficient.py --grid=D3Q19 --model=mrt

regtest_ldc:
	python -u regtest/ldc_3d.py

perf_plots:
	python perftest/make_plots.py perftest/2d_single.pdf perftest/results/single/GeForce_GTX_285 2d_sc 2d_ldc 2d_poiseuille
	python perftest/make_plots.py perftest/2d_binary.pdf perftest/results/single/GeForce_GTX_285 2d_bin
	python perftest/make_plots.py perftest/3d.pdf perftest/results/single/GeForce_GTX_285 3d_

perf_block_plots:
	python perftest/make_block_plots.py perftest perftest/results/single/GeForce_GTX_285/blocksize

test:
	python tests/block_runner.py
	python tests/geo_block.py
	python tests/sym.py
	python tests/util.py

regtest:
	python regtest/blocks/2d_propagation.py
	python regtest/blocks/2d_ldc.py
	python regtest/blocks/2d_cylinder.py
	python regtest/blocks/3d_propagation.py
	python regtest/blocks/3d_ldc.py
	python regtest/blocks/binary_pbc.py
	python regtest/blocks/2d_binary.py

# Necessary to trigger bulk/boundary split code.
regtest_small_block:
	python regtest/blocks/2d_propagation.py --block_size=16
	python regtest/blocks/2d_ldc.py --block_size=16
	python regtest/blocks/2d_cylinder.py --block_size=16
	python regtest/blocks/3d_propagation.py --block_size=16
	python regtest/blocks/3d_ldc.py --block_size=16
	python regtest/blocks/binary_pbc.py --block_size=16
	python regtest/blocks/2d_binary.py --block_size=16

presubmit: test regtest regtest_small_block

clean:
	rm -f sailfish/*.pyc
	rm -f perftest/*.pdf
	rm -rf regtest/results
