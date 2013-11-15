
.PHONY: clean regtest regtest_small_block presubmit test_examples test_gpu goldens test_goldens test

regtest_ldc:
	python -u regtest/ldc_3d.py

perf_plots:
	python perftest/make_plots.py perftest/2d_single.pdf perftest/results/single/GeForce_GTX_285 2d_sc 2d_ldc 2d_poiseuille
	python perftest/make_plots.py perftest/2d_binary.pdf perftest/results/single/GeForce_GTX_285 2d_bin
	python perftest/make_plots.py perftest/3d.pdf perftest/results/single/GeForce_GTX_285 3d_

perf_block_plots:
	python perftest/make_block_plots.py perftest perftest/results/single/GeForce_GTX_285/blocksize

test: test_short, test_med

# Max 5 sec runtime.
test_short:
	python tests/controller.py
	python tests/sim.py
	python tests/subdomain.py
	python tests/subdomain_connection.py
	python tests/subdomain_runner.py
	python tests/sym.py
	python tests/util.py

# Max 1 min runtime.
test_med:
	python tests/sym_equilibrium.py

test_examples:
	@bash tests/run_examples.sh

test_gpu:
	python tests/gpu/scratch_space.py
	python tests/gpu/do_nothing_node.py
	python tests/gpu/reduction.py
	python tests/gpu/kinetic_energy_enstrophy.py

test_checkpoint:
	@bash tests/gpu/checkpoint.sh examples/ldc_2d.py
	@bash tests/gpu/checkpoint.sh examples/ldc_3d.py
	@bash tests/gpu/checkpoint.sh examples/sc_drop.py
	@bash tests/gpu/checkpoint.sh "examples/binary_fluid/sc_rayleigh_taylor_2d.py --seed 123"
	@bash tests/gpu/checkpoint.sh examples/binary_fluid/fe_viscous_fingering.py

test_access_pattern:
	@bash tests/gpu/access_pattern.sh

regtest:
	python regtest/subdomains/2d_propagation.py
	python regtest/subdomains/2d_ldc.py
	python regtest/subdomains/2d_cylinder.py
	python regtest/subdomains/3d_propagation.py
	python regtest/subdomains/3d_ldc.py
	python regtest/subdomains/binary_pbc.py
	python regtest/subdomains/2d_binary.py

regtest_aa:
	python regtest/subdomains/2d_cylinder.py --access_pattern=AA
	python regtest/subdomains/2d_binary.py --access_pattern=AA
	python regtest/subdomains/2d_ldc.py --access_pattern=AA
	python regtest/subdomains/3d_ldc.py --access_pattern=AA
	python regtest/subdomains/binary_pbc.py --access_pattern=AA

# Necessary to trigger bulk/boundary split code.
regtest_small_block:
	python regtest/subdomains/2d_propagation.py --block_size=16
	python regtest/subdomains/2d_ldc.py --block_size=16
	python regtest/subdomains/2d_cylinder.py --block_size=16
	python regtest/subdomains/3d_propagation.py --block_size=16
	python regtest/subdomains/3d_ldc.py --block_size=16
	python regtest/subdomains/binary_pbc.py --block_size=16
	python regtest/subdomains/2d_binary.py --block_size=16

goldens:
	@mkdir -p goldens
	@bash tests/make_goldens.sh goldens

test_goldens:
	@mkdir -p tmp
	@bash tests/test_goldens.sh tmp goldens

presubmit: test test_gpu test_examples regtest regtest_small_block regtest_aa test_access_pattern

clean:
	rm -f sailfish/*.pyc
	rm -f perftest/*.pdf
	rm -rf regtest/results
