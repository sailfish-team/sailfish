
.PHONY: clean regtest regtest_small_block presubmit test_examples test_gpu

regtest_ldc:
	python -u regtest/ldc_3d.py

perf_plots:
	python perftest/make_plots.py perftest/2d_single.pdf perftest/results/single/GeForce_GTX_285 2d_sc 2d_ldc 2d_poiseuille
	python perftest/make_plots.py perftest/2d_binary.pdf perftest/results/single/GeForce_GTX_285 2d_bin
	python perftest/make_plots.py perftest/3d.pdf perftest/results/single/GeForce_GTX_285 3d_

perf_block_plots:
	python perftest/make_block_plots.py perftest perftest/results/single/GeForce_GTX_285/blocksize

test:
	python tests/subdomain_connection.py
	python tests/subdomain_runner.py
	python tests/subdomain.py
	python tests/sim.py
	python tests/sym.py
	python tests/util.py

test_examples:
	@bash tests/run_examples.sh

test_gpu:
	python tests/gpu/scratch_space.py

test_checkpoint:
	@bash tests/gpu/checkpoint.sh examples/ldc_2d.py
	@bash tests/gpu/checkpoint.sh examples/ldc_3d.py
	@bash tests/gpu/checkpoint.sh examples/sc_drop.py

regtest:
	python regtest/subdomains/2d_propagation.py
	python regtest/subdomains/2d_ldc.py
	python regtest/subdomains/2d_cylinder.py
	python regtest/subdomains/3d_propagation.py
	python regtest/subdomains/3d_ldc.py
	python regtest/subdomains/binary_pbc.py
	python regtest/subdomains/2d_binary.py

# Necessary to trigger bulk/boundary split code.
regtest_small_block:
	python regtest/subdomains/2d_propagation.py --block_size=16
	python regtest/subdomains/2d_ldc.py --block_size=16
	python regtest/subdomains/2d_cylinder.py --block_size=16
	python regtest/subdomains/3d_propagation.py --block_size=16
	python regtest/subdomains/3d_ldc.py --block_size=16
	python regtest/subdomains/binary_pbc.py --block_size=16
	python regtest/subdomains/2d_binary.py --block_size=16

presubmit: test test_gpu test_examples regtest regtest_small_block

clean:
	rm -f sailfish/*.pyc
	rm -f perftest/*.pdf
	rm -rf regtest/results
