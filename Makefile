# Change this to "coverage run -a" to generate test coverage data.
PYTHON=python

.PHONY: clean regtest regtest_small_block presubmit test_examples test_gpu goldens test_goldens test

presubmit: test test_gpu test_examples regtest regtest_small_block regtest_aa test_access_pattern
test: test_short test_med

# Host unit tests.
# ================

# Max 5 sec runtime.
test_short:
	$(PYTHON) tests/controller.py
	$(PYTHON) tests/node_type.py
	$(PYTHON) tests/sim.py
	$(PYTHON) tests/subdomain.py
	$(PYTHON) tests/subdomain_connection.py
	$(PYTHON) tests/subdomain_runner.py
	$(PYTHON) tests/sym.py
	$(PYTHON) tests/util.py

# Max 1 min runtime.
test_med:
	$(PYTHON) tests/sym_equilibrium.py

# GPU unit tests.
# ===============
test_gpu:
	$(PYTHON) tests/gpu/scratch_space.py
	$(PYTHON) tests/gpu/do_nothing_node.py
	$(PYTHON) tests/gpu/reduction.py
	$(PYTHON) tests/gpu/kinetic_energy_enstrophy.py
	${PYTHON} tests/gpu/time_series.py

# Other GPU tests.
# ================
test_checkpoint:
	@bash tests/gpu/checkpoint.sh examples/ldc_2d.py
	@bash tests/gpu/checkpoint.sh examples/ldc_3d.py
	@bash tests/gpu/checkpoint.sh examples/sc_drop.py
	@bash tests/gpu/checkpoint.sh "examples/binary_fluid/sc_rayleigh_taylor_2d.py --seed 123"
	@bash tests/gpu/checkpoint.sh examples/binary_fluid/fe_viscous_fingering.py

test_examples:
	@bash tests/run_examples.sh

test_access_pattern:
	@bash tests/gpu/access_pattern.sh

# Regression tests.
# =================
regtest:
	$(PYTHON) regtest/subdomains/2d_propagation.py
	$(PYTHON) regtest/subdomains/2d_ldc.py
	$(PYTHON) regtest/subdomains/2d_cylinder.py
	$(PYTHON) regtest/subdomains/3d_propagation.py
	$(PYTHON) regtest/subdomains/3d_ldc.py
	$(PYTHON) regtest/subdomains/binary_pbc.py
	$(PYTHON) regtest/subdomains/2d_binary.py

regtest_aa:
	$(PYTHON) regtest/subdomains/2d_cylinder.py --access_pattern=AA
	$(PYTHON) regtest/subdomains/2d_binary.py --access_pattern=AA
	$(PYTHON) regtest/subdomains/2d_ldc.py --access_pattern=AA
	$(PYTHON) regtest/subdomains/3d_ldc.py --access_pattern=AA
	$(PYTHON) regtest/subdomains/3d_sphere.py --access_pattern=AA
	$(PYTHON) regtest/subdomains/binary_pbc.py --access_pattern=AA

regtest_indirect:
	$(PYTHON) regtest/subdomains/2d_cylinder.py --access_pattern=AA --node_addressing=indirect
	#$(PYTHON) regtest/subdomains/2d_binary.py --access_pattern=AA --node_addressing=indirect
	$(PYTHON) regtest/subdomains/2d_ldc.py --access_pattern=AA --node_addressing=indirect
	$(PYTHON) regtest/subdomains/3d_ldc.py --access_pattern=AA --node_addressing=indirect
	$(PYTHON) regtest/subdomains/3d_sphere.py --access_pattern=AA --node_addressing=indirect
	#$(PYTHON) regtest/subdomains/binary_pbc.py --access_pattern=AA --node_addressing=indirect

# Necessary to trigger bulk/boundary split code.
regtest_small_block:
	$(PYTHON) regtest/subdomains/2d_propagation.py --block_size=16
	$(PYTHON) regtest/subdomains/2d_ldc.py --block_size=16
	$(PYTHON) regtest/subdomains/2d_cylinder.py --block_size=16
	$(PYTHON) regtest/subdomains/3d_propagation.py --block_size=16
	$(PYTHON) regtest/subdomains/3d_ldc.py --block_size=16
	$(PYTHON) regtest/subdomains/binary_pbc.py --block_size=16
	$(PYTHON) regtest/subdomains/2d_binary.py --block_size=16

# Verification tests.
# ===================
regtest_ldc:
	$(PYTHON) -u regtest/ldc_3d.py

# Regression tests against golden output files.
# =============================================
goldens:
	@mkdir -p goldens
	@bash tests/make_goldens.sh goldens

test_goldens:
	@mkdir -p tmp
	@bash tests/test_goldens.sh tmp goldens

# Performance tests.
# ==================
perf_plots:
	$(PYTHON) perftest/make_plots.py perftest/2d_single.pdf perftest/results/single/GeForce_GTX_285 2d_sc 2d_ldc 2d_poiseuille
	$(PYTHON) perftest/make_plots.py perftest/2d_binary.pdf perftest/results/single/GeForce_GTX_285 2d_bin
	$(PYTHON) perftest/make_plots.py perftest/3d.pdf perftest/results/single/GeForce_GTX_285 3d_

perf_block_plots:
	$(PYTHON) perftest/make_block_plots.py perftest perftest/results/single/GeForce_GTX_285/blocksize

clean:
	rm -f sailfish/*.pyc
	rm -f perftest/*.pdf
	rm -rf regtest/results
