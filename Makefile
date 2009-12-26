
.PHONY: clean regtest2d_single regtest2d_double regtest3d_single regtest3d_double

regtest2d_single:
	python -u regtest/poiseuille.py --dim=2
	python -u regtest/poiseuille.py --dim=2 --model=mrt

regtest2d_double:
	python -u regtest/poiseuille.py --dim=2 --precision=double
	python -u regtest/poiseuille.py --dim=2 --model=mrt --precision=double

regtest3d_single:
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q13 --model=mrt
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q15
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q15 --model=mrt
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q19
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q19 --model=mrt

regtest3d_double:
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q13 --model=mrt --precision=double
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q15 --precision=double
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q15 --model=mrt --precision=double
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q19 --precision=double
	python -u regtest/poiseuille.py --dim=3 --grid=D3Q19 --model=mrt --precision=double

clean:
	rm -f *.pyc
	rm -rf regtest/results


