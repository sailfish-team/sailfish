
.PHONY: clean regtest2d_single regtest2d_double regtest3d_single regtest3d_double

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

clean:
	rm -f *.pyc
	rm -rf regtest/results


