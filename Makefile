
.PHONY: clean regtest

regtest:
	python -u regtest/poiseuille.py
	python -u regtest/poiseuille.py --precision=double

clean:
	rm -f *.pyc
	rm -rf regtest/results


