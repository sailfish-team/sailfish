
.PHONY: clean regtest

regtest:
	python regtest/poiseuille.py

clean:
	rm -f *.pyc


