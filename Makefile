
.PHONY: clean regtest

regtest:
	[ ! -d regtest/results ] && mkdir regtest/results
	python regtest/poiseuille.py

clean:
	rm -f *.pyc
	rm -rf regtest/results


