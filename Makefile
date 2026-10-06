.PHONY: test validate figures paper clean pilot

test:
	python -m pytest tests -q

validate:
	python scripts/run_local_validation.py 40

figures:
	python scripts/make_figures.py

paper: figures
	python scripts/make_paper.py
	cd paper && tectonic main.tex

pilot:
	python scripts/run_pilot_cpu.py 40 70 50

clean:
	rm -rf paper/main.pdf paper/main.log paper/main.blg paper/generated
	find . -name __pycache__ -type d -exec rm -rf {} +
