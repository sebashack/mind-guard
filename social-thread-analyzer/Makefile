py_bin=./venv/bin/python
pip_bin=./venv/bin/pip

.PHONY: venv
venv:
	python3 -m venv venv
	${pip_bin} install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes==0.0.2
	${pip_bin} install -r requirements.txt

.PHONY: clean
clean:
	rm -rf ./venv
	rm -rf ./_build
