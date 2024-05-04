py_bin=./venv/bin/python
pip_bin=./venv/bin/pip

.PHONY: venv
venv:
	python3 -m venv venv
	${pip_bin} install -r requirements.txt
	${py_bin} -m spacy download en_core_web_sm

.PHONY: clean
clean:
	rm -rf ./venv
	rm -rf ./_build
