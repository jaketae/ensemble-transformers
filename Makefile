.PHONY: quality style clean release beta

check_dirs := ensemble_transformers tests

quality: # Check that source code meets quality standards
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs) --max-line-length 119

style: # Format source code automatically
	black $(check_dirs)
	isort $(check_dirs)

clean:
	rm -r build dist ensemble_transformers.egg-info

build:
	python setup.py sdist bdist_wheel

beta:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

publish:
	twine upload dist/*