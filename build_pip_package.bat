rm -rf dist/*
python setup.py clean --all bdist_wheel
twine upload dist/*