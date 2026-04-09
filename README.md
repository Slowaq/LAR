## Type checking
python3 -m mypy --ignore-missing-imports main.py
python3 -m flake8 main.py src/solution

## Documentation generation
python3 -m pdoc --html --output-dir output src/solution --force
