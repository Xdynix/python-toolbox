set dotenv-load

[windows]
set shell := ["powershell.exe", "-NoLogo", "-Command"]

export PYTHONUTF8 := "1"

default: lint test

# set up development environment
dev-setup:
    uv sync
    uv run pre-commit install

# run ruff linter and formatter
ruff:
    uv run ruff check --fix .
    uv run ruff format .

# execute all linters
lint:
    uv run pre-commit run --all-files

# audit locked dependencies for known vulnerabilities
audit:
    uv audit --frozen

# execute tests
test *args:
    uv run pytest --cov utilities {{ args }}
