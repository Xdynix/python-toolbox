set dotenv-load := true
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

export PYTHONUTF8 := "1"

default: lint test

# set up development environment
dev-setup:
    uv sync
    uv run pre-commit install

# execute linters
lint:
    uv run pre-commit run --all-files

# execute tests
test *args:
    uv run pytest --cov utilities {{ args }}
