SHELL:=/usr/bin/env bash

.PHONY: lint
lint:
	poetry run black --check --diff stereographic_link_prediction tests

.PHONY: black
black:
	poetry run black stereographic_link_prediction tests

.PHONY: unit
unit:
	poetry run pytest stereographic_link_prediction tests

.PHONY: package
package:
	poetry check

.PHONY: pl_test
pl_test:
	poetry run python pl_tests/main.py

.PHONY: test
test: lint package unit
