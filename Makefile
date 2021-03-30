SHELL:=/usr/bin/env bash

.PHONY: lint
lint:
	poetry run black --check --diff stereographic_link_prediction tests

.PHONY: black
black:
	poetry run black stereographic_link_prediction tests

.PHONY: unit
unit:
	poetry run pytest

.PHONY: package
package:
	poetry check

.PHONY: test
test: lint package unit
