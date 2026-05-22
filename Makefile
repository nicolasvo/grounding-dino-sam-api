.PHONY: deploy

ifneq (,$(wildcard .env))
include .env
export
endif

deploy:
	uv run --with modal --env-file .env modal deploy modal/app.py
