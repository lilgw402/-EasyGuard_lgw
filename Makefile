# Makefile 用于本地方便地执行一些指令


env:
	pip3 install -r requirements/develop.txt; \
	pre-commit install


lint:
	sh ./dev/code_style_check.sh


