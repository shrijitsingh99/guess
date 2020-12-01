#!/bin/bash
# This scripts needs to be run from the project root

export PYTHONPATH=./python
export MYPYPATH=./python

RED='\033[0;31m'
NOCOLOR='\033[0m'


# linter_results="$(flake8 --count)"
# if [ $? -ne 0 ]
# then
#     echo -e "${RED}Fix flake8 errors first:${NOCOLOR}"
#     echo "$linter_results"
#     exit 1
# fi


# mypy_results="$(mypy --config mypy.ini -p guess)"
# if [ ! -z "$mypy_results" ]
# then
#     echo -e "${RED}Fix typing (mypy) errors first:${NOCOLOR}"
#     echo "$mypy_results"
#     exit 1
# fi


if [ $# -eq 0 ]
then
    SPECIFIC_TEST=""
else
    SPECIFIC_TEST="-k $1"
fi

coverage run --source python -m py.test $SPECIFIC_TEST
coverage html
coverage report
