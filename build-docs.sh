#!/bin/bash

SOURCE_CODE_PATH=$1

echo "building docs for $SOURCE_CODE_PATH ..."

rm -Rf docs/build
# uncomment following two lines if you want to regenerate docs structure
# is currently disables because we tweaked the content of the generated files
#rm -Rf docs/source/docstring
#sphinx-apidoc -f -o docs/source/docstring/ $SOURCE_CODE_PATH

cd docs && make html && cd ..

