#!/bin/bash

echo "Removing unnecessary files: $EXCLUDE_FILES"
for dir in $EXCLUDE_FILES; do
    rm -rf "$SRC_DIR/$dir"
done

$PYTHON -m pip install .
