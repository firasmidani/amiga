echo "Excluding the following: $EXCLUDE_FILES"

echo $EXCLUDE_FILES | xargs rm -rf

$PYTHON -m pip install .