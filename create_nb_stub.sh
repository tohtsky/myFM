#!/bin/bash
modules=( \
"myfm._myfm"
)
for module_name in "${modules[@]}"
do
    echo "Create stub for $module_name"
    output_path="src/$(echo "${module_name}" | sed 's/\./\//g').pyi"
    python -m "nanobind.stubgen" -m "$module_name" -o "$output_path"
done
