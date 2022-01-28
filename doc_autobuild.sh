#!/bin/sh
# a convenient script to run sphinx-autobuild
sphinx-autobuild  \
    --host 0.0.0.0 \
    --port 9999 \
    --watch src/myfm/ \
    doc/source doc/build
