#!/usr/bin/env bash#!/bin/bash
set -e -x

# Build wheels with docker run --rm -v $(pwd):/io quay.io/pypa/manylinux1_x86_64 bash /io/build-wheels.sh

mkdir ~/rust-installer
curl -sL https://static.rust-lang.org/rustup.sh -o ~/rust-installer/rustup.sh
sh ~/rust-installer/rustup.sh --prefix=~/rust --spec=nightly -y --disable-sudo
export PATH="$HOME/rust/bin:$PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/rust/lib"

# Compile wheels
for PYBIN in /opt/python/cp{35,36}*/bin; do
    export PYTHON_SYS_EXECUTABLE="$PYBIN/python"
    export PYTHON_LIB=$(${PYBIN}/python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
    export LIBRARY_PATH="$LIBRARY_PATH:$PYTHON_LIB"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PYTHON_LIB"
    "${PYBIN}/pip" install -U  setuptools setuptools-rust wheel numpy scipy pandas
    #"${PYBIN}/pip" wheel /io/ -w /io/dist/
    pushd /io
    "${PYBIN}/python" setup.py bdist_wheel --dist-dir /io/dist/
    popd
done

# Bundle external shared libraries into the wheels
for whl in /io/dist/gap*.whl; do
    echo "Auditing wheel ${whl}"
    auditwheel repair "$whl" -w /io/dist/
done

# Install packages and test
for PYBIN in /opt/python/cp{35,36}*/bin/; do
    "${PYBIN}/pip" install gap-stat --no-index -f /io/dist/
done