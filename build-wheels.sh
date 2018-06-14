#!/bin/bash
set -e -x

function install_rust {
    curl https://static.rust-lang.org/rustup.sh > /tmp/rustup.sh
    chmod +x /tmp/rustup.sh
    /tmp/rustup.sh -y --disable-sudo --channel=$1
}


if [[ $TRAVIS_OS_NAME == "osx" ]]; then

    brew update
    curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh --output miniconda.sh
    bash ./miniconda.sh -bp $HOME/anaconda
    export PATH="$HOME/anaconda/bin:$PATH"
    echo "Installing python version: ${PYTHON_VERSION}"
    conda install -y virtualenv python==$PYTHON_VERSION
    virtualenv venv
    source ./venv/bin/activate
    export PATH=$(pwd)/venv/bin:$PATH
    echo "Python version: $(python --version)"
    pip install -U pip setuptools setuptools-rust wheel numpy scipy pandas joblib pytest scikit-learn
    install_rust nightly
    pip wheel . -w ./wheelhouse/
    pip install -v gap-stat --no-index -f ./wheelhouse/
    pip install -r "requirements.txt"
    pytest tests -vs

else

    # Build wheels with docker run --rm -v $(pwd):/io quay.io/pypa/manylinux1_x86_64 bash /io/build-wheels.sh linux
    echo "Installing rust!"
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
        pushd /io
        "${PYBIN}/pip" install -U  setuptools setuptools-rust wheel numpy scipy pandas joblib pytest scikit-learn
        "${PYBIN}/python" setup.py bdist_wheel --dist-dir /io/wheelhouse/
        popd
    done

    # Bundle external shared libraries into the wheels
    for whl in /io/wheelhouse/gap*.whl; do
        echo "Auditing wheel ${whl}"
        auditwheel repair "$whl" -w /io/wheelhouse/
    done

    # Install packages and test
    for PYBIN in /opt/python/cp{35,36}*/bin/; do
        pushd /io
        "${PYBIN}/pip" uninstall gap-stat
        "${PYBIN}/pip" install gap-stat --no-index -f /io/wheelhouse
        # Actually require glibc 2.14 which isn't available yet on Centos...
        #"${PYBIN}/python" -m pytest tests -vs
        popd
    done

fi