#!/usr/bin/env bash

@ECHO ON
rustc -v
cargo -v
rustup -v
cargo build --release --verbose