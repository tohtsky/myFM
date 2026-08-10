# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-08-10

### Added
- Support for Python 3.14, and declare Python 3.12/3.13/3.14 classifiers in packaging metadata (#20).

### Changed
- Bump build requirement to `scikit-build-core >= 1.0.0` and `nanobind >= 2.6.0` (#20).
- Build wheels only for `cp312-*` in CI (#20).
- Cross-compile macOS x86_64 wheels on an arm64 runner for faster CI (#22).
- Fetch version from `_version.py` instead of `pkg_resources` (#20).

### Fixed
- abi3 wheel build failing `abi3audit` (#21).
- `test_ml1m` failure on pandas 3 (#20).
- Broken doctests (#20).
