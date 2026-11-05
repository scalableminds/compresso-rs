# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0]

Initial release.

- `compress`: byte-identical to `compresso.compress` (Python package 3.x),
  including the automatic `(8, 8, 1)` steps fallback, both connectivities and
  stream formats 0 and 1.
- `decompress`, `decompress_bytes`, `decompress_z_range` and
  `decompress_z_range_bytes`: identical to `compresso.decompress`, including
  random access z ranges for format 1 streams.
- `Header`, `labels`, `remap` and `valid` helpers.
- Checked decoding: malformed streams return errors and never panic.
