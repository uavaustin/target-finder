# Changelog

## v0.3.1 (2018-10-12)

### Fixes

- Fixed an undefined reference to the `kernel` variable in the preprocessing.

## v0.3.0 (2018-10-20)

### Features

- Added compatiblity for `target-finder-model` `v0.2.0`.
- Added contour dilation and erosion to connect nearby contours.
- Now using color area to find background and alphanumeric color.
- Added a version flag to the cli.
- Blobs and targets can now be serialized to strings for debugging and
  printing.

### Chores

- Remove use of `FastGFile` per upcoming Tensorflow deprecation.

## v0.2.0 (2018-08-09)

### Features

- `target-finder-cli targets <file...>` subcommand has been added.
- Added support for specifying the maximum blob width in the `blobs`
  subcommand.
- CLI can have arguments added directly in `target_finder.cli.run(...)` that
  override `sys.argv`.

### Breaking Changes

- The [`target-finder-model`](https://github.com/uavaustin/target-finder-model)
  module must be installed separately, since the model is now treated as an
  external dependency that is not listed in `setup.py`.
- Renamed the `max_length` keyword argument in `target_finder.find_blobs(...)`
  to `max_width` to match the existing `min_width` argument.

### Fixes

- Fixed the default padding setting on the `blobs` subcommand to match the
  default value for `target_finder.find_blobs(...)`.
- Confidence numbers for targets are now correctly returned as standard floats
  instead of numpy floats in `target_finder.find_targets(...)`.

### Chores

- Added tox for unit testing.
- Enabled code coverage on tests.
- Dependencies are listed inside of `setup.py` instead of in their own file.
- Uploading releases is now done by Travis CI.

## v0.1.1 (2018-07-19)

### Fixes

- Fixed a problem where blobs which were identified as not being shapes were
  being returned in `target_finder.find_targets(...)`.

## v0.1.0 (2018-07-15)

Initial release.
