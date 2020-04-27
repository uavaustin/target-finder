#!/bin/bash -e

# Some helper scripts to quite the pushd and popd
pushd () {
    command pushd "$@" > /dev/null
}

popd () {
    command popd "$@" > /dev/null
}

cd $(dirname "$0")

pushd ..
# Find the version number to release.
version=$(grep -o -e '".*"' "target_finder/version.py" | sed 's/"//g')

echo "Detected version ""$version"

tf_stage_dir="release/staging/target-finder"
archive_name="target-finder-""$version"".tar.gz"

# Create the staging directory and the target-finder folder.
echo "Staging files"
mkdir -p "$tf_stage_dir"

# Copy over python files.
mkdir -p "$tf_stage_dir""/target_finder"
find "target_finder/" -name "*.py" -exec cp '{}' \
  "$tf_stage_dir/target_finder/" \;

# Copy over configuration and informational files.
cp README.md LICENSE setup.py "$tf_stage_dir"

# Compress the directory.
echo "Creating archive"

pushd release/staging
tar -C  "target-finder" -czf "../$archive_name" .
popd

echo -e "\033[32mCreated target-finder release" \
  "(""$archive_name"")\033[0m"

# Remove the staging directory.
echo "Removing staging files"

rm -rf release/staging
popd