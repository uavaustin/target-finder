#!/bin/sh -e

cd $(dirname "$0")

graph_file="../target_finder/data/graph.pb"
labels_file="../target_finder/data/labels.txt"

# Check that the model files exist.
[ -f "$graph_file" ] || (>&2 echo "Missing graph.pb" && exit 1)
[ -f "$labels_file" ] || (>&2 echo "Missing labels.txt" && exit 1)

# Find the version number to release.
version=$(grep -o -e "'.*'" "../target_finder/version.py" | tr -d "'")

echo "Detected version ""$version"

tf_stage_dir="../release/staging/target-finder"
archive_name="target-finder-""$version"".tar.gz"

# Create the staging directory and the target-finder folder.
echo "Staging files"
mkdir -p "$tf_stage_dir"

# Copy over python files.
mkdir -p "$tf_stage_dir""/target_finder"
find "../target_finder/" -name "*.py" -exec cp '{}' \
  "$tf_stage_dir/target_finder/" \;

# Copy over the graph and labels.
mkdir -p "$tf_stage_dir""/target_finder/data"
cp "$graph_file" "$tf_stage_dir""/target_finder/data/"
cp "$labels_file" "$tf_stage_dir""/target_finder/data/"

# Copy over configuration and informational files.
cp ../README.md ../LICENSE ../CHANGELOG.md ../MANIFEST.in ../requirements.txt \
  ../setup.py "$tf_stage_dir"

# Compress the directory.
echo "Creating archive"

cd "../release/staging"
tar -czvf "$archive_name" "target-finder"
mv "$archive_name" ..

echo "\033[32mCreated target-finder release" \
  "(""$archive_name"")\033[0m"

# Remove the staging directory.
echo "Removing staging files"
cd ..
rm -rf staging
