#!/bin/bash

# Set the URL and file name of the archive to download
URL="https://osf.io/download/73c6v/?view_only=5c205088485f4380a78e99064d37344a"
FILE="checkpoints.tar.xz"

# Download the file using wget
wget -O "$FILE" "$URL"

# Unpack the archive using tar
tar -xJf "$FILE"

# Remove the downloaded archive
rm "$FILE"