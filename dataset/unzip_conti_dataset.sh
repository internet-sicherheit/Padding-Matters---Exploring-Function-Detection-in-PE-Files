#!/bin/bash

# Set the path to the password-protected ZIP file
ZIP_FILE="conti/binaries"
# Set the password for the ZIP file
PASSWORD="infected"
# Set the output directory for extracted files
OUTPUT_DIR="conti"

# Extract the files from the ZIP file using the password
unzip -P $PASSWORD $ZIP_FILE -d $OUTPUT_DIR