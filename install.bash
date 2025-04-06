#!/bin/bash

if command -v pip &>/dev/null; then
    PIP_CMD=pip
elif command -v pip3 &>/dev/null; then
    PIP_CMD=pip3
else
    echo "Error: Neither pip nor pip3 is available. Please install Python and pip first."
    exit 1
fi

$PIP_CMD install --user -r requirements.txt

if [ $? -eq 0 ]; then
    echo "All packages installed successfully."
else
    echo "Some packages failed to install. Please check the error messages above."
    exit 1
fi