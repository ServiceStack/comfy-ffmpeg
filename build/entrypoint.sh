#!/bin/bash

set -Eeuo pipefail

# If REQUIRE_PIP_PACKAGES Environment Variable, iterate through all GitHub Repos and install to /data/custom_nodes
if [ -n "$REQUIRE_PIP_PACKAGES" ]; then
    IFS=',' read -ra ADDR <<< "$REQUIRE_PIP_PACKAGES"
    for i in "${ADDR[@]}"; do
        echo "Installing $i"
        pip install $i
    done
fi

# If REQUIRE_CUSTOM_NODES Environment Variable, iterate through all GitHub Repos and install to /data/custom_nodes
if [ -n "$REQUIRE_CUSTOM_NODES" ]; then
    IFS=',' read -ra ADDR <<< "$REQUIRE_CUSTOM_NODES"
    for i in "${ADDR[@]}"; do
        # Check if folder does not exist
        repo_name=$(basename $i)
        if [ ! -d /data/custom_nodes/$repo_name ]; then
            echo "/data/custom_nodes/$repo_name does not exist. Cloning $i"

            git clone https://github.com/$i.git /data/custom_nodes/$repo_name
            
            # Install requirements.txt if exists
            if [ -f /data/custom_nodes/$repo_name/requirements.txt ]; then
                echo "Installing requirements.txt for $i in virtual environment"
                cd /data/custom_nodes/$repo_name
                pip install -r requirements.txt
            fi
        else
            echo "/data/custom_nodes/$repo_name already exists"
            continue
        fi

    done
fi

exec "$@"
