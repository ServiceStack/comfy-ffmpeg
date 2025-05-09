#!/bin/bash

set -Eeuo pipefail

# If REQUIRE_CUSTOM_NODES Environment Variable, iterate through all GitHub Repos and install to /data/custom_nodes
if [ -n "$REQUIRE_CUSTOM_NODES" ]; then
    IFS=',' read -ra ADDR <<< "$REQUIRE_CUSTOM_NODES"
    for i in "${ADDR[@]}"; do
        # Check if folder does not exist
        repo_name=$(basename $i)
        if [ ! -d /data/custom_nodes/$repo_name ]; then
            echo "/data/custom_nodes/$repo_name does not exist. Cloning $i"

            git clone https://github.com/$i.git /data/custom_nodes/$repo_name
            
            # Create and activate virtual environment for this custom node
            echo "Creating Python virtual environment for $repo_name"
            python -m venv /data/custom_nodes/$repo_name/venv
            
            # Install requirements.txt if exists
            if [ -f /data/custom_nodes/$repo_name/requirements.txt ]; then
                echo "Installing requirements.txt for $i in virtual environment"
                cd /data/custom_nodes/$repo_name
                source venv/bin/activate
                pip install -r requirements.txt
                deactivate
            fi
        else
            echo "/data/custom_nodes/$repo_name already exists"
            continue
        fi

    done
fi

exec "$@"
