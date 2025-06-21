#!/bin/bash

set -Eeuo pipefail

# Look for environment variable or use default value
COMFY_CUSTOM_NODES=${COMFY_CUSTOM_NODES:-/comfy/custom_nodes}
COMFY_DATA=${COMFY_DATA:-/data}

echo "COMFY_CUSTOM_NODES: $COMFY_CUSTOM_NODES"
echo "COMFY_DATA: $COMFY_DATA"

install_custom_node() {
    repo_url=$1
    repo_name=$(basename $repo_url)

    # only clone if it doesn't exist
    if [ -d $COMFY_CUSTOM_NODES/$repo_name ]; then
        echo "$COMFY_CUSTOM_NODES/$repo_name already exists"
        return
    fi

    git clone $repo_url.git $COMFY_CUSTOM_NODES/$repo_name
    echo "Installed $repo_url" # e.g: https://github.com/user/repo1

    # if they have a requirements.txt, install it
    if [ -f $COMFY_CUSTOM_NODES/$repo_name/requirements.txt ]; then
        echo "Installing requirements.txt for $repo_url"
        cd $COMFY_CUSTOM_NODES/$repo_name
        pip install -r requirements.txt
    fi
}

install_custom_node "https://github.com/ServiceStack/comfy-agent"

exec "$@"
