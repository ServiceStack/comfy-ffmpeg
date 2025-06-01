#!/bin/bash

set -Eeuo pipefail

# Look for environment variable or use default value
COMFY_INSTALL=${COMFY_INSTALL:-/comfy/install}
COMFY_CUSTOM_NODES=${COMFY_CUSTOM_NODES:-/comfy/custom_nodes}
COMFY_DATA=${COMFY_DATA:-/data}

echo "COMFY_INSTALL: $COMFY_INSTALL"
echo "COMFY_DATA: $COMFY_DATA"

# Install all python pip packages in $COMFY_INSTALL/requirements.txt
if [ -f $COMFY_INSTALL/requirements.txt ]; then
    pip install -r $COMFY_INSTALL/requirements.txt
    echo "Installed python packages from $COMFY_INSTALL/requirements.txt"
fi

# Install all custom nodes in /comfy/install/require-nodes.txt
# Format:
# https://github.com/user/repo1
# https://github.com/user/repo2

if [ -f $COMFY_INSTALL/require-nodes.txt ]; then
    while IFS= read -r line; do
        repo_name=$(basename $line)

        # only clone if it doesn't exist
        if [ -d $COMFY_CUSTOM_NODES/$repo_name ]; then
            echo "$COMFY_CUSTOM_NODES/$repo_name already exists"
            continue
        fi

        git clone $line.git $COMFY_CUSTOM_NODES/$repo_name
        echo "Installed $line" # e.g: https://github.com/user/repo1

        # if they have a requirements.txt, install it
        if [ -f $COMFY_CUSTOM_NODES/$repo_name/requirements.txt ]; then
            echo "Installing requirements.txt for $line"
            cd $COMFY_CUSTOM_NODES/$repo_name
            pip install -r requirements.txt
        fi

    done < $COMFY_INSTALL/require-nodes.txt
fi

# Install all models in /comfy/install/require-models.txt
# Format is <path/to/download>\s+<env-with-token?>@?<url-to-download> e,g:
# nsfw/640m.onnx                                    $GITHUB_TOKEN@https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx
# nsfw/erax_nsfw_yolo11m.pt                         $HF_TOKEN@https://huggingface.co/erax-ai/EraX-NSFW-V1.0/resolve/5cb3aace4faa3e42ff6cfeb97fd93c250c65d7fb/erax_nsfw_yolo11m.pt
# diffusion_models/hidream_i1_fast_fp8.safetensors  https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/diffusion_models/hidream_i1_fast_fp8.safetensors

if [ -f $COMFY_INSTALL/require-models.txt ]; then
    while IFS= read -r line; do
        # model_path is left side of whitespace
        model_path=$(echo $line | cut -d' ' -f1)
        # model_name is right side of \s+ (multiple spaces)
        token_with_url=$(echo $line | cut -d' ' -f2-)

        url=$token_with_url
        token=""
        # if token_with_url does not start with http, extract token and url
        if [[ ! $token_with_url == http* ]]; then
            url=$(echo $token_with_url | cut -d'@' -f2)
            token=$(echo $token_with_url | cut -d'@' -f1)
        fi

        model_full_path=$COMFY_DATA/models/$model_path
        # if file already exists, skip
        if [ -f $model_full_path ]; then
            echo "$model_full_path already exists"
            continue
        fi

        mkdir -p $(dirname $model_full_path)

        # if has token, use it
        if [[ $token != "" ]]; then
            echo "Downloading $url to $model_full_path with token $token"
            # if url contains github.com, use 'Authorization token <token>' header
            if [[ $url == http*github.com* ]]; then
                curl -L -H "Authorization: token $token" $url -o $model_full_path
            else
                curl -L -H "Authorization: Bearer $token" $url -o $model_full_path
            fi
        else
            echo "Downloading $url to $model_full_path"
            curl -L $url -o $model_full_path
        fi

        # if file is < 100kb, delete it
        if [ -s $model_full_path ] && [ $(du -s $model_full_path | cut -f1) -lt 100 ]; then
            echo "Deleting incomplete $model_full_path"
            rm -f $model_full_path
        fi

        echo "Installed $url to $model_full_path"
    done < $COMFY_INSTALL/require-models.txt
fi

exec "$@"
