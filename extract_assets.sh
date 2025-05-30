#!/bin/bash

# Find all JSON files in workflows directory and extract AssetDownloader widgets_values
find data/user/default/workflows -name "*.json" -type f -not -name ".index.json" -exec jq '.nodes[] | select(.type == "AssetDownloader") | { url: .widgets_values[0], save_path: .widgets_values[1], filename: .widgets_values[2], token: .widgets_values[3] }' {} \; >> extract_url.json

# Convert to array format, sort by URL, and remove duplicates
jq -s '. | unique_by(.url) | sort_by(.url)' extract_url.json > extract_urls_temp.json

# Remove empty tokens and add file path check
jq '[ .[] | 
  if (.token == "" or .token == null) then
    del(.token)
  else
    .
  end
  | . + 
  if (.save_path != null and .filename != null and (.save_path | type) == "string" and (.filename | type) == "string") then
    {
      "filepath": ("./data/models/" + .save_path + "/" + .filename)
    }
  else
    {
      "filepath": null
    }
  end
]' extract_urls_temp.json > extract_urls_with_filepath.json

# Now check if files exist and add size information
cat extract_urls_with_filepath.json | jq -c '.[]' | while read -r item; do
  filepath=$(echo "$item" | jq -r '.filepath // ""')
  
  if [ -n "$filepath" ] && [ -f "$filepath" ]; then
    size=$(du -h "$filepath" | cut -f1)
    echo "$item" | jq ". + {\"size\": \"$size\"}" | jq -r 'del(.filepath)'
  else
    echo "$item" | jq '. + {}' | jq -r 'del(.filepath)'
  fi
done | jq -s '.' > extract_assets.json


# Clean up temporary files
rm extract_url.json extract_urls_temp.json extract_urls_with_filepath.json
echo "Asset URLs extracted to extract_urls.json (sorted, deduplicated, and with file size information)"

