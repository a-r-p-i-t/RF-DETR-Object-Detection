import json

# Load the JSON file
json_path = "/home/arpit/rf_detr/split_dataset/test/_annotations.coco.json"  # Replace with your actual file path

with open(json_path, "r") as f:
    data = json.load(f)

# Modify category_id values
for annotation in data.get("annotations", []):
    if annotation["category_id"] == 1:
        annotation["category_id"] = 0

# Save the modified JSON
with open(json_path, "w") as f:
    json.dump(data, f, indent=4)

print("Updated category_id from 1 to 0 successfully!")
