


import json
import csv
from collections import defaultdict

csv_filename = "box_counts_with_categories_human.csv"
# Load your JSON data
with open("/home/arpit/rf_detr/split_dataset/test/_annotations.coco.json", "r") as file:
    data = json.load(file)

# Dictionary to store box counts per (image_id, category_id)
box_count = defaultdict(lambda: defaultdict(int))

# Count number of boxes per image ID and category ID
for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]
    box_count[image_id][category_id] += 1

# Create a mapping of image ID to file name and ensure all images are included
image_name_map = {img["id"]: img["file_name"] for img in data["images"]}
all_image_ids = set(image_name_map.keys())

# Get all unique category IDs for column headers
all_categories = sorted({ann["category_id"] for ann in data["annotations"]})

# Prepare CSV header
csv_header = ["Image Name"]
for cat_id in all_categories:
    csv_header.extend([f"Category {cat_id+1}", f"Box Count {cat_id+1}"])

# Prepare data for CSV
csv_data = [csv_header]

for img_id in all_image_ids:
    row = [image_name_map.get(img_id, f"Unknown_{img_id}")]
    category_counts = box_count.get(img_id, {})  # Get category counts or empty dict
    for cat_id in all_categories:
        row.extend([cat_id+1, category_counts.get(cat_id, 0)])  # Assign 0 if not present
    csv_data.append(row)

# Save to CSV
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)

print(f"CSV file '{csv_filename}' created successfully!")
