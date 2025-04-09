import os
import csv
from rfdetr import RFDETRBase
from PIL import Image, ImageDraw
import time
# Initialize model
model = RFDETRBase(device="cuda")

# Paths
folder_path = "/home/arpit/rf_detr/split_dataset/test"  # Folder containing images
output_dir = "/home/arpit/rf_detr/test_results"  # Directory to save visualizations
csv_filename = "valid_box_counts_with_categories.csv"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store category-wise box counts
image_category_counts = {}
image_fps = []

def process_image(image_path, file_name):
    start_time = time.time()
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    detections = model.predict(image_path)
    end_time = time.time()
    
    processing_time = end_time - start_time  # Time taken to process the image
    
    # Calculate FPS
    fps = 1 / processing_time if processing_time > 0 else 0
    image_fps.append(fps)
    
    category_counts = {}
    
    for bbox, conf, cls_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"Class {cls_id} {conf:.2f}", fill="red")
        
        # Count boxes per category
        category_counts[cls_id] = category_counts.get(cls_id, 0) + 1
    
    image.save(os.path.join(output_dir, file_name))
    
    return category_counts

# Process all images in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(('.jpg', '.png', '.jpeg')):  # Process only image files
        image_path = os.path.join(folder_path, file_name)
        category_counts = process_image(image_path, file_name)
        image_category_counts[file_name] = category_counts

# Get all unique category IDs
all_categories = sorted(set(cat_id for counts in image_category_counts.values() for cat_id in counts))

# Prepare CSV header
csv_header = ["Image Name"]
for cat_id in all_categories:
    csv_header.extend([f"Category {cat_id}", f"Box Count {cat_id}"])

# Prepare data for CSV
csv_data = [csv_header]
for img_name, counts in image_category_counts.items():
    row = [img_name]
    for cat_id in all_categories:
        row.extend([cat_id, counts.get(cat_id, 0)])  # If category not present, count = 0
    csv_data.append(row)
    
average_fps = sum(image_fps) / len(image_fps) if image_fps else 0
print(f"Average FPS: {average_fps:.2f}")

# Save results to CSV
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)

print(f"CSV file '{csv_filename}' created successfully!")
