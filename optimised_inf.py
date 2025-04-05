import os
import csv
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

onnx_model_path = "inference_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])

providers = ort.get_available_providers()
print("Available providers:", providers)

transform = transforms.Compose([
    transforms.Resize((560, 560)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_folder = "/home/arpit/rf_detr/split_dataset/test/"
output_folder = "/home/arpit/rf_detr/split_dataset/predictions/"
csv_filename = "valid_box_counts_with_categories.csv"
confidence_threshold = 0.5

os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists

image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]

image_category_counts = {}
image_fps = []

def process_image(image_path, file_name):
    start_time = time.time()
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: image_tensor.cpu().numpy().astype(np.float32)}

    ort_outputs = ort_session.run(None, ort_inputs)
    
    end_time = time.time()
    processing_time = end_time - start_time  # Time taken to process the image
    
    fps = 1 / processing_time if processing_time > 0 else 0
    image_fps.append(fps)

    boxes = torch.tensor(ort_outputs[0], device=device)  # (1, 300, 4)
    class_logits = torch.tensor(ort_outputs[1], device=device)
    
    # print(class_logits)

    class_probs = torch.nn.functional.softmax(class_logits[:, :, 1:], dim=-1)
    print(class_probs)
    scores, labels = class_probs.max(dim=-1)

    confidence_threshold = confidence_threshold
    confident_indices = scores > confidence_threshold

    boxes = boxes[confident_indices].cpu().numpy()
    scores = scores[confident_indices].cpu().numpy()
    labels = labels[confident_indices].cpu().numpy()

    # Rescale boxes to original image size
    boxes[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2)
    boxes[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    draw = ImageDraw.Draw(image)

    category_counts = {}
    
    if len(boxes) > 0:  
        for box, score, label in zip(boxes, scores, labels):
            # if label == 0:  
            x_min, y_min, x_max, y_max = box
            x_min_scaled = x_min * orig_w
            y_min_scaled = y_min * orig_h
            x_max_scaled = x_max * orig_w
            y_max_scaled = y_max * orig_h

            draw.rectangle([x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled], outline="red", width=2)
            draw.text((x_min_scaled, y_min_scaled), f"{label}: {score:.2f}", fill="red")

            category_counts[label] = category_counts.get(label, 0) + 1

        output_path = os.path.join(output_folder, file_name)
        image.save(output_path)
        print(f"Saved: {output_path}")
    else:  

        output_path = os.path.join(output_folder, file_name)
        image.save(output_path)
        print(f"Saved: {output_path} (No objects detected)")

    return category_counts

for file_name in os.listdir(image_folder):
    if file_name.endswith(('.jpg', '.png', '.jpeg')):  # Process only image files
        image_path = os.path.join(image_folder, file_name)
        category_counts = process_image(image_path, file_name)
        image_category_counts[file_name] = category_counts

all_categories = sorted(set(cat_id for counts in image_category_counts.values() for cat_id in counts))

csv_header = ["Image Name"]
for cat_id in all_categories:
    csv_header.extend([f"Category {cat_id+1}", f"Box Count {cat_id+1}"])

csv_data = [csv_header]
for img_name, counts in image_category_counts.items():
    row = [img_name]
    for cat_id in all_categories:
        row.extend([cat_id+1, counts.get(cat_id, 0)])  # If category not present, count = 0
    csv_data.append(row)

average_fps = sum(image_fps) / len(image_fps) if image_fps else 0
print(f"Average FPS: {average_fps:.2f}")

with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)

print(f"CSV file '{csv_filename}' created successfully!")
