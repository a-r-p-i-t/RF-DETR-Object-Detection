import os
import csv
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import time
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model setup
onnx_model_path = "inference_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])

# Image transform
transform = transforms.Compose([
    transforms.Resize((560, 560)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths
image_folder = "/home/arpit/rf_detr/split_dataset/test/"
output_folder = "/home/arpit/rf_detr/split_dataset/predictions/"
csv_filename = "valid_box_counts_with_categories.csv"
gt_csv = "box_counts_with_categories_human.csv"

os.makedirs(output_folder, exist_ok=True)
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]

# Data containers
image_category_counts = {}
image_fps = []

# Inference function
def process_image(image_path, file_name):
    start_time = time.time()
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: image_tensor.cpu().numpy().astype(np.float32)}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    
    print(fps)
    image_fps.append(fps)

    boxes = torch.tensor(ort_outputs[0], device=device)
    class_logits = torch.tensor(ort_outputs[1], device=device)
    class_probs = torch.nn.functional.softmax(class_logits[:, :, 1:], dim=-1)
    scores, labels = class_probs.max(dim=-1)

    confidence_threshold = 0.5
    confident_indices = scores > confidence_threshold

    boxes = boxes[confident_indices].cpu().numpy()
    scores = scores[confident_indices].cpu().numpy()
    labels = labels[confident_indices].cpu().numpy()

    # Convert center x, y, w, h to x1, y1, x2, y2
    boxes[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2)
    boxes[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    draw = ImageDraw.Draw(image)
    category_counts = {}
    
    if len(boxes) > 0:
        for box, score, label in zip(boxes, scores, labels):
            if label == 0:  # Only class 0
                x_min, y_min, x_max, y_max = box
                x_min *= orig_w
                y_min *= orig_h
                x_max *= orig_w
                y_max *= orig_h
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                draw.text((x_min, y_min), f"{label}: {score:.2f}", fill="red")
                category_counts[label] = category_counts.get(label, 0) + 1

    image.save(os.path.join(output_folder, file_name))
    return category_counts

# ======================== START TRACKING ========================
with mlflow.start_run():
    # Run inference and collect prediction counts
    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.jpg', '.png', '.jpeg')): 
            image_path = os.path.join(image_folder, file_name)
            category_counts = process_image(image_path, file_name)
            image_category_counts[file_name] = category_counts

    # Create output CSV
    all_categories = sorted(set(cat_id for counts in image_category_counts.values() for cat_id in counts))
    csv_header = ["Image Name"]
    for cat_id in all_categories:
        csv_header.extend([f"Category {cat_id+1}", f"Box Count {cat_id+1}"])

    csv_data = [csv_header]
    for img_name, counts in image_category_counts.items():
        row = [img_name]
        for cat_id in all_categories:
            row.extend([cat_id+1, counts.get(cat_id, 0)])
        csv_data.append(row)

    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)

    average_fps = sum(image_fps) / len(image_fps) if image_fps else 0
    print(f"Average FPS: {average_fps:.2f}")
    mlflow.log_metric("Average FPS", average_fps)

    # =============== EVALUATION ===============
    def load_counts(csv_path):
        data = {}
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            category_ids = [int(header[i].split()[-1]) - 1 for i in range(1, len(header), 2)]
            for row in reader:
                image = row[0]
                counts = {cat_id: int(row[i+2]) for i, cat_id in enumerate(category_ids)}
                data[image] = counts
        return data, category_ids

    pred_data, categories = load_counts(csv_filename)
    gt_data, _ = load_counts(gt_csv)

    y_true, y_pred = [], []
    for image in gt_data:
        gt_counts = gt_data.get(image, {})
        pred_counts = pred_data.get(image, {})
        for cat_id in categories:
            gt_count = gt_counts.get(cat_id, 0)
            pred_count = pred_counts.get(cat_id, 0)
            y_true.append(gt_count)
            y_pred.append(pred_count)


    # =============== BOX-LEVEL METRICS ===============
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    for image in gt_data:
        gt_counts = gt_data.get(image, {})
        pred_counts = pred_data.get(image, {})
        for cat_id in categories:
            gt_count = gt_counts.get(cat_id, 0)
            pred_count = pred_counts.get(cat_id, 0)

            tp = min(gt_count, pred_count)
            fp = max(pred_count - gt_count, 0)
            fn = max(gt_count - pred_count, 0)
            tn = 1 if gt_count == 0 and pred_count == 0 else 0

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    box_level_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)
    box_level_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    box_level_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    box_level_f1 = 2 * box_level_precision * box_level_recall / (box_level_precision + box_level_recall) if (box_level_precision + box_level_recall) > 0 else 0

    print("\n--- Box-Level Metrics ---")
    print(f"Box-Level Accuracy:  {box_level_accuracy:.4f}")
    print(f"Box-Level Precision: {box_level_precision:.4f}")
    print(f"Box-Level Recall:    {box_level_recall:.4f}")
    print(f"Box-Level F1 Score:  {box_level_f1:.4f}")

    mlflow.log_metric("Box-Level Accuracy", box_level_accuracy)
    mlflow.log_metric("Box-Level Precision", box_level_precision)
    mlflow.log_metric("Box-Level Recall", box_level_recall)
    mlflow.log_metric("Box-Level F1 Score", box_level_f1)

    # Logging extras
    mlflow.log_artifact(csv_filename)
    mlflow.log_param("Confidence Threshold", 0.5)
    mlflow.log_param("Model", onnx_model_path)
