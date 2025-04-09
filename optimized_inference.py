import os
import csv
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import time

device = "cuda" if torch.cuda.is_available() else "cpu" 
# device = "cpu"
print(f"Using device: {device}")

onnx_model_path = "/home/arpit/rf_detr/inference_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])

providers = ort.get_available_providers()

transform = transforms.Compose([
    transforms.Resize((560, 560)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_folder = "/home/arpit/rf_detr/split_dataset/test"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]
output_folder = "/home/arpit/rf_detr/split_dataset/predictions"
os.makedirs(output_folder, exist_ok=True)
image_fps = []





def process_image(image_path, file_name):
    start_time = time.time()
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    image_tensor = transform(image).unsqueeze(0)
    
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: image_tensor.numpy().astype(np.float32)}

    ort_outputs = ort_session.run(None, ort_inputs)
    
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(fps)
    image_fps.append(fps)

    boxes = torch.tensor(ort_outputs[0])
    class_logits = torch.tensor(ort_outputs[1])
    
    confidence_threshold = 0.5

    if class_logits.shape[-1] != 1:
        class_probs = torch.nn.functional.softmax(class_logits[:, :, 1:], dim=-1)
        scores, labels = class_probs.max(dim=-1)
    else:
        scores = torch.nn.functional.sigmoid(class_logits[:, :, 0])
        labels = torch.zeros_like(scores, dtype=torch.int)

    confident_indices = scores > confidence_threshold

    # Move everything to CPU for safe indexing
    confident_indices = confident_indices.to("cpu")
    boxes = boxes.to("cpu")
    scores = scores.to("cpu")
    labels = labels.to("cpu")

    boxes = boxes[confident_indices].numpy()
    scores = scores[confident_indices].numpy()
    labels = labels[confident_indices.numpy()] if class_logits.shape[-1] == 1 else labels[confident_indices].numpy()

    # Rescale boxes to corner format
    boxes[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2)
    boxes[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    draw = ImageDraw.Draw(image)
    
    if len(boxes) > 0:
        for box, score, label in zip(boxes, scores, labels):
            if label == 0:
                x_min, y_min, x_max, y_max = box
                x_min_scaled = x_min * orig_w
                y_min_scaled = y_min * orig_h
                x_max_scaled = x_max * orig_w
                y_max_scaled = y_max * orig_h

                draw.rectangle([x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled], outline="red", width=2)
                draw.text((x_min_scaled, y_min_scaled), f"{int(label)}: {score:.2f}", fill="red")

    # Save output image
    output_path = os.path.join(output_folder, file_name)
    image.save(output_path)

    return boxes.tolist(), labels.tolist()


box_list = []
label_list = []
for file_name in os.listdir(image_folder):
    if file_name.endswith(('.jpg', '.png', '.jpeg')):  # Process only image files
        image_path = os.path.join(image_folder, file_name)
        boxes, labels = process_image(image_path, file_name)
        box_list.append(boxes)
        label_list.append(labels)
        
average_fps = sum(image_fps) / len(image_fps) if image_fps else 0
        
print("Box List",box_list)     
print("Label List",label_list)   
print(f"Average FPS: {average_fps:.2f}")