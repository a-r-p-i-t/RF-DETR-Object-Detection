import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import os
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

onnx_model_path = "inference_model_fp16.onnx"
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

image_size = 560
os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists

image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]
image_category_counts = {}
image_fps = []

for image_path in image_paths:
    print(f"Processing: {image_path}")

    # Load image
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)

    start_time = time.time()
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: image_tensor.cpu().numpy().astype(np.float32)}

    ort_outputs = ort_session.run(None, ort_inputs)
    
    end_time = time.time()
    processing_time = end_time - start_time  # Time taken to process the image
    
    fps = 1 / processing_time if processing_time > 0 else 0
    image_fps.append(fps)

    boxes = torch.tensor(ort_outputs[0], device=device)  # (1, 300, 4)
    class_logits = torch.tensor(ort_outputs[1], device=device)

    class_probs = torch.nn.functional.softmax(class_logits[:, :, 1:], dim=-1)
    scores, labels = class_probs.max(dim=-1)

    confidence_threshold = 0.85
    confident_indices = scores > confidence_threshold

    boxes = boxes[confident_indices].cpu().numpy()
    scores = scores[confident_indices].cpu().numpy()
    labels = labels[confident_indices].cpu().numpy()
    
   
    
    boxes[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2)  
    boxes[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2)  
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2] 
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    
    
    width_scale = orig_w
    height_scale = orig_h
    
    if len(boxes) >0:
        draw = ImageDraw.Draw(image)
        for box, score, label in zip(boxes, scores, labels):
            if label == 0:
                x_min, y_min, x_max, y_max = box
                x_min_scaled = x_min * width_scale
                y_min_scaled = y_min * height_scale
                x_max_scaled = x_max * width_scale
                y_max_scaled = y_max * height_scale
                
                draw.rectangle([x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled], outline="red", width=2)
                draw.text((x_min_scaled, y_min_scaled), f"{label}: {score:.2f}", fill="red")
                
                category_counts[label] = category_counts.get(label, 0) + 1

        output_path = os.path.join(output_folder, os.path.basename(image_path))
        image.save(output_path)
        image_category_counts[file_name] = category_counts
        print(f"Saved: {output_path}")
    else:
        print(f"No objects detected in: {image_path}")
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        image.save(output_path)
        
    
    # break
average_fps = sum(image_fps) / len(image_fps) if image_fps else 0
print(f"Average FPS: {average_fps:.2f}")