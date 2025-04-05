import os
import csv
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import time

device = "cpu"
print(f"Using device: {device}")


onnx_model_path = "inference_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

providers = ort.get_available_providers()

transform = transforms.Compose([
    transforms.Resize((560, 560)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_folder = "/home/arpit/rf_detr/split_dataset/test/"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]
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
    processing_time = end_time - start_time  
    
    fps = 1 / processing_time if processing_time > 0 else 0
    image_fps.append(fps)

    boxes = torch.tensor(ort_outputs[0]) 
    class_logits = torch.tensor(ort_outputs[1])
    
    confidence_threshold = 0.5
    
    if class_logits.shape[-1] != 1:
        
        class_probs = torch.nn.functional.softmax(class_logits[:, :, 1:], dim=-1)
        scores, labels = class_probs.max(dim=-1)
        confident_indices = scores > confidence_threshold
        boxes = boxes[confident_indices].numpy()
        scores = scores[confident_indices].numpy()
        labels = labels[confident_indices].numpy()
        
        
    else:
        
        scores = torch.nn.functional.sigmoid(class_logits[:, :, 0])
        labels = torch.zeros_like(scores, dtype=torch.int)
        confident_indices = scores > confidence_threshold
        boxes = boxes[confident_indices].numpy()
        scores = scores[confident_indices].numpy()
        labels = labels[confident_indices.numpy()]

    boxes[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2)
    boxes[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    
    boxes = boxes.tolist()
    scores = scores.tolist()
    labels = labels.tolist()


    
    if len(boxes) > 0:  
        abs_boxes = []
        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            x_min_scaled = x_min * orig_w
            y_min_scaled = y_min * orig_h
            x_max_scaled = x_max * orig_w
            y_max_scaled = y_max * orig_h
            abs_boxes.append([x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled])
        return abs_boxes,labels
    else:  
        return [], []

box_list = []
label_list = []
for file_name in os.listdir(image_folder):
    if file_name.endswith(('.jpg', '.png', '.jpeg')): 
        image_path = os.path.join(image_folder, file_name)
        boxes, labels = process_image(image_path, file_name)
        box_list.append(boxes)
        label_list.append(labels)
        
average_fps = sum(image_fps) / len(image_fps) if image_fps else 0
        
print("Box List",box_list)     
print("Label List",label_list)   
print(f"Average FPS: {average_fps:.2f}")


