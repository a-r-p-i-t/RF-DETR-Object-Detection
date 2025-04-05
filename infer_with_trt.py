import os
import csv
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image, ImageDraw
from torchvision import transforms
import time

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load TensorRT Engine
def load_engine(trt_model_path):
    with open(trt_model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

trt_model_path = "model.trt"
engine = load_engine(trt_model_path)

binding_name = engine[0]

# Allocate buffers
context = engine.create_execution_context()
# input_shape = engine.get_binding_shape(0)
# input_shape = engine.get_tensor_shape(engine.get_binding_name(0))
input_shape = engine.get_tensor_shape(binding_name)

print("Input Shape:",input_shape)

# Allocate memory
input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
d_input = cuda.mem_alloc(input_size)

# output_shapes = [engine.get_binding_shape(i) for i in range(1, engine.num_bindings)]
output_shapes = [engine.get_tensor_shape(engine.get_tensor_name(i)) for i in range(engine.num_io_tensors)]
print("Output Shapes:",output_shapes)
output_sizes = [trt.volume(shape) * np.dtype(np.float32).itemsize for shape in output_shapes]
d_outputs = [cuda.mem_alloc(size) for size in output_sizes]

# Stream for CUDA
stream = cuda.Stream()

# Image Processing
transform = transforms.Compose([
    transforms.Resize((560, 560)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_folder = "/home/arpit/rf_detr/split_dataset/test/"
output_folder = "/home/arpit/rf_detr/split_dataset/predictions_trt/"
csv_filename = "valid_box_counts_with_categories_trt.csv"

os.makedirs(output_folder, exist_ok=True)

image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]

image_category_counts = {}
image_fps = []

def process_image(image_path, file_name):
    start_time = time.time()
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    image_tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)

    # Copy to device
    cuda.memcpy_htod_async(d_input, image_tensor, stream)
    
    # input_binding_name = engine.get_binding_name(0)
    
    # context.set_tensor_address(0, int(d_input))
    context.set_tensor_address(binding_name, int(d_input))

    # Execute model
    bindings = [int(d_input)] + [int(out) for out in d_outputs]
    context.execute_async_v3(stream_handle=stream.handle)

    # Copy output back to host
    host_outputs = [np.empty(shape, dtype=np.float32) for shape in output_shapes]
    for host, dev in zip(host_outputs, d_outputs):
        cuda.memcpy_dtoh_async(host, dev, stream)

    stream.synchronize()

    _, boxes, class_logits = host_outputs  # Get model outputs

    end_time = time.time()
    processing_time = end_time - start_time
    fps = 1 / processing_time if processing_time > 0 else 0
    image_fps.append(fps)

    boxes = torch.tensor(boxes, device="cuda")  # (1, 300, 4)
    class_logits = torch.tensor(class_logits, device="cuda")

    class_probs = torch.nn.functional.softmax(class_logits[:, :, 1:], dim=-1)
    scores, labels = class_probs.max(dim=-1)

    confidence_threshold = 0.85
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
            if label == 0:
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
    if file_name.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_folder, file_name)
        category_counts = process_image(image_path, file_name)
        image_category_counts[file_name] = category_counts
        # break

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

average_fps = sum(image_fps) / len(image_fps) if image_fps else 0
print(f"Average FPS: {average_fps:.2f}")

with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)

print(f"CSV file '{csv_filename}' created successfully!")
