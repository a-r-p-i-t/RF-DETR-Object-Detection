# RF-DETR Object Detection

## **Overview**

This project impements the RF-Detr Model for Training Custom Object Detection Model and Inferencing Uisng Onnx for real-time applications

## Table of Contents

- [Objective](#objective)
- [Installation](#installation)
- [Data Description](#data-description)
- [Fine-Tuning](#fine-tuning)
- [Early Stopping](#early-stopping)
- [Logging with TensorBoard](#logging-with-tensorboard)
- [Logging with Weights and Biases](#logging-with-weights-and-biases)
- [Load and run fine-tuned model](#load-and-run-fine-tuned-model)
- [ONNX Export](#onnx-export)
- [BenchMarking (With MLFlow Integration)](#benchmarking-with-mlflow-integration)


## Objective
Implementing Custom Object Detection Model (Training and Infernence) using RF-DETR for Real time applications. 

## Installation

1. **Pip install the rfdetr package in a Python>=3.9 environment.:**

   ```bash
   Pip install the rfdetr
   ```

2. **Additional Installation:**

    ```bash
    pip install git+https://github.com/roboflow/rf-detr.git
    ```
    

## Data Description

RF-DETR expects the dataset to be in COCO format. Divide your dataset into three subdirectories: train, valid, and test. Each subdirectory should contain its own _annotations.coco.json file that holds the annotations for that particular split, along with the corresponding image files. Below is an example of the directory structure:

  ```bash
  dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    ├── image2.jpg
    └── ... (other image files)
```
## Fine-Tuning
You can fine-tune RF-DETR from pre-trained COCO checkpoints. By default, the RF-DETR-B checkpoint will be used.

 ```bash
from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(dataset_dir=<DATASET_PATH>, epochs=10, batch_size=4, grad_accum_steps=4, lr=1e-4, output_dir=<OUTPUT_PATH>)
```
**More Parameters**
![Screenshot 2025-04-05 193856](https://github.com/user-attachments/assets/ef8f9a7b-742c-4f83-8887-00eba92bd7d2)

![Screenshot 2025-04-05 193912](https://github.com/user-attachments/assets/a278cf79-8d8a-4ab3-b271-e0c17bd975c1)

## Resume Training
You can resume training from a previously saved checkpoint by passing the path to the checkpoint.pth file using the resume argument. This is useful when training is interrupted or you want to continue fine-tuning an already partially trained model. The training loop will automatically load the weights and optimizer state from the provided checkpoint file.

 ```bash
from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(dataset_dir=<DATASET_PATH>, epochs=10, batch_size=4, grad_accum_steps=4, lr=1e-4, output_dir=<OUTPUT_PATH>, resume=<CHECKPOINT_PATH>)
 ```
## Early Stopping
Early stopping monitors validation mAP and halts training if improvements remain below a threshold for a set number of epochs. This can reduce wasted computation once the model converges. Additional parameters—such as early_stopping_patience, early_stopping_min_delta, and early_stopping_use_ema—let you fine-tune the stopping behavior
```bash
from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(dataset_dir=<DATASET_PATH>, epochs=10, batch_size=4, grad_accum_steps=4, lr=1e-4, output_dir=<OUTPUT_PATH>, early_stopping=True)
```
**Result Checkpoints**
During training, two model checkpoints (the regular weights and an EMA-based set of weights) will be saved in the specified output directory. The EMA (Exponential Moving Average) file is a smoothed version of the model’s weights over time, often yielding better stability and generalization.

## Logging with TensorBoard
TensorBoard is a powerful toolkit that helps you visualize and track training metrics. With TensorBoard set up, you can train your model and keep an eye on the logs to monitor performance, compare experiments, and optimize model training. To enable logging, simply pass tensorboard=True when training the model.

**Using Tensorboard with RF-detr**
TensorBoard logging requires additional packages. Install them with:

```bash
pip install "rfdetr[metrics]"
```
To activate logging, pass the extra parameter tensorboard=True to .train():

```bash
from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    dataset_dir=<DATASET_PATH>,
    epochs=10,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=<OUTPUT_PATH>,
    tensorboard=True
)
```

To use TensorBoard locally, navigate to your project directory and run:

```bash
tensorboard --logdir <OUTPUT_DIR>
```
Then open http://localhost:6006/ in your browser to view your logs

## Logging with Weights and Biases

Weights and Biases (W&B) is a powerful cloud-based platform that helps you visualize and track training metrics. With W&B set up, you can monitor performance, compare experiments, and optimize model training using its rich feature set. To enable logging, simply pass wandb=True when training the model.

**Using Weights and Biases with RF-Detr**

Weights and Biases logging requires additional packages. Install them with:

```bash
pip install "rfdetr[metrics]"
```

Before using W&B, make sure you are logged in:

```bash
wandb login
```
You can retrieve your API key at wandb.ai/authorize.

**To activate logging, pass the extra parameter wandb=True to .train():**
```bash
from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    dataset_dir=<DATASET_PATH>,
    epochs=10,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=<OUTPUT_PATH>,
    wandb=True,
    project=<PROJECT_NAME>,
    run=<RUN_NAME>
)
```

In W&B, projects are collections of related machine learning experiments, and runs are individual sessions where training or evaluation happens. If you don't specify a name for a run, W&B will assign a random one automatically.

## Load and run fine-tuned model

```bash
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)

detections = model.predict(<IMAGE_PATH>)
```

## ONNX Export

RF-DETR supports exporting models to the ONNX format, which enables interoperability with various inference frameworks and can improve deployment efficiency. To export your model, simply initialize it and call the .export() method.

```bash
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)

model.export()
```
This command saves the ONNX model to the output directory.

**Installing onnxruntime for inference using exported onnx model**
**For Running Inference On cpu: Run**

```bash
pip install onnxruntime
```

**For Running Inference on GPU**
```bash
pip install onnxruntime-gpu
```

## BenchMarking (With MLFlow Integration)

**Installation**
```bash
pip install mlflow
```

**Generating Ground Truth Category wise bbox count**

Run the script **[ground_truth_bbox_count.py](./ground_truth_bbox_count.py)** to get Ground truth csv file containing category wise bbox count

**Required Parameters:**

1. <test_json_path> (__annotations.coco.json)
2. <csv_file_path> where you want to have the category wise bbox count data appended.

**MLFlow Integrated Evaluation Metrics with prediction results**

Run the script **[inf_with_mlflow.py](./inf_with_mlflow.py)** to get

1. Predictions Saved in Saved Dir.
2. Csv file containing the category-wise predicted box counts
3. Average Fps of processing of frames by the model
4. ML flow integrated logs saved in mlruns folder, containing the accurcy metrics

**Required Parameters:**

1. <test_folder_path>
2. <saved_dir_path> where you want to sve the model predictions
3. <csv_file_path> where the category wise box counts info will be appended.
4. <confidence_threshold> for model
5. <ground_truth_csv_path> containing category-wise bbox-count

   























