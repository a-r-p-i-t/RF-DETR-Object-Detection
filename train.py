# from rfdetr import RFDETRBase

# DATASET_PATH = "/home/arpit/rf_detr/split_dataset"

# OUTPUT_PATH = "/home/arpit/rf_detr/trained_output"


# model = RFDETRBase()


# model.train(dataset_dir=DATASET_PATH, epochs=100, batch_size=4, grad_accum_steps=4, lr=1e-4, output_dir=OUTPUT_PATH)




import mlflow
import mlflow.pytorch
import sys
import io
import re
from rfdetr import RFDETRBase

# Define paths
DATASET_PATH = "/home/arpit/rf_detr/split_dataset"
OUTPUT_PATH = "/home/arpit/rf_detr/trained_output"

# Initialize MLflow experiment
mlflow.set_experiment('RFDETR_Training')  # You can name your experiment here

# Create model
model = RFDETRBase()

# Define training parameters
params = {
    "epochs": 100,
    "batch_size": 4,
    "grad_accum_steps": 4,
    "lr": 1e-4,
    "output_dir": OUTPUT_PATH
}

# Function to parse printed log for metrics (loss and class_error)
def parse_training_log(log_output):
    # Regular expression to match the loss and class_error values in the log
    loss_pattern = r'loss: ([\d\.]+)'
    class_error_pattern = r'class_error: ([\d\.-]+)'
    
    # Search for loss and class_error in the log
    loss_match = re.search(loss_pattern, log_output)
    class_error_match = re.search(class_error_pattern, log_output)
    
    loss = float(loss_match.group(1)) if loss_match else None
    class_error = float(class_error_match.group(1)) if class_error_match else None
    
    return loss, class_error

# Custom logger to print logs to the terminal and capture for MLflow
class StreamToLogger(io.StringIO):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def write(self, message):
        # Print message to the terminal (stdout)
        sys.__stdout__.write(message)
        
        # Capture and log the message to MLflow
        if message.strip():  # Avoid empty messages
            self.logger.write(message)

# Start an MLflow run
with mlflow.start_run():

    # Log hyperparameters
    mlflow.log_params(params)

    # Create a logger to capture and print logs
    captured_output = io.StringIO()
    logger = StreamToLogger(captured_output)

    # Start training
    for epoch in range(params['epochs']):
        print(f"Epoch {epoch+1}/{params['epochs']}")

        # Redirect stdout to capture the printed log and also log it in MLflow
        sys.stdout = logger

        # Train the model (assuming train() prints logs)
        model.train(dataset_dir=DATASET_PATH,
                    epochs=params['epochs'], 
                    batch_size=params['batch_size'], 
                    grad_accum_steps=params['grad_accum_steps'], 
                    lr=params['lr'], 
                    output_dir=params['output_dir'])
        
        # Reset stdout to the original state
        sys.stdout = sys.__stdout__

        # Get the log output captured
        log_output = captured_output.getvalue()

        # Parse the log output for loss and class_error
        epoch_loss, class_error = parse_training_log(log_output)

        # Log the metrics automatically to MLflow
        if epoch_loss is not None and class_error is not None:
            mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
            mlflow.log_metric("class_error", class_error, step=epoch)
        
        print(f"Epoch {epoch+1} - Loss: {epoch_loss}, Class Error: {class_error}")
    
    # After training, log the model artifact
    mlflow.pytorch.log_model(model, "model")
