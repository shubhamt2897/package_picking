import os
from ultralytics import YOLO
###########################################################################
###########################################################################
##### Orignally the model was trained in google collab with the following code, due to 
##### the limitations of local machine.
###########################################################################
###########################################################################



# --- 1. Define Local Paths (USER MUST EDIT THESE) ---
# Set the path to the root directory of your downloaded dataset.
# This directory should contain the 'data.yaml' file.
LOCAL_DATASET_PATH = '/path/to/your/dataset/root' 

# Set the path to a previously trained model to continue training from.
PREVIOUS_BEST_MODEL_PATH = '/path/to/your/previous/runs/train/weights/best.pt' 

# --- 2. Construct the Dataset YAML Path ---
# This assumes the 'data.yaml' is directly inside your dataset's root folder.
DATASET_YAML_PATH = os.path.join(LOCAL_DATASET_PATH, 'data.yaml')

# --- 3. Load Your Model ---
if PREVIOUS_BEST_MODEL_PATH and os.path.exists(PREVIOUS_BEST_MODEL_PATH):
    print(f"Loading model from '{PREVIOUS_BEST_MODEL_PATH}' to continue training...")
    model = YOLO(PREVIOUS_BEST_MODEL_PATH)
else:
    print("Previous model not found or not specified. Starting from scratch with 'yolov8s-seg.pt'.")
    # Load a pretrained segmentation model (e.g., yolov8s-seg.pt, yolov8m-seg.pt)
    model = YOLO('yolov8s-seg.pt')

# --- 4. Start or Continue Training on the Local Dataset ---
print(f"Starting training using dataset config: {DATASET_YAML_PATH}")

results = model.train(
    data=DATASET_YAML_PATH,
    epochs=200,
    imgsz=640,
    patience=20,
    flipud=0.5,
    fliplr=0.5,
    optimizer='AdamW',
    lr0=0.0005
)

# --- 5. Get the Final Model Path ---
# The results object contains the path to the directory where results are saved.
FINAL_BEST_MODEL_PATH = os.path.join(results.save_dir, 'weights/best.pt')

print(f"\nTraining complete. Your newly improved model is saved at: {FINAL_BEST_MODEL_PATH}")
