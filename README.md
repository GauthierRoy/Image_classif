# Image Classification Exercice
## Setup and Execution

1.  **Prerequisites:**
    *   Access to Google Colab (or a local environment with gpu and PyTorch, torchvision, scikit-learn, pandas, numpy, matplotlib, seaborn, tqdm).
    *   Google Drive.

2.  **Data Preparation:**
    *   Ensure the `ml_exercise_therapanacea.zip` file (with the internal structure described above) is placed in your Google Drive under a path that will be set as `TARGET_WORKING_DIR` in the notebook.
    *   Example: Create a folder `Deep/thera` in your `MyDrive` and place `ml_exercise_therapanacea.zip` inside it.

3.  **Code Files:**
    *   Place `main_trainer.ipynb` and `model_utils.py` in the same `TARGET_WORKING_DIR` in your Google Drive.

4.  **Running the Notebook (`main_trainer.ipynb`):**
    *   Open `main_trainer.ipynb` in Google Colab.
    *   **Cell 1 (Google Drive Mount):** Execute to mount your Google Drive.
    *   **Cell 2 (Path Configuration):**
        *   **Crucially, update `TARGET_WORKING_DIR`** to match the path in your Google Drive where you placed the data and code files (e.g., `"/content/drive/MyDrive/Deep/thera"`).
        *   Model Initialization: The notebook provides sections for either a **Simple Custom CNN (Section A)** or **Transfer Learning with MobileNetV2 (Section B)**. Execute only one of these sections to select your model.
            *   For MobileNetV2, you can configure `FINE_TUNE_ALL_LAYERS` (True/False) and `FINE_TUNE_LR_FACTOR`.
        *   Train Model.
        *   Plot Training History and Evaluate.
        *   Predict on Validation Images: This will generate `label_val.txt` in `/content/face_classification_data/`.

5.  **Output:**
    *   The primary output for the exercise is `label_val.txt`, which will be created in `/content/face_classification_data/` within the Colab environment after running the prediction cell. You will need to download this file.
    *   Trained model weights (ex `Mobile_Net_full_finetuning.pth`) are saved in `TARGET_WORKING_DIR` on your Google Drive.

## Model Details

The notebook allows experimentation with two main approaches:

1.  **SimpleCNN:** A custom-built Convolutional Neural Network defined in `model_utils.py`.
2.  **MobileNetV2 (Transfer Learning):** Uses a MobileNetV2 model pre-trained on ImageNet.
    *   **Feature Extraction:** Freezes pretrained layers and trains only a new classifier head.
    *   **Fine-Tuning:** Unfreezes all pretrained layers and trains them with a smaller learning rate along with the new classifier head.

Class imbalance can optionally be handled using `pos_weight` in the `BCEWithLogitsLoss` function.

## Results


**Example (Fill with your actual best results):**

**Observations:**

