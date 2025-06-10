# Image Classification Exercice

## Preliminary analysis

*   **Dataset:** Composed of 64x64 pixel images of human faces.
*   **Hypothesized Classes:** Manual inspection suggests:
    *   Class 0: Person with an accessory (e.g., hat, glasses, cap).
    *   Class 1: Person without an accessory.
    *   This is an initial hypothesis, and some images appear contradictory.
*   **Imbalance:** The dataset is highly imbalanced, with Class 1 (hypothesized "without accessory") much more prevalent
*   **Similarity:** The dataset bears resemblance to publicly available datasets like CelebA.
Dataset looks similar to 'Microsoft Celeb'.


Two primary model architectures were explored:

1.  **Custom SimpleCNN:**
    *   A foundational CNN architecture designed for this task.
    *   **Architecture Overview:**
        *   Input: 3x64x64
        *   Block 1: Conv2d(3, 32, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(k=2, s=2)
        *   Block 2: Conv2d(32, 64, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(k=2, s=2)
        *   Block 3: Conv2d(64, 128, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(k=2, s=2)
        *   Flatten
        *   FC1: Linear(8192, 128) -> BatchNorm1d -> ReLU -> Dropout(0.5)
        *   Output: Linear(128, 1) (for binary classification with sigmoid)
    *   This architecture was chosen for its simplicity and as a baseline to understand the dataset's learnability.

2.  **MobileNetV2 (Transfer Learning):**
    *   **Why ?:** Chosen for its balance of lightweight architecture (good for limited compute resources like Google Colab free tier) and strong performance on general image recognition tasks (pre-trained on ImageNet).
    *   **Modes Explored:**
        *   **Feature Extraction:** Only the final classifier layer was replaced and trained, keeping pre-trained weights frozen.
        *   **Full Fine-tuning:** All layers of the MobileNetV2 model were unfrozen and trained with a reduced learning rate.


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

The primary metric for evaluation is the Half-Total Error Rate (HTER): `HTER = ((1 - Recall_Class_0) + (1 - Recall_Class_1)) / 2`.

| Experiment                                        | Recall Class 0 | Recall Class 1  | HTER   | Notes                                       |
| :------------------------------------------------ | :----------------------- | :-------------------- | :----- | :------------------------------------------ |
| SimpleCNN (40 epochs)                             | 0.6740                   | 0.9809                | 0.1725 | Baseline custom architecture.               |
| SimpleCNN (40 epochs, Weighted Loss)              | 0.7079                   | 0.9791                | 0.1565 | Weighted loss improved minority class recall. |
| MobileNetV2 (Classifier Fine-tuned)             | 0.7496                   | 0.9722                | 0.1391 | Significant improvement over SimpleCNN.     |
| MobileNetV2 (Full Fine-tuning)                    | 0.7116                   | 0.9778                | 0.1553 | Slightly underperformed classifier-only FT.   |
| MobileNetV2 (Classifier Fine-tuned,Weighted Loss)                    | 0.7074         | 0.7246                |  0.2840 | Didn't performed well at all.   |
| **MobileNetV2 (Full Fine-tuned, Weighted Loss)** | **0.9207**               | **0.9094**            | **0.0850** | **Best performing model.**                  |

**Observations:**

*   **Transfer Learning Advantage:** MobileNetV2, even when only fine-tuning the classifier head, has better performance over the custom SimpleCNN. This is likely due to the robust, general-purpose features learned from pre-training on ImageNet. Training the head was also relatively fast (converging in ~15-20 epochs).
*   **Impact of Weighted Loss:** Applying class weighting consistently improved the HTER by significantly boosting the recall of the minority class often with a minor, acceptable trade-off in majority class recall or precision.
*   **Full Fine-tuning vs. Classifier-Only:** Initially, classifier only fine tuning (HTER 0.1391) outperformed full fine-tuning (HTER 0.1553). Fine-tuning all layers on a smaller dataset risks overfitting or disrupting the valuable pre-trained features. However, the true power of full fine-tuning was observed when combined with a weighted loss. In this scenario, allowing the entire network to adapt enabled the model to learn more specific features for the minority class, leading to the best overall HTER (0.0850). The classifier-only model, in contrast could not sufficiently adapt its fixed feature representations to the strong signal from the weighted loss.
*   The combination of **MobileNetV2 (classifier full-tuned) with weighted loss provided the lowest HTER (0.0850)**, indicating indicating that giving the model maximum flexibility (all layers trainable) and a clear objective (weighted loss for class balance) was the most effective strategy


