# Face Smile Classifier

This repository contains a simple Streamlit-based application that classifies whether a person in an uploaded image is smiling or not. The model used is a fine-tuned ResNet50 trained for binary classification (Smile vs. No Smile).

## Features

- Upload and classify images directly in the browser.
- Easy-to-use interactive interface provided by Streamlit.
- Uses a pre-trained ResNet50 model hosted on Hugging Face.

## Getting Started

### Installation

Ensure you have Python installed. Then, install the dependencies:

```bash
pip install streamlit torch torchvision pillow
```

### Running the App

Run the Streamlit application with the following command:

```bash
streamlit run face_smile_classifier.py
```

Then, open the URL displayed in your terminal, typically:

```
http://localhost:8501
```

## Model Details

- **Architecture:** ResNet50
- **Task:** Binary Classification (Smile vs. No Smile)
- **Framework:** PyTorch

## Training Results

The model achieved high accuracy and a stable loss curve. The detailed metrics:

| Metric           | Score       |
|------------------|-------------|
| Training Loss    | [Insert Loss Curve Image Here] |
| Validation Accuracy | [Insert Accuracy Image Here] |

*(Replace placeholders above with your actual metrics images.)*

## Usage

1. Upload an image in JPG, JPEG, or PNG format.
2. The app will automatically display the uploaded image.
3. The predicted label (Smile or No Smile) will be displayed below the image.

## Example

![Example Image](path_to_example_result_image.png)

## Model Weights

Model weights are hosted on Hugging Face. Ensure you update the URL in the code:

```python
'https://huggingface.co/yourusername/yourmodelname/resolve/main/model_final.pth'
```

Replace `yourusername` and `yourmodelname` with your actual Hugging Face repository details.

---

Enjoy classifying smiles!
