# Face Smile Classifier

This repository contains a simple Streamlit-based application that classifies whether a person in an uploaded image is smiling or not. The model used is a fine-tuned ResNet50 trained for binary classification (Smile vs. No Smile).

This project train the model using few labeled data and use the transfer learning and active learning to improve the model performance. The model is trained on the dataset from [here](https://www.kaggle.com/datasets/atulanandjha/lfwpeople?resource=download), which contains images of faces of various celebrities.

## Features

- Upload and classify images directly in the browser.
- Easy-to-use interactive interface provided by Streamlit.
- Train the model using few labeled data and use the transfer learning and active learning to improve the model performance.

## Getting Started

### Installation

Ensure you have Python installed. Then, install the dependencies:

```bash
pip install -r requirements.txt
```

### Running the App

Run the Streamlit application with the following command:

```bash
streamlit run web/layout.py
```
You could download the model weights from [here](https://huggingface.co/NEWKUN/smile_classification) and put it in the root.

Then, open the URL displayed in your terminal, typically:

```
http://localhost:8501
```

## Model Details

- **Architecture:** ResNet50
- **Task:** Binary Classification (Smile vs. No Smile)
- **Framework:** PyTorch

## Training Results

![loss Image](https://github.com/JasonTaoY/smile_classification/blob/main/assets/loss_matrix.png)
![arrucacy Image](https://github.com/JasonTaoY/smile_classification/blob/main/assets/confusion_matrix.png)

## Usage

1. Upload an image in JPG, JPEG, or PNG format.
2. The app will automatically display the uploaded image.
3. The predicted label (Smile or No Smile) will be displayed below the image.

## Example

![Example Image](https://github.com/JasonTaoY/smile_classification/blob/main/assets/Aaron_Eckhart_0001.jpg)

## Model Weights

Model weights are hosted on Hugging Face. Ensure you update the URL in the code:

```python
'https://huggingface.co/NEWKUN/smile_classification'
```

---

If you want to fine-tune the model on your own dataset, you could add the dataset in the data/dataset folder.

Enjoy classifying smiles!
