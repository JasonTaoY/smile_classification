import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F


@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    state_dict = torch.hub.load_state_dict_from_url(
        'https://huggingface.co/yourusername/yourmodelname/resolve/main/model_final.pth',
        map_location='cpu',
        progress=True
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title('Smile Classification App')

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)

    label = "Smile" if pred.item() == 1 else "No Smile"
    st.subheader(f"Prediction: {label}")
