"""
Demo de inferência — Classificação de Pneumonia Pediátrica em Raios-X
Projeto acadêmico — Módulo 7 (PPI/SiDi)

App em Streamlit que carrega o checkpoint treinado (EfficientNet-B3) e expõe
upload de imagem. Para cada radiografia enviada, retorna:
  - classe predita (NORMAL / PNEUMONIA)
  - probabilidade associada
  - mapa Grad-CAM sobreposto à imagem original

O pré-processamento é IDÊNTICO ao usado em treino/validação/teste no
notebook original (resize 224x224 + normalização ImageNet, sem augmentation),
para evitar train-serving skew.
"""

import numpy as np
import torch
import timm
import streamlit as st
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# =============================================================
# Configuração
# =============================================================
CHECKPOINT_PATH = "app/best_efficientnet_b3.pt"   # coloque o arquivo na raiz do repo
IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # mesma ordem do ImageFolder no notebook (alfabética)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =============================================================
# Carregamento do modelo (cacheado — roda uma única vez por sessão do app)
# =============================================================
@st.cache_resource
def load_model():
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_cam(_model):
    target_layers = [_model.conv_head]  # última camada convolucional — mesma usada no notebook
    return GradCAM(model=_model, target_layers=target_layers)


def predict(model, cam_extractor, image: Image.Image):
    image = image.convert("RGB")
    input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

    pred_class = CLASS_NAMES[pred_idx]
    confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0]

    resized_img = image.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(resized_img).astype(np.float32) / 255.0
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    return pred_class, confidences, cam_image


# =============================================================
# Interface Streamlit
# =============================================================
st.set_page_config(page_title="Classificação de Pneumonia Pediátrica", layout="wide")

st.title("Classificação de Pneumonia Pediátrica em Raios-X de Tórax")
st.markdown(
    "Faça upload de uma radiografia de tórax pediátrica. O modelo "
    "(EfficientNet-B3, Transfer Learning) retorna a classificação prevista, "
    "a probabilidade associada e um mapa Grad-CAM indicando as regiões "
    "que mais influenciaram a decisão."
)
st.warning(
    "⚠️ **Projeto acadêmico — Módulo 7 (PPI/SiDi).** Este modelo NÃO "
    "constitui dispositivo médico e não deve ser usado para decisões "
    "diagnósticas reais sem supervisão profissional qualificada. "
    "Dataset: Kermany et al., *Cell* 172(5):1122–1131, 2018."
)

model = load_model()
cam_extractor = load_cam(model)

uploaded_file = st.file_uploader(
    "Radiografia de entrada (JPEG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Classificando..."):
        pred_class, confidences, cam_image = predict(model, cam_extractor, image)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Imagem enviada")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Grad-CAM")
        st.image(cam_image, use_container_width=True)

    with col3:
        st.subheader("Resultado")
        st.metric("Classe predita", pred_class)
        for cls, prob in confidences.items():
            st.write(f"**{cls}**")
            st.progress(prob, text=f"{prob:.1%}")
else:
    st.info("Aguardando upload de uma imagem.")

st.markdown("---")
st.markdown(
    "Repositório: [github.com/pauloeliezerg/pneumonia-chest-xray]"
    "(https://github.com/pauloeliezerg/pneumonia-chest-xray)"
)
