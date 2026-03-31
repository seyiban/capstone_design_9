import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2

from model import build_model
from gradcam import GradCAM, make_overlay

# ── 페이지 설정 ──────────────────────────────────────
st.set_page_config(
    page_title="PCB 결함 탐지 시스템",
    page_icon="🔍",
    layout="wide"
)

# ── 모델 로드 (캐시 적용 — 새로고침해도 재로드 안 함) ──
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model("best_model.pth", device)
    grad_cam = GradCAM(model, target_layer=model.layer4[-1])
    return model, grad_cam, device

model, grad_cam, device = load_model()

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── UI 레이아웃 ──────────────────────────────────────
st.title("PCB 결함 탐지 시스템")
st.caption("ResNet18 + Grad-CAM 기반 PCB 이미지 결함 분석")
st.divider()

uploaded = st.file_uploader(
    "PCB 이미지를 업로드하세요 (jpg, png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    # 이미지 로드
    img_pil = Image.open(uploaded).convert("RGB")
    img_np  = np.array(img_pil)

    # 전처리 및 추론
    input_tensor = val_transform(img_pil).unsqueeze(0).to(device)
    cam, probs, pred_class = grad_cam.generate(input_tensor)

    prob_normal = probs[0][0].item()
    prob_defect = probs[0][1].item()
    pred_label  = "불량" if pred_class == 1 else "정상"

    # heatmap 생성
    heatmap, overlay = make_overlay(img_np, cam)

    # ── 판정 결과 ──
    st.subheader("판정 결과")

    if pred_class == 1:
        st.error(f"불량 판정  |  신뢰도: {prob_defect:.1%}")
    else:
        st.success(f"정상 판정  |  신뢰도: {prob_normal:.1%}")

    # 신뢰도 progress bar
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption(f"정상 확률: {prob_normal:.1%}")
        st.progress(prob_normal)
    with col_b:
        st.caption(f"불량 확률: {prob_defect:.1%}")
        st.progress(prob_defect)

    st.divider()

    # ── 이미지 시각화 ──
    st.subheader("분석 결과 시각화")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("원본 이미지")
        st.image(img_np, use_container_width=True)

    with col2:
        st.caption("Grad-CAM heatmap")
        st.image(heatmap, use_container_width=True)
        st.caption("빨간 영역: 판단에 가장 큰 영향을 준 부분")

    with col3:
        st.caption("overlay")
        st.image(overlay, use_container_width=True)

    st.divider()

    # ── 상세 확률 ──
    st.subheader("상세 분석")
    st.json({
        "예측 결과": pred_label,
        "정상 확률": f"{prob_normal:.4f}",
        "불량 확률": f"{prob_defect:.4f}",
        "사용 모델": "ResNet18 (전이학습)",
        "시각화 방법": "Grad-CAM (layer4)"
    })

else:
    # 업로드 전 안내 화면
    st.info("왼쪽 업로더에서 PCB 이미지를 업로드하면 결함 분석이 시작됩니다.")
    st.markdown("""
    **분석 항목**
    - 정상 / 불량 분류
    - 신뢰도 (확률값)
    - Grad-CAM heatmap으로 결함 위치 시각화
    """)
