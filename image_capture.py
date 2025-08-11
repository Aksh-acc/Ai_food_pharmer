import cv2
import os
from datetime import datetime
import streamlit as st
def capture_image(save_dir="captured_images"):
    """
    Captures an image from the default webcam and saves it as a PNG.

    Args:
        save_dir (str): Directory where captured image will be stored.

    Returns:
        str: File path of the captured image.

    """
    enable = st.checkbox("Enable camera")
    picture = st.camera_input("Capture Image" , disabled=not enable)
    if picture:
        st.image(picture)
    filename = f"label_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    captured_path = os.path.join(save_dir, filename)
    print(captured_path)

    return captured_path
    return captured_path
