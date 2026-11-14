import os

import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("Digit Recognizer")

# Sidebar for API URL (docker-compose sets API_URL to the backend container)
api_url = st.sidebar.text_input(
    "FastAPI URL", os.getenv("API_URL", "http://localhost:8000")
).rstrip("/")

# Canvas for drawing
drawing_mode = "freedraw"
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode=drawing_mode,
    key="canvas",
)


# Preprocess the image for prediction
def preprocess(img_data):
    if img_data is None:
        return None

    # Convert to grayscale by taking the mean of RGB channels
    img = np.mean(img_data[:, :, :3], axis=2)

    # Nothing drawn yet
    if img.max() == 0:
        return None

    # Normalize to [0, 1]
    img = img / 255.0

    # Convert to PIL Image for proper resizing
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized) / 255.0

    return img_array.tolist()


def call_api(method, path, **kwargs):
    try:
        response = requests.request(method, f"{api_url}{path}", timeout=10, **kwargs)
    except requests.ConnectionError:
        st.error(f"Can't reach the API at {api_url}. Is the backend running?")
        return None
    if response.status_code != 200:
        st.error(f"Request failed: {response.text}")
        return None
    return response.json()


img_array = preprocess(canvas_result.image_data)

# Streamlit reruns this script on every click, so the latest prediction is kept
# in session state rather than in a local variable
if st.button("Predict"):
    if img_array is None:
        st.warning("Draw a digit first.")
    else:
        with st.spinner("Predicting..."):
            result = call_api("POST", "/predict", json={"image": img_array})
        if result is not None:
            st.session_state["result"] = result

result = st.session_state.get("result")
if result is not None:
    st.markdown(f"**Prediction:** {result['prediction']}")
    st.markdown(f"**Confidence:** {result['confidence']*100:.2f}%")

    # True label input (keyed by prediction id so it resets for each new prediction)
    true_label = st.number_input(
        "True label (0-9)",
        min_value=0,
        max_value=9,
        step=1,
        value=result["prediction"],
        key=f"true_label_{result['id']}",
    )

    if st.button("Submit"):
        saved = call_api(
            "PATCH",
            f"/predictions/{result['id']}",
            json={"true_label": int(true_label)},
        )
        if saved is not None:
            st.success("Saved.")

# History table (latest 50 predictions from the database)
history = call_api("GET", "/history")
if history:
    st.markdown("### History")
    st.table(
        [
            {
                "Time (UTC)": row["timestamp"][:19].replace("T", " "),
                "Predicted": row["predicted_digit"],
                "True label": "" if row["true_label"] is None else row["true_label"],
            }
            for row in history
        ]
    )
