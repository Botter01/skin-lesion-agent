import streamlit as st
import os
import json
import time
import tempfile
from dotenv import load_dotenv
from PIL import Image

from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

load_dotenv()

PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT")
AGENT_ID = os.getenv("AGENT_ID")
CV_PREDICTION_ENDPOINT = os.getenv("CV_PREDICTION_ENDPOINT")
CV_PREDICTION_KEY = os.getenv("CV_PREDICTION_KEY")
CV_PROJECT_ID = os.getenv("CV_PROJECT_ID")
CV_PUBLISHED_NAME = os.getenv("CV_PUBLISHED_NAME")

st.set_page_config(page_title="Skin Lesion Educational Assistant", page_icon="🔬")

DX_FULL_NAMES = {
    "nv": "melanocytic nevus (common mole, generally benign)",
    "mel": "melanoma (a serious form of skin cancer)",
    "bkl": "benign keratosis-like lesion",
    "bcc": "basal cell carcinoma (a common, usually slow-growing skin cancer)",
    "akiec": "actinic keratosis / intraepithelial carcinoma (a precancerous lesion)",
    "vasc": "vascular lesion (e.g. angioma)",
    "df": "dermatofibroma (a benign skin growth)"
}

@st.cache_resource
def get_clients():
    agents_client = AgentsClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())
    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": CV_PREDICTION_KEY})
    predictor = CustomVisionPredictionClient(CV_PREDICTION_ENDPOINT, prediction_credentials)
    return agents_client, predictor

agents_client, predictor = get_clients()

def analyze_skin_lesion_image(image_path: str) -> str:
    with open(image_path, "rb") as image_data:
        results = predictor.classify_image(CV_PROJECT_ID, CV_PUBLISHED_NAME, image_data.read())
    predictions = sorted(results.predictions, key=lambda p: p.probability, reverse=True)[:3]
    output = [{
        "category_code": p.tag_name,
        "category_full_name": DX_FULL_NAMES.get(p.tag_name, p.tag_name),
        "confidence": round(p.probability * 100, 1)
    } for p in predictions]
    return json.dumps(output)

def run_agent(image_path, age, location, duration, changed):
    thread = agents_client.threads.create()

    user_message = f"""
I have a skin lesion I'd like to understand better. Here's some context:
- Age: {age}
- Location on body: {location}
- Present for approximately: {duration}
- Has it changed recently: {changed}

Please analyze the image at this path: {image_path}
"""
    agents_client.messages.create(thread_id=thread.id, role="user", content=user_message)
    run = agents_client.runs.create(thread_id=thread.id, agent_id=AGENT_ID)

    while run.status in ["queued", "in_progress", "requires_action"]:
        time.sleep(1.5)
        run = agents_client.runs.get(thread_id=thread.id, run_id=run.id)

        if run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tool_call in tool_calls:
                if tool_call.function.name == "analyze_skin_lesion_image":
                    args = json.loads(tool_call.function.arguments)
                    result = analyze_skin_lesion_image(**args)
                    tool_outputs.append({"tool_call_id": tool_call.id, "output": result})
            run = agents_client.runs.submit_tool_outputs(
                thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
            )

    if run.status == "failed":
        return f"Error: {run.last_error}"

    messages = list(agents_client.messages.list(thread_id=thread.id))
    for msg in messages:
        if msg.role == "assistant" and msg.content:
            return msg.content[0].text.value
    return "No response received."

# --- UI ---
st.title("🔬 Skin Lesion Educational Assistant")
st.warning(
    "⚠️ This is an educational demo only, NOT a medical device or diagnostic tool. "
    "It does not provide medical advice. Always consult a qualified dermatologist "
    "or doctor for any concerns about your skin."
)

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    location = st.text_input("Location on body", placeholder="e.g. forearm, back, scalp")
with col2:
    duration = st.text_input("How long present?", placeholder="e.g. 6 months")
    changed = st.selectbox("Has it changed recently?", ["No noticeable change", "Yes, in size", "Yes, in color", "Yes, in shape", "Not sure"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=300)

if st.button("Analyze", type="primary", disabled=(uploaded_file is None)):
    with st.spinner("Analyzing image and consulting the assistant..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            response = run_agent(tmp_path, age, location, duration, changed)
            st.success("Analysis complete")
            st.markdown(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            os.unlink(tmp_path)