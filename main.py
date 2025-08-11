from io import BytesIO
import requests
from autogen_agentchat.messages import MultiModalMessage , TextMessage
from autogen_core import Image as AGImage
from PIL import Image
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination  ,MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
import os
import image_capture
from dotenv import load_dotenv
from tavily import TavilyClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
import streamlit as st

# Load environment variables
load_dotenv()
print("DEBUG: Environment variables loaded.")

# Tavily search client
tavily_client = TavilyClient(api_key=os.getenv("tavily_api_key"))

# Streamlit page setup
st.set_page_config(page_title="Your Food Pharmer", page_icon="🤖")
st.title("Personal healthy doctor")
print("DEBUG: Streamlit app initialized.")

# --- TOOL FUNCTIONS ---
def web_search_tool(query: str) -> str:
    """Perform a web search using Tavily API."""
    print(f"DEBUG: Performing web search for query: '{query}'")
    response = tavily_client.search(query=query, num_results=2)
    if response and response.results:
        print(f"DEBUG: Web search found {len(response.results)} results.")
        return "\n".join([f"{i+1}. {result.title} - {result.url}" for i, result in enumerate(response.results)])
    print("DEBUG: Web search found no results.")
    return "No results found."

def image_processing_tool(image_path: str) -> MultiModalMessage:
    """Process the captured image and return a MultiModalMessage."""
    print(f"DEBUG: Attempting to process image from path: {image_path}")
    try:
        pil_image = Image.open(image_path)
        img = AGImage(pil_image)
        print("DEBUG: Image opened successfully. Creating MultiModalMessage.")
        return MultiModalMessage(
            content=["Can you describe the content of this image?", img],
            source="User"
        )
    except FileNotFoundError:
        print(f"ERROR: Image file not found at path: {image_path}")
        st.error(f"Error: Image file not found at {image_path}")
        return None

# --- MODEL CLIENT ---
client = OpenAIChatCompletionClient(
    model="gemini-1.5-pro",
    api_key=os.getenv("GEMINI_API_KEY")
)
print("DEBUG: OpenAI client initialized.")

# --- AGENTS ---
planning_agent = AssistantAgent(

    "PlanningAgent",

    description="Plans tasks and delegates them to other agents.",

    model_client=client,

    system_message="""
    You are a planning agent.
    Your job is to break down complex tasks into smaller, manageable subtasks.
    Your team members are:
        - IMAGE_PROCESSING_AGENT: An advanced image preprocessing AI specialized in food packaging and nutrition labels.
        - Insights_Extraction_Agent: A food science and nutrition insights AI.
        - default_food_pharmer_agent: An evidence-based nutrition and food science expert.
    You only plan and delegate tasks - you do not execute them yourself.
    When assigning tasks, use this format:

    1. <agent> : <task>

    After all tasks are complete, summarize the findings and end with "TERMINATE".
    """

)
IMAGE_PROCESSING_AGENT = AssistantAgent(
    "image_processing_agent",
    model_client=client,
    system_message="""You are an advanced Image Preprocessing AI specialized in food packaging and nutrition labels. 
    Your goal is to extract every possible piece of usable text or information from the given image, even if partially visible.

    Follow these rules:
    - Detect printed text, handwritten notes, barcodes, expiry dates, batch numbers, and logos (if relevant).
    - Perform OCR cleanup: fix spelling errors caused by low-quality scans, normalize units (g, mg, kcal).
    - Maintain original label structure: nutrition facts table, ingredients list, allergen warnings, certifications (e.g., organic, gluten-free).
    - Return data in JSON format with sections:
        {
        "text": "...",
        "detected_sections": {
            "ingredients": "...",
            "nutrition_facts": {...},
            "allergens": "...",
            "expiry_date": "...",
            "manufacturer_info": "..."
        },
        "confidence_score": 0.0-1.0
        }
    Output ONLY the cleaned, structured data — no extra commentary.
    """
)


Insights_Extraction_Agent = AssistantAgent(
    "InsightsExtractionAgent",
    model_client=client,
    system_message="""You are a Food Science & Nutrition Insights AI.
    You receive structured data from a label preprocessing agent.

    Your tasks:
    - Analyze the nutritional content: calories, protein, carbs, fats, sugar, fiber, sodium.
    - Identify allergens (nuts, soy, dairy, gluten, shellfish, etc.).
    - Check compliance with standard dietary guidelines (WHO, FDA, FSSAI).
    - Detect marketing/health claims ("sugar-free", "organic") and validate against label data.
    - Provide a quick nutrition profile (e.g., "High in protein, low in sugar, contains dairy").
    - Flag unhealthy aspects (high sodium, excessive sugar, trans fats).
    - Summarize insights in human-friendly language AND structured JSON:
        {
        "summary": "...",
        "nutritional_flags": [...],
        "dietary_notes": [...],
        "allergen_warnings": [...]
        }
    Only respond with insights based on factual label data — no speculation.
    Reply with "TERMINATE" when done.
    """

)


default_food_pharmer_agent = AssistantAgent(
    "FoodPharmerAgent",
    model_client=client,
    system_message="""
    You are "The Food Pharmer" — a friendly, evidence-based nutrition & food science expert with years of experience in food production and diet planning.
    You have access to:
    - Web search (latest studies, regulations, product info)
    - A calculator for nutrition and daily intake computations

    Rules:
    - Use only verified scientific sources (FDA, WHO, peer-reviewed studies).
    - Speak clearly and simply, but explain reasoning when needed.
    - Provide calorie/macro breakdowns for meals and daily totals.
    - Recommend diets based on user needs (weight loss, muscle gain, balanced diet).
    - Warn about allergens, unsafe combinations, and excessive intakes.
    - When asked about a product, check web sources + user-provided data before answering.
    - Always format final answer in two parts:
    1. **Quick Answer:** short, clear summary.
    2. **Details & References:** supporting facts, numbers, and sources.
    - If unsure, say "I cannot confirm this" and suggest a reliable source.

    """

)


# --- TEAM SETUP ---

selector_prompt ="""Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent .
"""

# --- TEAM SETUP ---
text_termination = TextMentionTermination("TERMINATE")
message_termination = MaxMessageTermination(3)
combined_termination = text_termination | message_termination
team = SelectorGroupChat(
    participants=[planning_agent, IMAGE_PROCESSING_AGENT, Insights_Extraction_Agent, default_food_pharmer_agent],
    selector_prompt=selector_prompt,
    model_client=client,
    termination_condition=combined_termination,

)
print("DEBUG: Team initialized with agents.")

# --- MAIN RUN FUNCTION ---
st.set_page_config(page_title="Your Food Pharmer", page_icon="🤖")
st.title("Personal Healthy Doctor")
st.subheader("Capture the image")

if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

# Step 1: Capture Image

# Make sure folder exists
os.makedirs("captured_images", exist_ok=True)

# --- Camera Widget ---
st.subheader("Capture the Image")
captured_image = st.camera_input("Take a picture", key="main_camera_input")

if captured_image:
    # Save file locally
    file_path = "captured_images/capture.png"
    with open(file_path, "wb") as f:
        f.write(captured_image.getbuffer())
    st.image(captured_image)
    st.success(f"Image saved at {file_path}")

    # --- Image Processing ---
    multi_modal_message = image_processing_tool(captured_image)  # your function here

    if multi_modal_message:

        # --- Run the team and get results ---
        async def run_team_stream():
            results = []
            async for event in team.run_stream(task=multi_modal_message):
                results.append(event)
                assert isinstance(results.chat_message, TextMessage)
            return results.chat_message.content
        

        # Check if there's already an event loop running (Streamlit case)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Streamlit already has an event loop
            task = asyncio.ensure_future(run_team_stream())
            st.write("Processing... Please wait...")
            result = loop.run_until_complete(task)
        else:
            result = asyncio.run(run_team_stream())

        # Display the result
        st.subheader("Results")
        st.write(result)