from io import BytesIO

import requests
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image as AGImage
from PIL import Image
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.ui import Console
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tavily import TavilyClient
tavily_client = TavilyClient(api_key="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")
def web_search_tool(query:str) -> str:
    """Perform a web search using Tavily API."""
    response = tavily_client.search(query=query, num_results=2)
    if response and response.results:
        return "\n".join([f"{i+1}. {result.title} - {result.url}" for i, result in enumerate(response.results)])        
    return "No results found."





model_client = OpenAIChatCompletionClient(model="gemini-1.5-pro", api_key="AIzaSyBNA4vbBXBa2z2ADqwv_bnwi1RNgB0v6Hw")
agent = AssistantAgent(
    name="ImageDescriptionAgent",
    model_client=model_client,
    tools=[web_search_tool],
    system_message="Use the web search tool to find information about the image if needed. Describe the content of the image in detail.",
)
pil_image = Image.open(BytesIO(requests.get("https://encrypted-tbn1.gstatic.com/shopping?q=tbn:ANd9GcSb-6RCM9tmBjNFitR2lHCV7ZouVdfJrL5aJaV5geNUSn4sCL5cWGkGwMmexHEMf1NcpIZURH2fd2YErJxzz5xCOOxrZqAhqpXTVISLRcPzdAow7FetTxQINoQ").content))
img = AGImage(pil_image)
multi_modal_message = MultiModalMessage(content=["Can you describe the content of this image?", img], source="User")

async def assistant_run_stream() -> None:
    await Console(
        agent.run_stream(task=multi_modal_message),
        output_stats=True,  # Enable stats printing.
    )
if __name__ == "__main__":
    import asyncio
    asyncio.run(assistant_run_stream())