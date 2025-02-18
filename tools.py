from crewai_tools import PDFSearchTool
from crewai.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from config import GROQ_API_KEY

def create_pdf_tool(pdf_path):
    """
    A tool to create a PDF search utility.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        PDFSearchTool: A configured PDF search tool instance.
    """
    return PDFSearchTool(
        pdf=pdf_path,
        config=dict(
            llm=dict(
                provider="groq",  
                config=dict(
                    model="deepseek-r1-distill-qwen-32b",
                    api_key=GROQ_API_KEY,
                    # temperature=0.3,
                    # max_tokens=2048,
                ),
            ),
            embedder=dict(
                provider="huggingface",
                config=dict(
                    model="BAAI/bge-small-en-v1.5",
                ),
            ),
        ),
    )


@tool
def web_search_tool(query):
    """
    Web Search Tool.

    Args:
        query (str): The search query.

    Returns:
        str: The search results as text.
    """
    web_search_tool = TavilySearchResults(k=3)
    return web_search_tool.run(query)
