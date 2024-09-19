from langchain.agents import tool
from langchain_community.utilities import SerpAPIWrapper
from private_key import SERPAPI_API_KEY


@tool
def get_google_search_information(query: str) -> str:
    """Searches for information on the internet using Google and returns the search results."""
    serpapi = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
    res = serpapi.run(query)
    return res
