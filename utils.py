import json
import time
from typing import List
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from trafilatura import fetch_url, extract

def get_server_time():
    utc_time = datetime.now(timezone.utc)
    return utc_time.strftime("%Y-%m-%d %H:%M:%S")

def get_website_content_from_url(url: str) -> str:
    """
    Get website content from a URL using Selenium and BeautifulSoup for improved content extraction and filtering.
    Args:
        url (str): URL to get website content from.
    Returns:
        str: Extracted content including title, main text, and tables.
    """

    try:
        downloaded = fetch_url(url)

        result = extract(downloaded, include_formatting=True, include_links=True, output_format='json', url=url)

        if result:
            result = json.loads(result)
            return f'=========== Website Title: {result["title"]} ===========\n\n=========== Website URL: {url} ===========\n\n=========== Website Content ===========\n\n{result["raw_text"]}\n\n=========== Website Content End ===========\n\n'
        else:
            return ""
    except Exception as e:
        return f"An error occurred: {str(e)}"



class CitingSources(BaseModel):
    """
    This represents the citing of the sources you used to answer the user query.
    """
    sources: List[str] = Field(
        ...,
        description="List of sources to cite. Should be an URL of the source. E.g. GitHub URL, Blogpost URL or Newsletter URL."
    )