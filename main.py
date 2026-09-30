import os
import json
import logging
import asyncio
from typing import List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load API keys from .env
load_dotenv()

# Setup simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("NewsSummarizer")


# =============================================================================
# 1. DATA MODELS (Pydantic Validation)
# =============================================================================
class ArticleInput(BaseModel):
    """Stores the raw content fetched from a URL."""
    url: str
    content: str
    fetch_status: str  # "success" or "failed"
    error_message: Optional[str] = None


class LLMArticleInsight(BaseModel):
    """Enforces the structure of the AI summary."""
    title: str = Field(description="Headline or title of the article")
    summary: str = Field(description="2-3 sentence summary of the article")
    topics: List[str] = Field(description="2 to 5 important topics or tags")
    urgency_score: int = Field(..., ge=1, le=10, description="Urgency score from 1 (low) to 10 (critical)")
    category: Literal["Tech", "Business", "Health", "Other"] = Field(description="Primary category")


# =============================================================================
# 2. ASYNC ARTICLE FETCHER (httpx + asyncio)
# =============================================================================
async def fetch_article(client: httpx.AsyncClient, url: str) -> ArticleInput:
    """Fetches a single web page asynchronously with basic error handling."""
    logger.info(f"Request started: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = await client.get(url, timeout=8.0, headers=headers, follow_redirects=True)
        response.raise_for_status()
        logger.info(f"Request completed: {url} ({response.status_code})")
        return ArticleInput(url=url, content=response.text, fetch_status="success")
    except httpx.TimeoutException:
        logger.error(f"Request timed out: {url}")
        return ArticleInput(url=url, content="", fetch_status="failed", error_message="Timeout")
    except Exception as e:
        logger.error(f"Request failed for {url}: {e}")
        return ArticleInput(url=url, content="", fetch_status="failed", error_message=str(e))


async def fetch_all(urls: List[str]) -> List[ArticleInput]:
    """Fetches all URLs concurrently using asyncio.gather."""
    async with httpx.AsyncClient() as client:
        tasks = [fetch_article(client, url) for url in urls]
        return await asyncio.gather(*tasks)


# =============================================================================
# 3. LLM SUMMARIZATION (LangChain Structured Output)
# =============================================================================
def get_structured_llm():
    """Initializes the Groq LLM with structured output."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY missing in .env. Running in simulation mode.")
        return None
    model_name = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    llm = ChatGroq(model=model_name, api_key=api_key, temperature=0.1)
    return llm.with_structured_output(LLMArticleInsight)


async def summarize_article(llm, article: ArticleInput) -> Optional[LLMArticleInsight]:
    """Sends clean text to the LLM to get a structured summary."""
    content_text = article.content[:4000].strip()
    if not content_text:
        return None

    if llm is None:
        # Fallback if no API key is set
        return LLMArticleInsight(
            title="Analysis of Online News Article",
            summary="A comprehensive overview generated and validated with Pydantic.",
            topics=["Technology", "Automation", "News"],
            urgency_score=7,
            category="Tech"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert news analyst. Extract key information from the article into the requested structured format."),
        ("human", "Article Source URL: {url}\n\nArticle Content:\n{content}")
    ])
    chain = prompt | llm

    try:
        insight = await chain.ainvoke({"url": article.url, "content": content_text})
        logger.info(f"LLM processing completed for: {article.url}")
        return insight
    except Exception as e:
        logger.error(f"LLM processing error for {article.url}: {e}")
        return None


# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================
# Curated live news feeds by domain
DOMAIN_OPTIONS = {
    "1": ("Technology & AI", [
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "https://dev.to/api/articles/latest?per_page=1",
        "https://arxiv.org/abs/2609.36941"
    ]),
    "2": ("Business & Finance", [
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://www.thehindu.com/business/feeder/default.rss",
        "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml"
    ]),
    "3": ("Health & Science", [
        "https://feeds.bbci.co.uk/news/health/rss.xml",
        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "https://www.thehindu.com/sci-tech/health/feeder/default.rss"
    ]),
    "4": ("Politics & National", [
        "https://www.thehindu.com/news/national/feeder/default.rss",
        "https://www.thehindu.com/news/national/other-states/feeder/default.rss",
        "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml"
    ])
}


async def main():
    print("\n" + "=" * 65)
    print(" [*] CONCURRENT NEWS SCRAPER & LLM SUMMARIZER")
    print("=" * 65 + "\n")

    # Step 1: Let the user choose a domain
    print("Select a domain:")
    for key, (name, _) in DOMAIN_OPTIONS.items():
        print(f"  [{key}] {name}{' (Default)' if key == '1' else ''}")

    choice = input("\nEnter choice (1-4) or press Enter for default: ").strip()
    domain_name, live_urls = DOMAIN_OPTIONS.get(choice, DOMAIN_OPTIONS["1"])
    print(f"\n>> Selected: '{domain_name}'")

    # Step 2: Combine live URLs with error test URLs (404 and timeout)
    urls_to_scrape = live_urls + [
        "https://httpbin.org/status/404",  # Error test: 404
        "https://httpbin.org/delay/15"     # Error test: Timeout
    ]

    # Step 3: Fetch all URLs concurrently
    print(f"\n[1/3] Fetching {len(urls_to_scrape)} URLs in parallel...")
    fetched_articles = await fetch_all(urls_to_scrape)

    # Step 4: Process successful articles with LLM
    print("\n[2/3] Generating structured AI insights...")
    llm = get_structured_llm()
    insights = []
    failed = []

    for art in fetched_articles:
        if art.fetch_status == "success" and art.content:
            insight = await summarize_article(llm, art)
            if insight:
                insights.append(insight.model_dump())
        else:
            failed.append({
                "url": art.url,
                "fetch_status": "failed",
                "error": art.error_message or "Request failed"
            })

    # Step 5: Save results to JSON files
    print("\n[3/3] Saving outputs to JSON files...")
    with open("final_insights.json", "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)

    if failed:
        with open("failed_articles.json", "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)

    # Step 6: Print clean final summary dashboard
    print("\n" + "=" * 70)
    print(" [REPORT] EXECUTION SUMMARY DASHBOARD")
    print("=" * 70)
    print(f" • Domain Selected       : {domain_name}")
    print(f" • Total URLs Ingested   : {len(urls_to_scrape)}")
    print(f" • Successfully Analyzed : {len(insights)}")
    print(f" • Handled Errors/Tests  : {len(failed)}")
    print(f" • Output File Saved To  : {os.path.abspath('final_insights.json')}")
    print("=" * 70)

    if insights:
        print("\n" + "-" * 70)
        print(" [INSIGHTS] ARTICLE SUMMARIES & RATINGS")
        print("-" * 70)
        for i, item in enumerate(insights, 1):
            print(f"\n[{i}] {item.get('title')}")
            print(f"    Category : {item.get('category')}  |  Urgency Score : {item.get('urgency_score')}/10")
            print(f"    Topics   : {', '.join(item.get('topics', []))}")
            print(f"    Summary  : {item.get('summary')}")
            print("-" * 70)

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
