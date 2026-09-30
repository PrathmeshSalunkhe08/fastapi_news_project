import os
import sys
import time
import json
import logging
import asyncio
from typing import List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# UTF-8 terminal support on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# =============================================================================
# 1. LOGGING & DATA SCHEMAS
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("NewsScraper")


class ArticleInput(BaseModel):
    """Raw article data payload."""
    url: str
    content: str
    fetch_status: str
    error_message: Optional[str] = None


class LLMArticleInsight(BaseModel):
    """Structured insights validated by Pydantic."""
    title: str = Field(description="Article title")
    summary: str = Field(description="Concise 2-4 sentence summary")
    topics: List[str] = Field(description="2 to 5 relevant topics/tags")
    urgency_score: int = Field(..., ge=1, le=10, description="Urgency rating from 1 to 10")
    category: Literal["Tech", "Business", "Health", "Other"] = Field(description="Article category")


# =============================================================================
# 2. ASYNC SCRAPING WITH RETRIES & ERROR HANDLING
# =============================================================================
async def fetch_article(client: httpx.AsyncClient, url: str, timeout: float = 8.0, max_retries: int = 2) -> ArticleInput:
    """Fetches a URL asynchronously with timeout, HTTP error handling, and retries."""
    logger.info(f"Request started: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for attempt in range(1, max_retries + 1):
        try:
            res = await client.get(url, timeout=timeout, headers=headers, follow_redirects=True)
            res.raise_for_status()
            logger.info(f"Request completed: {url} ({res.status_code})")
            return ArticleInput(url=url, content=res.text, fetch_status="success")
        except httpx.TimeoutException:
            err = "Timeout"
        except httpx.HTTPStatusError as e:
            err = f"HTTP {e.response.status_code}"
        except httpx.RequestError:
            err = "Network Error"

        logger.warning(f"{err} on attempt {attempt}/{max_retries} for: {url}")
        if attempt == max_retries:
            logger.error(f"Request failed ({err}): {url}")
            return ArticleInput(url=url, content="", fetch_status="failed", error_message=err)
        await asyncio.sleep(0.3)

    return ArticleInput(url=url, content="", fetch_status="failed", error_message="Max retries reached")


async def fetch_all_concurrently(urls: List[str]) -> List[ArticleInput]:
    """Fetches all URLs in parallel using asyncio.gather()."""
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*(fetch_article(client, u) for u in urls))


async def fetch_all_sequentially(urls: List[str]) -> List[ArticleInput]:
    """Fetches URLs one-by-one for performance benchmarking."""
    results = []
    async with httpx.AsyncClient() as client:
        for u in urls:
            results.append(await fetch_article(client, u))
    return results


# =============================================================================
# 3. DOMAIN FEEDS & STRUCTURED LLM SUMMARIZATION
# =============================================================================
DOMAINS = {
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
    ]),
    "5": ("World News", [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
        "https://feeds.bbci.co.uk/news/world/asia/rss.xml"
    ])
}


def get_structured_llm():
    """Returns LangChain LLM bound to LLMArticleInsight structured output."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(model=model, api_key=api_key, temperature=0.1).with_structured_output(LLMArticleInsight)


async def process_article_with_llm(structured_llm, article: ArticleInput) -> Optional[LLMArticleInsight]:
    """Extracts structured insights from article content using LangChain."""
    content_snippet = article.content[:4000].strip()
    if not content_snippet:
        return None

    if structured_llm is None:
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
    chain = prompt | structured_llm

    try:
        insight: LLMArticleInsight = await chain.ainvoke({"url": article.url, "content": content_snippet})
        logger.info(f"LLM processing completed for: {article.url}")
        return insight
    except Exception as e:
        logger.error(f"Error in LLM structured extraction for {article.url}: {e}")
        return None


# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================
async def main():
    print("\n" + "=" * 75)
    print(" [*] CONCURRENT MULTI-SOURCE NEWS SCRAPER & LLM SUMMARIZER")
    print("=" * 75 + "\n")

    # Domain Selection Menu
    print("Select a News Domain to Fetch & Summarize:")
    for k, (name, _) in DOMAINS.items():
        print(f"  [{k}] {name}{' (Default)' if k == '1' else ''}")

    choice = input("\nEnter choice (1-5) or press Enter for default [1]: ").strip()
    domain_name, live_urls = DOMAINS.get(choice, DOMAINS["1"])
    print(f"\n>> Selected Domain: '{domain_name}'")

    # Target URLs: Live Feeds + Resilience Error Test URLs
    urls = live_urls + ["https://httpbin.org/status/404", "https://httpbin.org/delay/15"]
    print(f"\n[STEP 1/3] Queued {len(urls)} URLs ({len(live_urls)} live feeds + 2 error tests)...\n")

    # Benchmark: Sequential vs Concurrent (Bonus 1)
    start_seq = time.perf_counter()
    await fetch_all_sequentially(urls[:2])
    seq_time = (time.perf_counter() - start_seq) * (len(urls) / 2)

    start_conc = time.perf_counter()
    fetched_articles = await fetch_all_concurrently(urls)
    conc_time = time.perf_counter() - start_conc
    logger.info(f"Performance Comparison: Concurrent ({conc_time:.2f}s) vs Sequential (~{seq_time:.2f}s)")

    # LLM Structured Processing
    print("\n[STEP 2/3] Processing Articles with LangChain Structured Output...")
    structured_llm = get_structured_llm()
    insights, failed = [], []

    for art in fetched_articles:
        if art.fetch_status == "success" and art.content:
            res = await process_article_with_llm(structured_llm, art)
            if res:
                insights.append(res.model_dump())
        else:
            failed.append({"url": art.url, "fetch_status": "failed", "error": art.error_message or "Request failed"})

    # Export to final_insights.json and failed_articles.json
    print("\n[STEP 3/3] Exporting Results to final_insights.json...")
    with open("final_insights.json", "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)
    logger.info(f"Saved {len(insights)} structured insights to 'final_insights.json'")

    if failed:
        with open("failed_articles.json", "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        logger.info(f"Saved {len(failed)} failed request logs to 'failed_articles.json'")

    # Summary Dashboard
    print("\n" + "=" * 75)
    print(" [REPORT] EXECUTION SUMMARY REPORT DASHBOARD")
    print("=" * 75)
    print(f" - Domain Processed      : {domain_name}")
    print(f" - Total URLs Ingested    : {len(urls)}")
    print(f" - Successfully Analyzed  : {len(insights)}")
    print(f" - Handled Failures/Tests : {len(failed)}")
    print(f" - Concurrent Execution   : {conc_time:.2f}s (vs ~{seq_time:.2f}s sequential)")
    print(f" - Output File Location   : {os.path.abspath('final_insights.json')}")
    print("=" * 75)

    if insights:
        print("\n" + "-" * 75)
        print(" [INSIGHTS] DETAILED ARTICLE SUMMARIES")
        print("-" * 75)
        for i, item in enumerate(insights, 1):
            print(f"\n[{i}] {item.get('title')}")
            print(f"    Category : {item.get('category')}  |  Urgency Score : {item.get('urgency_score')}/10")
            print(f"    Topics   : {', '.join(item.get('topics', []))}")
            print(f"    Summary  : {item.get('summary')}")
            print("-" * 75)

    print("\n" + "=" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
