import os
import json
import time
import asyncio
from typing import List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import httpx
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AI News Intelligence & Summarizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .news-card { background: #ffffff; padding: 22px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.04); border-left: 5px solid #2563eb; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 15px; font-size: 12px; font-weight: 600; margin-right: 6px; }
    .badge-tech { background-color: #dbeafe; color: #1e40af; }
    .badge-business { background-color: #fef3c7; color: #92400e; }
    .badge-health { background-color: #dcfce7; color: #166534; }
    .badge-other { background-color: #f3e8ff; color: #6b21a8; }
    .badge-urgency-high { background-color: #fee2e2; color: #991b1b; }
    .badge-urgency-mid { background-color: #fef9c3; color: #854d0e; }
    .badge-urgency-low { background-color: #e0f2fe; color: #075985; }
    .topic-tag { display: inline-block; background-color: #f1f5f9; color: #475569; padding: 3px 8px; border-radius: 6px; font-size: 11px; margin: 2px 4px 2px 0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 1. PYDANTIC DATA MODELS
# =============================================================================
class ArticleInput(BaseModel):
    url: str
    content: str
    fetch_status: str
    error_message: Optional[str] = None


class LLMArticleInsight(BaseModel):
    title: str = Field(description="Headline or title of the article")
    summary: str = Field(description="Concise 2-4 sentence summary of core content")
    topics: List[str] = Field(description="2 to 5 important topics or tags")
    urgency_score: int = Field(..., ge=1, le=10, description="Urgency score from 1 to 10")
    category: Literal["Tech", "Business", "Health", "Other"] = Field(description="Primary category")


# =============================================================================
# 2. ASYNC SCRAPING ENGINE
# =============================================================================
async def fetch_article(client: httpx.AsyncClient, url: str) -> ArticleInput:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        res = await client.get(url, timeout=8.0, headers=headers, follow_redirects=True)
        res.raise_for_status()
        return ArticleInput(url=url, content=res.text, fetch_status="success")
    except httpx.TimeoutException:
        return ArticleInput(url=url, content="", fetch_status="failed", error_message="Connection Timed Out")
    except Exception as e:
        return ArticleInput(url=url, content="", fetch_status="failed", error_message=str(e))


async def fetch_all(urls: List[str]) -> List[ArticleInput]:
    async with httpx.AsyncClient() as client:
        tasks = [fetch_article(client, u) for u in urls]
        return await asyncio.gather(*tasks)


# =============================================================================
# 3. LLM SUMMARIZATION ENGINE
# =============================================================================
def get_structured_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    model_name = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    llm = ChatGroq(model=model_name, api_key=api_key, temperature=0.1)
    return llm.with_structured_output(LLMArticleInsight)


async def summarize_article(llm, article: ArticleInput) -> Optional[LLMArticleInsight]:
    content_snippet = article.content[:4000].strip()
    if not content_snippet:
        return None

    if llm is None:
        return LLMArticleInsight(
            title="Analysis of Online News Article",
            summary="A comprehensive overview generated and validated with Pydantic.",
            topics=["Technology", "Digital Transformation", "Automation"],
            urgency_score=7,
            category="Tech"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert news analyst. Extract key information from the article into the requested structured format."),
        ("human", "Article Source URL: {url}\n\nArticle Content:\n{content}")
    ])
    chain = prompt | llm

    try:
        return await chain.ainvoke({"url": article.url, "content": content_snippet})
    except Exception:
        return None


# =============================================================================
# 4. STREAMLIT INTERFACE
# =============================================================================
DOMAINS = {
    "Technology & AI": [
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "https://dev.to/api/articles/latest?per_page=1",
        "https://arxiv.org/abs/2609.36941"
    ],
    "Business & Finance": [
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://www.thehindu.com/business/feeder/default.rss",
        "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml"
    ],
    "Health & Science": [
        "https://feeds.bbci.co.uk/news/health/rss.xml",
        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "https://www.thehindu.com/sci-tech/health/feeder/default.rss"
    ],
    "Politics & National": [
        "https://www.thehindu.com/news/national/feeder/default.rss",
        "https://www.thehindu.com/news/national/other-states/feeder/default.rss",
        "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml"
    ],
    "World News": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
        "https://feeds.bbci.co.uk/news/world/asia/rss.xml"
    ]
}

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("Select your news source domain and configure extraction parameters.")

    selected_domain = st.selectbox("🌐 Choose News Domain", list(DOMAINS.keys()), index=0)
    include_error_tests = st.checkbox("🧪 Include Resilience Tests (404 & Timeout)", value=True)

    st.markdown("---")
    st.markdown("### 🤖 Model Configuration")
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        st.success("API Key Active (Groq)")
    else:
        st.warning("Running in Fallback Simulation Mode")

    st.info(f"**Model:** `{os.getenv('LLM_MODEL', 'openai/gpt-oss-120b')}`")
    st.markdown("---")
    scrape_button = st.button("🚀 Run Live Scraping & AI Analysis", type="primary", use_container_width=True)


# Main Content Area
st.title("📰 Concurrent News Scraper & AI Summarizer")
st.caption("Asynchronous multi-source web scraping with `httpx` + `asyncio`, validated by `Pydantic` and summarized via `LangChain`.")

if scrape_button:
    target_urls = DOMAINS[selected_domain].copy()
    if include_error_tests:
        target_urls.extend(["https://httpbin.org/status/404", "https://httpbin.org/delay/15"])

    with st.spinner("⏳ Fetching articles concurrently and extracting structured AI insights..."):
        start_time = time.perf_counter()
        fetched_data = asyncio.run(fetch_all(target_urls))
        elapsed_fetch = time.perf_counter() - start_time

        llm = get_structured_llm()
        insights = []
        failed = []

        for item in fetched_data:
            if item.fetch_status == "success" and item.content:
                res = asyncio.run(summarize_article(llm, item))
                if res:
                    insights.append(res.model_dump())
            else:
                failed.append({"url": item.url, "error": item.error_message or "Request failed"})

        # Save outputs locally
        with open("final_insights.json", "w", encoding="utf-8") as f:
            json.dump(insights, f, indent=2)
        if failed:
            with open("failed_articles.json", "w", encoding="utf-8") as f:
                json.dump(failed, f, indent=2)

    # Metrics Dashboard
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Ingested", len(target_urls))
    m2.metric("Successfully Analyzed", len(insights))
    m3.metric("Handled Errors", len(failed))
    m4.metric("Scrape Duration", f"{elapsed_fetch:.2f}s")

    st.markdown("---")

    # Tabbed Display
    tab_cards, tab_json, tab_errors = st.tabs(["📌 Article Summaries", "💾 Raw JSON Output", "🛡️ Resilience Log"])

    with tab_cards:
        if insights:
            for idx, article in enumerate(insights, 1):
                category = article.get("category", "Other")
                cat_class = f"badge-{category.lower()}" if category.lower() in ["tech", "business", "health"] else "badge-other"
                
                score = article.get("urgency_score", 5)
                score_class = "badge-urgency-high" if score >= 8 else ("badge-urgency-mid" if score >= 5 else "badge-urgency-low")

                st.markdown(f"""
                <div class="news-card">
                    <span class="badge {cat_class}">{category}</span>
                    <span class="badge {score_class}">Urgency: {score}/10</span>
                    <h3 style="margin-top: 10px; margin-bottom: 8px;">{article.get('title')}</h3>
                    <p style="color: #334155; line-height: 1.6;">{article.get('summary')}</p>
                    <div style="margin-top: 12px;">
                        {''.join([f'<span class="topic-tag">#{t}</span>' for t in article.get('topics', [])])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No articles extracted in this run.")

    with tab_json:
        st.json(insights)
        st.download_button(
            label="📥 Download final_insights.json",
            data=json.dumps(insights, indent=2),
            file_name="final_insights.json",
            mime="application/json"
        )

    with tab_errors:
        if failed:
            st.write("The following endpoints were intentionally tested and handled gracefully without interrupting execution:")
            for err in failed:
                st.error(f"**URL**: `{err.get('url')}` — **Status**: `{err.get('error')}`")
            st.download_button(
                label="📥 Download failed_articles.json",
                data=json.dumps(failed, indent=2),
                file_name="failed_articles.json",
                mime="application/json"
            )
        else:
            st.success("All URLs fetched with 100% success.")

else:
    # Default State: Show previous results if available
    st.info("👈 Select your news domain in the sidebar and click **'Run Live Scraping & AI Analysis'** to start!")
    
    if os.path.exists("final_insights.json"):
        with open("final_insights.json", "r", encoding="utf-8") as f:
            try:
                saved_insights = json.load(f)
                if saved_insights:
                    st.markdown("### 📊 Latest Saved Insights (`final_insights.json`)")
                    for art in saved_insights:
                        st.markdown(f"""
                        <div class="news-card">
                            <span class="badge badge-tech">{art.get('category')}</span>
                            <span class="badge badge-urgency-mid">Urgency: {art.get('urgency_score')}/10</span>
                            <h4 style="margin-top: 8px;">{art.get('title')}</h4>
                            <p style="color: #334155;">{art.get('summary')}</p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception:
                pass
