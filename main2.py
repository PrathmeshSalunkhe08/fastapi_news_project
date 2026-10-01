import os
import json
import re
import html
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from groq import Groq
import uvicorn

# -----------------------------------------------------------------------------
# STEP 1: Environment Variables setup (.env file read karne ke liye)
# -----------------------------------------------------------------------------
load_dotenv()


# -----------------------------------------------------------------------------
# STEP 2: Data Model (Pydantic Schema - Har article ka structure decide karta hai)
# -----------------------------------------------------------------------------
class LLMArticleInsight(BaseModel):
    id: int = 1
    title: str
    summary: str
    topics: List[str]
    urgency_score: int = Field(default=7, ge=1, le=10)
    category: str = "General"
    source: str = "Live News"
    link: str = "#"
    takeaways: List[str] = []
    impact_analysis: str = ""


# -----------------------------------------------------------------------------
# STEP 3: Google News RSS Scraper (Live News fetch karne ke liye)
# -----------------------------------------------------------------------------
def clean_html(text: str) -> str:
    """HTML tags aur special characters ko clean karta hai"""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


async def fetch_news_from_google(query: str, limit: int = 5) -> List[dict]:
    """Google News RSS Feed se live articles scrape karta hai"""
    encoded_query = urllib.parse.quote(query.strip())
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=6.0, follow_redirects=True)
            response.raise_for_status()

        # RSS XML Parse karna
        root = ET.fromstring(response.text)
        items = root.findall(".//item")

        articles = []
        for index, item in enumerate(items[:limit], start=1):
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")
            source_el = item.find("source")

            raw_title = clean_html(title_el.text if title_el is not None else "")
            raw_link = link_el.text if link_el is not None and link_el.text else "#"
            raw_source = clean_html(source_el.text if source_el is not None else "Live News")
            raw_desc = clean_html(desc_el.text if desc_el is not None else "")

            # Title aur Publisher name split karna
            parts = raw_title.rsplit(" - ", 1)
            title = parts[0].strip()
            source = parts[1].strip() if len(parts) > 1 else raw_source

            desc = re.sub(r'\s*-\s*[A-Za-z0-9\.\s]+$', '', raw_desc).strip()
            if not desc or len(desc) < 20:
                desc = title

            articles.append({
                "id": index,
                "title": title,
                "raw_desc": desc,
                "source": source,
                "link": raw_link
            })

        return articles

    except Exception as err:
        print(f"Scraping Error: {err}")
        return []


# -----------------------------------------------------------------------------
# STEP 4: Groq AI Summarization (AI Se News Analyze Karne Ke Liye)
# -----------------------------------------------------------------------------
def summarize_with_groq_ai(query: str, raw_articles: List[dict]) -> Optional[List[LLMArticleInsight]]:
    """Groq LLM Model se JSON array me AI Insights generate karta hai"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        client = Groq(api_key=api_key)
        model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        count = len(raw_articles)

        # Articles Context Build Karna
        context = "\n\n".join([
            f"Article {a['id']}:\nTitle: {a['title']}\nSource: {a['source']}\nDetails: {a['raw_desc']}"
            for a in raw_articles
        ])

        system_prompt = (
            f"Analyze the live news and return a JSON array of {count} objects.\n"
            "Each object must have these keys:\n"
            "- id (int), title (str), summary (2-3 sentences str), topics (array of 3-4 str),\n"
            "- urgency_score (int 1-10), category (str), source (str),\n"
            "- takeaways (array of 3 str), impact_analysis (str).\n"
            "Return ONLY raw JSON with no markdown formatting."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Topic: '{query}'\n\nArticles:\n{context}"}
            ],
            temperature=0.2,
            max_tokens=2500
        )

        content = response.choices[0].message.content.strip()
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

        parsed = json.loads(content)
        if isinstance(parsed, list) and len(parsed) > 0:
            results = []
            for i, p in enumerate(parsed):
                link = raw_articles[i]["link"] if i < len(raw_articles) else "#"
                p["link"] = link
                if "id" not in p:
                    p["id"] = i + 1
                results.append(LLMArticleInsight(**p))
            return results

    except Exception as err:
        print(f"Groq AI Error (Fallback will be used): {err}")

    return None


def fallback_summarize(query: str, raw_articles: List[dict]) -> List[LLMArticleInsight]:
    """Agar Groq API Key Na Ho Ya Error Aaye Toh Fallback Summary Banata Hai"""
    results = []
    stopwords = {"this", "that", "with", "from", "after", "will", "have", "more", "over", "into", "their", "under", "news", "live", "about", "what", "here", "says"}

    for art in raw_articles:
        words = [w.strip('.,:;()[]"?!').capitalize() for w in (query + " " + art["title"]).split() if len(w) > 3]
        topics = list(dict.fromkeys([w for w in words if w.lower() not in stopwords]))[:4]
        if not topics:
            topics = [query.title(), "Current Affairs", "Breaking News"]

        results.append(LLMArticleInsight(
            id=art["id"],
            title=art["title"],
            summary=f"{art['title']}. Continuous live coverage reported by {art['source']} regarding {query}.",
            topics=topics,
            urgency_score=7,
            category=query.title()[:20],
            source=art["source"],
            link=art["link"],
            takeaways=[
                f"Headline: {art['title']}.",
                f"Source: Verified by {art['source']}.",
                f"Key Topics: {', '.join(topics[:3])}."
            ],
            impact_analysis=f"Ongoing public developments regarding {query} on {art['source']}."
        ))

    return results


# -----------------------------------------------------------------------------
# STEP 5: FastAPI Application & Endpoints
# -----------------------------------------------------------------------------
app = FastAPI(title="NewsIntel AI", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/news")
async def get_news(
    query: Optional[str] = Query("Technology"),
    category: Optional[str] = Query(None),
    limit: Optional[int] = Query(5, ge=1, le=15)
):
    """Main API Endpoint: Live News Scrape Karke AI Summaries Return Karta Hai"""
    search_query = query.strip() if query and query.strip() else (category.strip() if category else "Technology")
    article_limit = limit if limit and 1 <= limit <= 15 else 5

    # 1. Live Google News Scrape Karna
    raw_articles = await fetch_news_from_google(search_query, limit=article_limit)
    if not raw_articles:
        raw_articles = await fetch_news_from_google(f"{search_query} news", limit=article_limit)

    # 2. AI Summaries Generate Karna (Groq AI ya Fallback)
    articles = summarize_with_groq_ai(search_query, raw_articles)
    if not articles:
        articles = fallback_summarize(search_query, raw_articles)

    articles_list = [a.model_dump() for a in articles]

    # Cache output file for backup
    try:
        with open("final_insights.json", "w", encoding="utf-8") as f:
            json.dump(articles_list, f, indent=2)
    except Exception:
        pass

    return {
        "status": "success",
        "query": search_query,
        "total_briefs": len(articles_list),
        "articles": articles_list
    }


# Static Frontend Serving
@app.get("/")
@app.get("/index.html")
async def serve_index():
    return FileResponse("index.html")

@app.get("/style.css")
async def serve_css():
    return FileResponse("style.css")

@app.get("/app.js")
async def serve_js():
    return FileResponse("app.js")


# -----------------------------------------------------------------------------
# STEP 6: Server Launcher
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(" ⚡ NewsIntel AI Backend Started: http://127.0.0.1:8000")
    print("=" * 50 + "\n")
    uvicorn.run("main2:app", host="127.0.0.1", port=8000, reload=True)
