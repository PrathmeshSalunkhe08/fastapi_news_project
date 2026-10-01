// =============================================================================
// NewsIntel — Universal Real-Time News Client
// Theme System: White, Red, Black, Grey & Blue
// =============================================================================

let articlesData = [];
let isFetching = false;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  setupSearchInput();
});

// Setup Search Input (Enter key and Clear button)
function setupSearchInput() {
  const input = document.getElementById("searchInput");
  const clearBtn = document.getElementById("clearBtn");

  if (input) {
    input.addEventListener("input", () => {
      clearBtn.style.display = input.value ? "flex" : "none";
    });

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        executeSearch();
      }
    });
  }
}

// Quick Pill Tag Click Helper
function setQueryAndSearch(topicText) {
  const input = document.getElementById("searchInput");
  const clearBtn = document.getElementById("clearBtn");
  if (input) {
    input.value = topicText;
    if (clearBtn) clearBtn.style.display = "flex";
    executeSearch();
  }
}

// Select Article Count via Segmented Control
function selectCount(val) {
  const hiddenInput = document.getElementById("newsLimitSelect");
  const btnText = document.getElementById("searchBtnText");

  if (hiddenInput) hiddenInput.value = val;
  if (btnText) btnText.textContent = `⚡ Search ${val} Latest News`;

  const btns = document.querySelectorAll(".segment-btn");
  btns.forEach(btn => {
    const isThis = btn.getAttribute("data-val") === String(val);
    btn.classList.toggle("active", isThis);
  });
}

// Clear Search Input
function clearSearch() {
  const input = document.getElementById("searchInput");
  const clearBtn = document.getElementById("clearBtn");
  if (input) {
    input.value = "";
    if (clearBtn) clearBtn.style.display = "none";
    input.focus();
  }
}

// Main Action: Fetch Live News for Whatever is in the Search Bar
async function executeSearch() {
  if (isFetching) return;

  const input = document.getElementById("searchInput");
  const limitSelect = document.getElementById("newsLimitSelect");
  const query = input?.value.trim();
  const limit = limitSelect ? limitSelect.value : "5";

  if (!query) {
    showToast("⚠️ Please type any topic to search (e.g. 'Politics of Maharashtra')");
    input?.focus();
    return;
  }

  isFetching = true;
  const btn = document.getElementById("fetchNewsBtn");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span>⏳ Searching "${escapeHtml(query)}" (${limit} articles)...</span>`;
  }

  renderNewsLoading(query);

  try {
    const url = `/api/news?query=${encodeURIComponent(query)}&limit=${encodeURIComponent(limit)}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`FastAPI returned HTTP ${response.status}`);
    }

    const payload = await response.json();
    
    if (payload.status === "success" && Array.isArray(payload.articles)) {
      articlesData = payload.articles;

      // Update Div 2 (Latest News & Inline Deep Dive)
      updateDiv2Header(payload.query || query, articlesData.length);
      renderDiv2News();

      showToast(`⚡ Ingested ${articlesData.length} live real-time news for "${query}"`);

      // Smooth scroll to Div 2
      document.getElementById("div-news")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      throw new Error("Invalid payload format received from backend");
    }

  } catch (err) {
    console.error("Backend error:", err);
    renderNewsError(err.message);
  } finally {
    isFetching = false;
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<span id="searchBtnText">⚡ Search ${limit} Latest News</span>`;
    }
  }
}

// Update Div 2 Header
function updateDiv2Header(targetName, count) {
  const title = document.getElementById("newsSectionTitle");
  const sub = document.getElementById("newsSectionSub");
  const meta = document.getElementById("metaIndicators");

  const articleCount = count || articlesData.length || 5;
  if (title) title.textContent = `${articleCount} Live Articles: ${targetName}`;
  if (sub) sub.textContent = `Live real-time feed fetched & validated via FastAPI backend`;
  if (meta) meta.style.display = "flex";
}

// Render 3 News Cards in Div 2
function renderDiv2News() {
  const grid = document.getElementById("newsGrid");
  if (!grid) return;
  grid.innerHTML = "";

  if (articlesData.length === 0) {
    grid.innerHTML = `<div class="empty-state">No live articles found for this search. Try another topic!</div>`;
    return;
  }

  // Update Stats
  const countEl = document.getElementById("statArticleCount");
  const avgEl = document.getElementById("statAvgUrgency");
  if (countEl) countEl.textContent = articlesData.length;
  if (avgEl) {
    const avg = (articlesData.reduce((acc, a) => acc + (a.urgency_score || 0), 0) / articlesData.length).toFixed(1);
    avgEl.textContent = avg;
  }

  articlesData.forEach((art, index) => {
    const card = document.createElement("article");
    card.className = "news-card";

    const score = art.urgency_score || 7;
    const urgencyClass = score >= 8 ? "urgency-high" : (score >= 6 ? "urgency-mid" : "urgency-low");
    const sourceLabel = art.source || "Live Publisher";

    const topicsHtml = (art.topics || []).map(t => 
      `<span class="topic-chip">#${escapeHtml(t)}</span>`
    ).join("");

    const takeawaysHtml = (art.takeaways || [art.summary]).map(t => 
      `<li>${escapeHtml(t)}</li>`
    ).join("");

    card.innerHTML = `
      <div>
        <div class="card-top">
          <span class="category-tag" title="${escapeHtml(sourceLabel)}">${escapeHtml(sourceLabel)}</span>
          <span class="urgency-badge ${urgencyClass}">⚡ Urgency: ${score}/10</span>
        </div>
        <h3 class="card-title">${escapeHtml(art.title)}</h3>
        <p class="card-summary">${escapeHtml(art.summary)}</p>
      </div>

      <div>
        <div class="card-topics">
          ${topicsHtml}
        </div>
        
        <!-- Toggle Inline Deep Dive Button -->
        <button class="btn-toggle-dive" id="diveBtn-${index}" onclick="toggleCardDeepDive(${index})">
          <span>🔍 Deep Dive Analysis</span>
          <span id="diveArrow-${index}">↓</span>
        </button>

        <!-- Inline Deep Dive Details Panel -->
        <div class="card-deep-dive-panel" id="divePanel-${index}">
          <div class="dive-section-label">💡 Executive Key Takeaways</div>
          <ul class="takeaway-list">
            ${takeawaysHtml}
          </ul>

          <div class="dive-section-label">🎯 Strategic Impact</div>
          <div class="dive-impact-text">${escapeHtml(art.impact_analysis || "Real-time coverage and developments.")}</div>

          <button class="btn-copy-mini" onclick="copyCardBriefing(${index})">
            📋 Copy Briefing
          </button>
        </div>
      </div>
    `;

    grid.appendChild(card);
  });
}

// Toggle Inline Deep Dive
function toggleCardDeepDive(index) {
  const panel = document.getElementById(`divePanel-${index}`);
  const arrow = document.getElementById(`diveArrow-${index}`);
  if (!panel) return;

  const isOpen = panel.classList.contains("open");
  panel.classList.toggle("open", !isOpen);
  if (arrow) arrow.textContent = isOpen ? "↓" : "↑";
}

// Copy Card Briefing
function copyCardBriefing(index) {
  const art = articlesData[index];
  if (!art) return;
  const text = `📰 [${art.source}] ${art.title}\n\nSummary:\n${art.summary}\n\nTakeaways:\n${(art.takeaways || []).map(t => `• ${t}`).join("\n")}\n\nUrgency: ${art.urgency_score}/10 | Source: ${art.source}`;
  
  navigator.clipboard.writeText(text)
    .then(() => showToast("📋 Executive briefing copied!"))
    .catch(() => showToast("❌ Failed to copy."));
}

// Loading & Error States
function renderNewsLoading(target) {
  const grid = document.getElementById("newsGrid");
  if (grid) {
    grid.innerHTML = `
      <div class="empty-state">
        <p style="color: var(--brand-black); font-weight: 700;">⚡ Fetching 100% live real-time news for "<span style="color: var(--brand-red);">${escapeHtml(target)}</span>"...</p>
        <small style="color: var(--text-tertiary); margin-top: 6px; display: block;">Querying live news wire & running Pydantic validation</small>
      </div>
    `;
  }
}

function renderNewsError(msg) {
  const grid = document.getElementById("newsGrid");
  if (grid) {
    grid.innerHTML = `
      <div class="empty-state" style="border-color: var(--brand-red);">
        <p style="color: var(--brand-red); font-weight: 700;">⚠️ Could not connect to FastAPI backend</p>
        <p style="color: var(--text-secondary); font-size: 12px; margin-top: 4px;">Make sure <code>python main.py</code> is running.</p>
        <button class="btn btn-outline" style="margin-top: 12px;" onclick="executeSearch()">🔄 Retry</button>
      </div>
    `;
  }
}

// Toast Notification
function showToast(msg) {
  const toast = document.getElementById("toast");
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => {
    toast.classList.remove("show");
  }, 2500);
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/[&<>'"]/g, t => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  }[t] || t));
}
