import os
import re
from typing import Dict, Any, Type, List
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr
from pydantic.config import ConfigDict
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# ── Setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CrewAI + YouTube Blog (Groq)", page_icon="🎬", layout="wide")
st.title("🎬 CrewAI + YouTube Blog Generator")
load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192")
DEFAULT_CHANNEL_VIDEOS_URL = os.getenv("YOUTUBE_CHANNEL_VIDEOS_URL", "https://www.youtube.com/@codebasics/videos")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY is not set. Add it to a .env file or your environment before running.")

# ── Token presets ───────────────────────────────────────────────────────
TOKEN_PRESETS: Dict[str, Dict[str, float]] = {
    "Ultra-Saver": {"max_tokens": 512,  "temperature": 0.2},
    "Balanced":    {"max_tokens": 1024, "temperature": 0.3},
    "Detailed":    {"max_tokens": 2048, "temperature": 0.35},
}

def make_llm(model_name: str, preset_key: str) -> LLM:
    p = TOKEN_PRESETS[preset_key]
    return LLM(
        model=f"groq/{model_name}",
        api_key=GROQ_API_KEY,
        temperature=p["temperature"],
        max_tokens=int(p["max_tokens"]),
        
    )

# ── Regex helpers ───────────────────────────────────────────────────────
VIDEOS_URL_RE = re.compile(r"^https?://(www\.)?youtube\.com/[^?#]+/videos/?$", re.I)

def require_videos_url(s: str) -> str:
    s = (s or "").strip()
    if not VIDEOS_URL_RE.match(s):
        raise ValueError("Please paste the FULL channel videos URL, e.g. https://www.youtube.com/@codebasics/videos")
    return s

def strip_videos_tab(videos_url: str) -> str:
    return re.sub(r"/videos/?$", "", videos_url, flags=re.I)

def _mmss(seconds: float) -> str:
    s = int(seconds)
    m = s // 60
    s = s % 60
    return f"{m:02d}:{s:02d}"

# ── Transcript snippets ─────────────────────────────────────────────────
def fetch_transcript_snippets(video_id: str, terms: List[str], max_hits: int = 3) -> List[str]:
    hits: List[str] = []
    try:
        tr = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    except (TranscriptsDisabled, NoTranscriptFound, Exception):
        return hits
    for seg in tr:
        txt = seg.get("text", "")
        if not txt:
            continue
        low = txt.lower()
        if any(t in low for t in terms):
            ts = _mmss(seg.get("start", 0.0))
            hits.append(f"[{ts}] {txt}")
            if len(hits) >= max_hits:
                break
    return hits

# ── Fallback search with yt_dlp ─────────────────────────────────────────
def yt_dlp_channel_search(channel_base_url: str, query: str, limit: int = 10, want_snippets: bool = True) -> str:
    url = f"{channel_base_url}/videos"
    ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": True, "playlistend": 100}
    results_md: List[str] = []
    ql = query.lower()

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        entries = info.get("entries", []) if isinstance(info, dict) else []
        for e in entries:
            title = (e.get("title") or "").strip()
            vid_id = e.get("id")
            if not vid_id or not title:
                continue
            if ql in title.lower():
                video_url = f"https://www.youtube.com/watch?v={vid_id}"
                md = f"- **{title}**  \n  {video_url}"
                if want_snippets:
                    snips = fetch_transcript_snippets(vid_id, [query], max_hits=3)
                    if snips:
                        md += "\n  " + "\n  ".join(f"• {s}" for s in snips)
                results_md.append(md)
                if len(results_md) >= limit:
                    break

    if not results_md:
        return "No live fallback results."
    return "FALLBACK_RESULTS (yt_dlp live scan)\n\n" + "\n".join(results_md)

# ── FixedChannelYouTubeSearchTool (yt-dlp only) ─────────────────────────
class SearchArgs(BaseModel):
    search_query: str = Field(..., description="Text to search in this channel")
    expand: bool = Field(default=True)
    max_variants: int = Field(default=6)
    fallback_limit: int = Field(default=6)
    fallback_snippets: bool = Field(default=True)

class FixedChannelYouTubeSearchTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "Search a YouTube Channel"
    description: str = "Search within ONE fixed YouTube channel using yt_dlp + transcripts."
    args_schema: Type[BaseModel] = SearchArgs

    _base_channel_url: str = PrivateAttr()

    def __init__(self, channel_videos_url: str):
        super().__init__()
        channel_videos_url = require_videos_url(channel_videos_url)
        self._base_channel_url = strip_videos_tab(channel_videos_url)

    def _run(self, search_query: str, expand: bool = True,
             max_variants: int = 6, fallback_limit: int = 6,
             fallback_snippets: bool = True, **kwargs: Any) -> Any:
        return yt_dlp_channel_search(
            self._base_channel_url,
            search_query,
            limit=fallback_limit,
            want_snippets=fallback_snippets
        )

# ── Agent & Task Builders ───────────────────────────────────────────────
def build_agents(search_tool: FixedChannelYouTubeSearchTool, llm_obj: LLM, memory: bool, topic: str):
    researcher = Agent(
        role="Blog Researcher",
        goal=f"Find the best-matching video and collect notes for {topic}.",
        backstory="Expert in analyzing transcripts and metadata.",
        tools=[search_tool],
        verbose=True,
        memory=memory,
        allow_delegation=True,
        llm=llm_obj,
    )
    writer = Agent(
        role="Blog Writer",
        goal=f"Write an engaging blog post about {topic} using the research brief.",
        backstory="Simplifies complex topics into clear posts.",
        tools=[search_tool],
        verbose=True,
        memory=memory,
        allow_delegation=False,
        llm=llm_obj,
    )
    return researcher, writer

def build_tasks(researcher: Agent, writer: Agent, search_tool: FixedChannelYouTubeSearchTool,
                topic: str, top_n: int, expand_queries: bool, max_variants: int,
                fallback_limit: int, fallback_snippets: bool):
    research_task = Task(
        description=(
            f"Search the channel for videos related to '{topic}'. "
            f"Limit consideration to the top {top_n} most relevant results. "
            "Extract key points and transcript excerpts with timestamps if available."
        ),
        expected_output="A 3-paragraph research brief with reasoning, key points, and source URLs.",
        tools=[search_tool],
        agent=researcher,
        output_file="research-brief.md",
    )

    write_task = Task(
        description=(
            f"Using the research brief, write a concise blog post on '{topic}'. "
            "Include an intro, 3–5 subheadings, and a short conclusion with a clear takeaway. "
            "Cite the video (URL + title) at the end."
        ),
        expected_output="A polished Markdown blog post ready to publish.",
        tools=[search_tool],
        agent=writer,
        async_execution=False,
        output_file="new-blog-post.md",
    )

    return research_task, write_task

# ── Crew Runner ─────────────────────────────────────────────────────────
def run_crew(channel_videos_url_in: str, topic: str,
             expand_queries: bool, max_variants: int,
             fallback_limit: int, fallback_snippets: bool,
             search_depth: int, model_name: str, preset_key: str,
             enable_memory: bool):
    fixed_tool = FixedChannelYouTubeSearchTool(channel_videos_url_in)
    llm_obj = make_llm(model_name, preset_key)

    researcher, writer = build_agents(fixed_tool, llm_obj, enable_memory, topic)
    research_task, write_task = build_tasks(researcher, writer, fixed_tool, topic,
                                            search_depth, expand_queries, max_variants,
                                            fallback_limit, fallback_snippets)

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        memory=enable_memory,
        cache=True,
        max_rpm=100,
        share_crew=True,
    )

    with st.spinner("Running agents…"):
        return crew.kickoff(inputs={
            "topic": topic,
            "expand": expand_queries,
            "max_variants": max_variants,
            "fallback_limit": fallback_limit,
            "fallback_snippets": fallback_snippets
        })


# ---------------------
# Streamlit Run & Display
# ---------------------
with st.sidebar:
    st.header("Controls")
    channel_videos_url = st.text_input(
        "Paste the channel's **/videos** URL",
        value=DEFAULT_CHANNEL_VIDEOS_URL,
        help="Must end with /videos, e.g. https://www.youtube.com/@codebasics/videos"
    )
    model_name = st.selectbox("Groq model",
    options=[
    
        "qwen/qwen3-32b"
    ],
    index=0
)




    preset_key = st.radio("Token budget preset", list(TOKEN_PRESETS.keys()), index=1)
    search_depth = st.slider("Search depth", 1, 10, 3)
    enable_memory = st.toggle("Enable agent memory", value=False)
    expand_queries = st.toggle("Expand acronyms/keywords if no results", value=True)
    max_variants = st.slider("Max expansion variants", 1, 10, 6)
    fallback_limit = st.slider("Live fallback: max results", 1, 12, 6)
    fallback_snippets = st.toggle("Include transcript snippets (fallback)", value=True)

topic = st.text_input("Enter a topic for the blog", value="What is AI?")
run_clicked = st.button("▶️ Run Agents", type="primary")

if run_clicked:
    try:
        videos_url = require_videos_url(channel_videos_url)
        final_output = run_crew(videos_url, topic,
                                expand_queries, max_variants,
                                fallback_limit, fallback_snippets,
                                search_depth, model_name, preset_key,
                                enable_memory)

        st.success("✅ Agents finished! Scroll for outputs.")

        brief_text = None
        blog_text = None
        if os.path.exists("research-brief.md"):
            with open("research-brief.md", "r", encoding="utf-8") as f:
                brief_text = f.read()
        if os.path.exists("new-blog-post.md"):
            with open("new-blog-post.md", "r", encoding="utf-8") as f:
                blog_text = f.read()

        tab1, tab2, tab3 = st.tabs(["Research Brief", "Blog (Markdown)", "Raw Result"])

        with tab1:
            st.markdown(brief_text or "No research brief file found.")
            if brief_text:
                st.download_button("Download research-brief.md", brief_text,
                                   "research-brief.md", "text/markdown")

        with tab2:
            st.markdown(blog_text or "No blog markdown file found.")
            if blog_text:
                st.download_button("Download new-blog-post.md", blog_text,
                                   "new-blog-post.md", "text/markdown")

        with tab3:
            st.code(str(final_output))

        st.divider()
        st.subheader("Token Budget Preset Recap")
        p = TOKEN_PRESETS[preset_key]
        st.write(f"**Preset:** {preset_key} — max_tokens={p['max_tokens']} • temperature={p['temperature']}")

    except Exception as e:
        st.error("Channel/LLM issue. Paste a FULL channel videos URL like https://www.youtube.com/@codebasics/videos")
        st.exception(e)


# ---------------------
# Footer / Help
# ---------------------
with st.expander("Setup & Notes"):
    st.markdown(
        """
        **Quickstart**
        1) Create/activate Python 3.11 virtualenv  
        2) `pip install -U crewai crewai-tools litellm streamlit python-dotenv yt_dlp youtube-transcript-api`  
        3) Create `.env` with `GROQ_API_KEY`, optional `GROQ_MODEL_NAME`, `YOUTUBE_CHANNEL_VIDEOS_URL`  
        4) `streamlit run app.py`

        **How it avoids misses**
        - Tool runs first (semantic search).  
        - If empty → **live fallback** scans the channel’s /videos with `yt_dlp`, and optionally
          fetches a few transcript lines with timestamps. This catches **brand-new uploads**.
        """
    )
