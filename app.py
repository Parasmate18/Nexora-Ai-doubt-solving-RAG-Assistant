
import os
import streamlit as st
import streamlit.components.v1 as components

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from groq import Groq
from collections import Counter

load_dotenv()

st.set_page_config(
    page_title="NEXORA AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

COLLECTION_NAME = "knowledge_filter"

# -------------------- CONNECTIONS --------------------

@st.cache_resource
def load_clients():
    qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120
    )

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    return qdrant, embedding_model, groq_client


try:
    client, model, groq_client = load_clients()
    connected = client.collection_exists(COLLECTION_NAME)
except Exception as e:
    st.error(f"Connection error: {e}")
    st.stop()


# -------------------- LOAD FILTER OPTIONS --------------------

@st.cache_data(ttl=60)
def get_filter_options():
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=["category", "subcategory", "video_id", "duration"],
        with_vectors=False
    )

    categories = sorted({
        r.payload.get("category")
        for r in records
        if r.payload and r.payload.get("category")
    })

    subcategories = sorted({
        r.payload.get("subcategory")
        for r in records
        if r.payload and r.payload.get("subcategory")
    })

    lectures = len({
        r.payload.get("video_id")
        for r in records
        if r.payload and r.payload.get("video_id")
    })

    return categories, subcategories, lectures, len(records)


try:
    categories, subcategories, lecture_count, segment_count = get_filter_options()
except Exception as e:
    st.error(f"Could not load Qdrant data: {e}")
    st.stop()


# -------------------- SEARCH FUNCTION --------------------

def search_transcripts(query, category=None, subcategory=None, top_k=5):

    query_vector = model.encode(query).tolist()

    conditions = []

    if category and category != "All":
        conditions.append(
            FieldCondition(
                key="category",
                match=MatchValue(value=category)
            )
        )

    if subcategory and subcategory != "All":
        conditions.append(
            FieldCondition(
                key="subcategory",
                match=MatchValue(value=subcategory)
            )
        )

    search_filter = Filter(must=conditions) if conditions else None

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True
    ).points

    return results


# -------------------- CONTEXT --------------------

def prepare_context(results):

    context = []

    for result in results:
        p = result.payload

        start = int(p.get("start", 0))
        end = int(p.get("end", 0))

        start_time = f"{start // 60:02d}:{start % 60:02d}"
        end_time = f"{end // 60:02d}:{end % 60:02d}"

        context.append(f"""
Video: {p.get('title', 'Unknown')}
Category: {p.get('category', 'Unknown')}
Subcategory: {p.get('subcategory', 'Unknown')}
Timestamp: {start_time} - {end_time}
Video ID: {p.get('video_id', '')}
Content: {p.get('text', '')}
""")

    return "\n".join(context)


# -------------------- GROQ RESPONSE --------------------

def ask_nexora(question, context):

    prompt = f"""
You are NEXORA AI, a concise Web Development tutor.

Answer using ONLY the transcript context.

Rules:
1. Give a direct answer in 2-4 short sentences.
2. Explain only what the user asked.
3. Use simple Hinglish or English.
4. Avoid unnecessary introductions and long explanations.
5. Give the relevant video timestamp.
6. Never invent information.
7. If unavailable, say:
"I don't know based on the provided information."

Context:
{context}

Question:
{question}
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# -------------------- SESSION STATE --------------------

if "results" not in st.session_state:
    st.session_state.results = []

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "selected_video" not in st.session_state:
    st.session_state.selected_video = None

if "selected_time" not in st.session_state:
    st.session_state.selected_time = 0

if "question" not in st.session_state:
    st.session_state.question = ""


# -------------------- CUSTOM CSS --------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, #17191e 0%, transparent 40%),
        #090b0e;
    color: #f1f3f7;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

.nexora-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0 22px 0;
    border-bottom: 1px solid #292d35;
    margin-bottom: 35px;
}

.brand {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #f4f5f7;
}

.brand span {
    color: #ff682d;
}

.brand-icon {
    color: #ff682d;
    font-size: 30px;
    margin-right: 10px;
}

.nav-text {
    color: #aeb8c8;
    font-size: 14px;
}

.eyebrow {
    color: #ff682d;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-bottom: 14px;
}

.hero-title {
    font-size: clamp(32px, 4vw, 49px);
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -2px;
    margin-bottom: 14px;
}

.hero-title span {
    color: #ff682d;
}

.hero-description {
    font-size: 17px;
    line-height: 1.8;
    color: #aab8ca;
    max-width: 850px;
}

.stat-row {
    display: flex;
    gap: 28px;
    margin-top: 25px;
    margin-bottom: 30px;
    flex-wrap: wrap;
}

.stat {
    color: #8995a7;
    font-size: 14px;
}

.stat strong {
    color: #ffffff;
    font-size: 17px;
}

.search-box {
    background: #11151b;
    border: 1px solid #ff682d;
    border-radius: 16px;
    padding: 10px;
    box-shadow: 0 0 15px #ff682d25;
}

.suggestion {
    display: inline-block;
    border: 1px solid #303640;
    background: #151920;
    color: #b7c0ce;
    border-radius: 30px;
    padding: 10px 17px;
    margin: 5px 5px 5px 0;
    font-size: 13px;
}

.section-heading {
    color: #aeb9ca;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 1.1px;
    border-left: 4px solid #ff682d;
    padding-left: 12px;
    margin: 25px 0 15px 0;
}

.result-card {
    background: #151920;
    border: 1px solid #303640;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 13px;
}

.result-title {
    color: #f1f3f7;
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 8px;
}

.result-meta {
    color: #8995a7;
    font-size: 12px;
    margin-bottom: 12px;
}

.result-text {
    color: #c1c9d5;
    font-size: 14px;
    line-height: 1.7;
}

.answer-card {
    background: #141920;
    border: 1px solid #343b47;
    border-radius: 15px;
    padding: 22px;
    margin-top: 20px;
    line-height: 1.8;
}

.video-card {
    background: #000000;
    border: 1px solid #303640;
    border-radius: 17px;
    overflow: hidden;
}

.video-caption {
    background: #171b22;
    color: #aeb8c8;
    padding: 15px;
    font-size: 13px;
}

.stButton > button {
    background: #ff682d;
    color: white;
    border: none;
    border-radius: 11px;
    font-weight: 700;
    min-height: 43px;
}

.stButton > button:hover {
    background: #ff7c45;
    color: white;
    border: none;
    box-shadow: 0 0 15px #ff682d55;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #141920;
    border-color: #303640;
    border-radius: 10px;
}

div[data-testid="stTextInput"] input {
    color: white;
}

.stMarkdown, p, label {
    color: #c4ccd8;
}

hr {
    border-color: #292d35;
}

/* Clear text visibility */
.stApp, .stMarkdown, p, label {
    color: #F5F5F5 !important;
}

.result-card, .answer-card {
    background: #151920;
    color: #FFFFFF;
    border: 1px solid #343b47;
}

.result-title {
    color: #FFFFFF !important;
}

.result-text {
    color: #E5E7EB !important;
}

.result-meta {
    color: #B8C2D1 !important;
}

input {
    color: #FFFFFF !important;
    background: #151920 !important;
}

</style>
""", unsafe_allow_html=True)


# -------------------- HEADER --------------------

st.markdown("""
<div class="nexora-header">
    <div class="brand">
        <span class="brand-icon">▶</span>NEXORA
    </div>
    <div class="nav-text">
        ⚡ AI KNOWLEDGE ASSISTANT
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------- HERO --------------------

st.markdown("""
<div class="eyebrow">● YOUR OWN DOUBT SOLVING AI ASSISTANT</div>

<div class="hero-title">
Find the exact second I explained <span>anything.</span>
</div>

<div class="hero-description">
Ask any Web Development doubt in Hinglish or English.
Get an answer built only from what was actually said in the lectures —
and links that jump straight to that moment.
</div>
""", unsafe_allow_html=True)

count = client.count(
    collection_name=COLLECTION_NAME,
    exact=True
).count

st.markdown(f"""
<div class="stat-row">
    <div class="stat"><strong>{lecture_count}</strong> lectures</div>
    <div class="stat"><strong>{segment_count:,}</strong> indexed moments</div>
    <div class="stat"><strong>{count:,}</strong> searchable points</div>
</div>
""", unsafe_allow_html=True)

st.divider()


# -------------------- MAIN LAYOUT --------------------

left, right = st.columns([1.08, 0.95], gap="large")


with left:

    st.markdown('<div class="section-heading">ASK NEXORA AI</div>',
                unsafe_allow_html=True)

    with st.form("search_form"):

        question = st.text_input(
            "Ask your question",
            placeholder="e.g. CSS flexbox kaise work karta hai?",
            label_visibility="collapsed"
        )

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            selected_category = st.selectbox(
                "Category",
                ["All"] + categories
            )

        filtered_subcategories = [
            s for s in subcategories
        ]

        with filter_col2:
            selected_subcategory = st.selectbox(
                "Subcategory",
                ["All"] + filtered_subcategories
            )

        top_k = st.slider(
            "Number of relevant moments",
            min_value=3,
            max_value=10,
            value=5
        )

        submitted = st.form_submit_button(
            "⚡ Ask NEXORA",
            use_container_width=True
        )

    suggestions = [
        "What is the difference between inline and external CSS?",
        "HTML images kaise add karte hai?",
        "CSS flexbox kaise work karta hai?",
        "What is CSS specificity?",
        "Explain HTML tables",
        "What are CSS sizing units?"
    ]

    st.markdown('<div class="section-heading">TRY ASKING</div>',
                unsafe_allow_html=True)

    for i in range(0, len(suggestions), 2):
        col1, col2 = st.columns(2)

        with col1:
            if st.button(suggestions[i], key=f"suggestion_{i}",
                         use_container_width=True):
                st.session_state.question = suggestions[i]
                st.session_state.run_suggestion = True
                st.rerun()

        if i + 1 < len(suggestions):
            with col2:
                if st.button(suggestions[i + 1], key=f"suggestion_{i+1}",
                             use_container_width=True):
                    st.session_state.question = suggestions[i + 1]
                    st.session_state.run_suggestion = True
                    st.rerun()

    if submitted and question.strip():
        st.session_state.question = question
        st.session_state.run_suggestion = False
        st.session_state.selected_video = None

        with st.spinner("NEXORA is searching your lectures..."):
            try:
                results = search_transcripts(
                    question,
                    category=selected_category,
                    subcategory=selected_subcategory,
                    top_k=top_k
                )

                st.session_state.results = results

                if results:
                    context = prepare_context(results)
                    st.session_state.answer = ask_nexora(
                        question,
                        context
                    )
                else:
                    st.session_state.answer = (
                        "I couldn't find relevant information "
                        "in the selected lectures."
                    )

            except Exception as e:
                st.error(f"Search error: {e}")

    if st.session_state.get("run_suggestion", False):
        st.session_state.run_suggestion = False

        with st.spinner("NEXORA is searching your lectures..."):
            try:
                results = search_transcripts(
                    st.session_state.question,
                    top_k=5
                )

                st.session_state.results = results

                if results:
                    context = prepare_context(results)
                    st.session_state.answer = ask_nexora(
                        st.session_state.question,
                        context
                    )
                else:
                    st.session_state.answer = (
                        "No relevant lecture moments found."
                    )

            except Exception as e:
                st.error(f"Search error: {e}")

    ## -------------------- ANSWER + RESULTS LAYOUT --------------------

if st.session_state.answer or st.session_state.results:

    answer_col, results_col = st.columns([1, 1], gap="large")

    with answer_col:

        st.markdown(
            '<div class="section-heading">NEXORA ANSWER</div>',
            unsafe_allow_html=True
        )

        if st.session_state.answer:
            st.markdown(
                f'<div class="answer-card">{st.session_state.answer}</div>',
                unsafe_allow_html=True
            )

    with results_col:

        st.markdown(
            '<div class="section-heading">WHERE TO WATCH</div>',
            unsafe_allow_html=True
        )

        if st.session_state.results:

            for index, result in enumerate(st.session_state.results):

                p = result.payload
                start = int(p.get("start", 0))
                start_time = f"{start // 60:02d}:{start % 60:02d}"
                video_id = p.get("video_id")

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">
                        {p.get('title', 'Unknown')}
                    </div>
                    <div class="result-meta">
                        {p.get('category', 'Unknown')} ·
                        {p.get('subcategory', 'Unknown')}
                    </div>
                    <div class="result-text">
                        {p.get('text', '')[:180]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(
                    f"▶ [{start_time} — Watch on YouTube]"
                    f"(https://www.youtube.com/watch?v={video_id}&t={start}s)"
                )

                if st.button("Play here", key=f"play_{index}"):
                    st.session_state.selected_video = video_id
                    st.session_state.selected_time = start
                    st.rerun()
# -------------------- VIDEO PLAYER --------------------

with right:

    st.markdown(
        '<div class="section-heading">LECTURE PLAYER</div>',
        unsafe_allow_html=True
    )

    if st.session_state.selected_video:

        video_id = st.session_state.selected_video
        start_time = st.session_state.selected_time

        embed_url = (
            f"https://www.youtube.com/embed/{video_id}"
            f"?start={start_time}&autoplay=1"
        )

        components.iframe(
            embed_url,
            height=390,
            scrolling=False
        )

        st.markdown(
            f'<div class="video-caption">● Playing lecture at {start_time // 60:02d}:{start_time % 60:02d}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"[Open video on YouTube](https://www.youtube.com/watch?v={video_id}&t={start_time}s)"
        )

    else:

        st.markdown("""
        <div class="video-card">
            <div style="
                height:390px;
                display:flex;
                flex-direction:column;
                justify-content:center;
                align-items:center;
                text-align:center;
                color:#7d8797;
                padding:25px;
            ">
                <div style="font-size:55px;color:#343a44;">▶</div>
                <br>
                <div style="font-size:16px;">
                    Click any timestamp and the lecture starts playing here,
                    <br>at that exact second.
                </div>
            </div>
            <div class="video-caption">
                ● No lecture loaded
            </div>
        </div>
        """, unsafe_allow_html=True)


# -------------------- FOOTER --------------------

st.markdown("<br><hr>", unsafe_allow_html=True)

st.markdown("""
<div style="
    text-align:center;
    color:#6f7a8b;
    font-size:12px;
    padding:15px;
">
    NEXORA AI · Your Knowledge. Your Intelligence.
    <br>
    Built with Python · Qdrant · Sentence Transformers · Groq · Streamlit
</div>
""", unsafe_allow_html=True)


