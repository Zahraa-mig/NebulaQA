import streamlit as st
import random
import requests

st.set_page_config(page_title="Ask the Cosmos", page_icon="🪐", layout="centered")


API_URL = "http://localhost:8000/ask"


def find_answer(question: str) -> str:
    q = question.strip()
    if not q:
        return "Type a question above and I'll do my best to answer it! 🚀"

    try:
        response = requests.post(
            API_URL,
            json={"question": q, "max_new_tokens": 150},
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "").strip()
        return answer if answer else "⚠️ الموديل رجع إجابة فاضية، جربي تصيغي السؤال بشكل مختلف."
    except requests.exceptions.ConnectionError:
        return "⚠️ مش قادر أوصل للـ API — تأكدي إنه `api.py` شغال على المنفذ 8000 (شغلي `python api.py` أو `uvicorn api:app --port 8000`)."
    except requests.exceptions.Timeout:
        return "⚠️ الموديل ماخد وقت أطول من اللازم بالرد، جربي سؤال أقصر أو زيدي الـ timeout."
    except Exception as e:
        return f"⚠️ صار خطأ غير متوقع: {e}"


def _generate_stars(n: int, seed: int) -> str:
    rnd = random.Random(seed)
    return ", ".join(f"{rnd.randint(0, 2000)}px {rnd.randint(0, 2000)}px #FFF" for _ in range(n))


SMALL_STARS = _generate_stars(260, seed=42)
BIG_STARS = _generate_stars(90, seed=7)

BACKGROUND_TEMPLATE = """
<style>
/* ---------- page base ---------- */
.stApp {
    background: radial-gradient(ellipse at bottom, #0b0e2a 0%, #000010 100%) !important;
    overflow-x: hidden;
}

/* keep app content readable over the animated background */
div[data-testid="stMainBlockContainer"] {
    background: rgba(6, 8, 28, 0.65) !important;
    border-radius: 18px;
    padding: 2rem 2.2rem !important;
    backdrop-filter: blur(5px);
    box-shadow: 0 0 40px rgba(120, 80, 255, 0.15);
    margin-top: 2rem;
}

h1, h2, h3, p, label, .stMarkdown, .stTextInput label {
    color: #eae6ff !important;
}

/* ---------- starfield ---------- */
#space-bg {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0;
    overflow: hidden;
    pointer-events: none;
}

.stars-small, .stars-big {
    position: absolute;
    width: 1px; height: 1px;
    background: transparent;
    animation: twinkle 4s infinite ease-in-out alternate;
}
.stars-small { box-shadow: SMALL_STARS; opacity: 0.8; }
.stars-big   { box-shadow: BIG_STARS; width: 2px; height: 2px; animation-duration: 6s; }

@keyframes twinkle {
    from { opacity: 0.4; }
    to   { opacity: 1; }
}

/* ---------- planets: each spins around its own axis ---------- */
.planet {
    position: absolute;
    border-radius: 50%;
    animation-name: spin;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

.planet-mercury {
    top: 6%; left: 4%;
    width: 60px; height: 60px;
    background: radial-gradient(circle at 35% 35%, #cfc7bd, #8c8378 70%, #55504a);
    animation-duration: 9s;
    box-shadow: 0 0 25px rgba(200,190,180,0.35);
}
.planet-venus {
    top: 68%; left: 2%;
    width: 90px; height: 90px;
    background: radial-gradient(circle at 35% 35%, #f5d68b, #d99a4e 70%, #a5652a);
    animation-duration: 14s;
    box-shadow: 0 0 30px rgba(245,214,139,0.35);
}
.planet-earth {
    top: 12%; right: 6%;
    width: 100px; height: 100px;
    background: radial-gradient(circle at 35% 35%, #5ec8ff, #2a6fd6 55%, #123a7a);
    animation-duration: 11s;
    box-shadow: 0 0 35px rgba(90,170,255,0.4);
}
.planet-mars {
    top: 42%; left: 78%;
    width: 55px; height: 55px;
    background: radial-gradient(circle at 35% 35%, #ff9d6c, #c1440e 70%, #7a2704);
    animation-duration: 8s;
    box-shadow: 0 0 22px rgba(255,120,60,0.35);
}
.planet-jupiter {
    top: 78%; left: 70%;
    width: 140px; height: 140px;
    background: repeating-linear-gradient(0deg, #d9b38c, #b98657 8%, #e8cba7 16%, #a9744b 24%);
    animation-duration: 22s;
    box-shadow: 0 0 45px rgba(220,180,130,0.35);
}
.planet-saturn {
    top: 4%; left: 42%;
    width: 80px; height: 80px;
    background: radial-gradient(circle at 35% 35%, #f0dcae, #c9a86a 70%, #8f7440);
    animation-duration: 17s;
    box-shadow: 0 0 25px rgba(240,220,175,0.35);
}
.planet-saturn::after {
    content: "";
    position: absolute;
    top: 50%; left: 50%;
    width: 150px; height: 150px;
    border: 6px solid rgba(230, 210, 170, 0.55);
    border-radius:
    50%;
    transform: translate(-50%, -50%) rotateX(75deg);
}
.planet-neptune {
    top: 55%; right: 3%;
    width: 70px; height: 70px;
    background: radial-gradient(circle at 35% 35%, #7ea8ff, #2f4fb0 70%, #182a70);
    animation-duration: 13s;
    box-shadow: 0 0 28px rgba(100,140,255,0.4);
}
</style>

<div id="space-bg">
    <div class="stars-small"></div>
    <div class="stars-big"></div>
    <div class="planet planet-mercury"></div>
    <div class="planet planet-venus"></div>
    <div class="planet planet-earth"></div>
    <div class="planet planet-mars"></div>
    <div class="planet planet-jupiter"></div>
    <div class="planet planet-saturn"></div>
    <div class="planet planet-neptune"></div>
</div>
"""

BACKGROUND_HTML = BACKGROUND_TEMPLATE.replace("SMALL_STARS", SMALL_STARS).replace("BIG_STARS", BIG_STARS)

st.markdown(BACKGROUND_HTML, unsafe_allow_html=True)



st.title("🪐 Ask the Nebula")
st.write("Ask me anything about planets, stars, galaxies, black holes, and the rest of the universe.")

if "question" not in st.session_state:
    st.session_state.question = ""

# quick example buttons
example_questions = [
    "What is a black hole?",
    "How many planets are there?",
    "What is the Milky Way?",
    "Why is Mars red?",
]
cols = st.columns(len(example_questions))
for col, ex_q in zip(cols, example_questions):
    if col.button(ex_q, use_container_width=True):
        st.session_state.question = ex_q

question = st.text_input(
    "Your question:",
    value=st.session_state.question,
    placeholder="e.g. What is a light-year?",
    key="question_input",
)

if st.button("🚀 Get answer", use_container_width=True) or question:
    if question:
        answer = find_answer(question)
        st.markdown(
            f"""
            <div style="
                margin-top: 1rem;
                padding: 1rem 1.2rem;
                border-radius: 12px;
                background: rgba(120, 100, 255, 0.15);
                border: 1px solid rgba(160, 140, 255, 0.35);
                color: #f2efff;
                font-size: 1.05rem;
                line-height: 1.5;">
                {answer}
            </div>
            """,
            unsafe_allow_html=True,
        )
