
# app.py
# ================================
# IMPORTS
# ================================
import os
import json
import faiss
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# ================================
# CONFIGURE LLM (GROQ API)
# ================================
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def call_llm(prompt):
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You generate structured JSON personality questions only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return completion.choices[0].message.content.strip()

# ================================
# GENERATE TEST QUESTIONS (JSON)
# ================================
def generate_personality_questions():
    prompt = """
    Acting as an expert psychologist. You are creating a personality assessment based on the Big Five traits:

- Openness
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism

REQUIREMENTS:
- Generate EXACTLY 15 questions personality assessment based on the TIPI Big Five model.
- Each question must be a natural full sentence.
- Questions may implicitly measure one or multiple traits, but DO NOT mention the traits in the output.
- Questions should feel conversational, real, and not scientific.
- Output JSON ONLY as:
JSON FORMAT:
{
  "questions": [
     "Question 1",
     ...
     "Question 15"
  ],
  "trait_mapping": [
     {"index": 0, "traits": ["Extraversion"]},
     ...
  ]
}

Allowed trait names:
Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness

No extra text, no comments, no formatting outside JSON.
    No explanations. No markdown.
    """

    output = call_llm(prompt)
    output = output.replace("```json","").replace("```","").strip()

    try:
        data = json.loads(output)

        final = []
        for i, qtext in enumerate(data["questions"]):
            # default trait extraction from trait_mapping
            trait = "Openness"
            for tm in data.get("trait_mapping", []):
                if tm.get("index") == i and tm.get("traits"):
                    trait = tm["traits"][0]
                    break

            final.append({
                "question": qtext,
                "options": [
                    "Strongly Disagree",
                    "Disagree",
                    "Neutral",
                    "Agree",
                    "Strongly Agree"
                ],
                "trait": trait
            })

        return final

    except Exception as e:
        print("LLM OUTPUT:", output)
        print("JSON parse error:", e)
        return [{"error": "INVALID JSON RETURNED"}]

# ================================
# SCORING THE TEST
# ================================
def score_answers(answers):
    trait_scores = {}
    trait_counts = {}

    for ans in answers:
        trait = ans["trait"]
        value = ans["value"]
        if trait not in trait_scores:
            trait_scores[trait] = 0
            trait_counts[trait] = 0
        trait_scores[trait] += value
        trait_counts[trait] += 1

    # normalize 0-100
    for t in trait_scores:
        trait_scores[t] = round((trait_scores[t] / (trait_counts[t] * 5)) * 100)

    return trait_scores

# ================================
# GENERATE LLM PERSONALITY REPORT
# ================================
def generate_report(trait_scores):
    prompt = f"""
You are a personality psychology expert.

Based on these Big Five scores:
{trait_scores}

Write a **personalized report in second person**.
1. A 3-4 sentence personality summary.
2. 4 strengths (bullet list)
3. 4 growth areas (bullet list)
4. 4 actionable recommendations

Use friendly, conversational language.
No headings, just formatted text.
"""
    return call_llm(prompt)

#==========================
# Save FAISS ACROSS SESSION
#==========================
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(384)
    st.session_state.documents = []

# ================================
# SETUP FAISS VECTOR STORE
# ================================
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dimension = 384

def store_in_faiss(text):
    vec = embed_model.encode([text])
    st.session_state.faiss_index.add(np.array(vec).astype('float32'))
    st.session_state.documents.append(text)

# ================================
# RAG CHATBOT
# ================================
def rag_chat(user_query):
    """Answer user queries based on stored personality report using LLM."""
    if len(st.session_state.documents) == 0:
        return "No personality profile found yet. Take the test first."

    # Encode user query
    query_vec = embed_model.encode([user_query])
    distances, neighbors = st.session_state.faiss_index.search(np.array(query_vec).astype('float32'), 3)

    # Retrieve top 3 relevant documents (guard against -1)
    neighbor_idxs = [int(i) for i in neighbors[0] if i != -1]
    retrieved = "\n\n".join(st.session_state.documents[i] for i in neighbor_idxs)

    prompt = f"""
Based only on this personality profile:

{retrieved}

Answer the user's question in plain text, not JSON.
Answer in second person always try to address the user with you, your etc.
Use bullet points if multiple items, and make it friendly and readable.

QUESTION: {user_query}
"""
    answer = call_llm(prompt)
    return answer

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="AI Personality Assessment & Insight System", page_icon="🧠", layout="wide")

# Gradient header using HTML
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #ff7eb9, #ff758c); padding: 20px; border-radius: 10px">
        <h1 style="color: white; text-align: center;">🧠 AI Personality Assessment & Coach</h1>
    </div>
    """, unsafe_allow_html=True
)

# Initialize session state
if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = []
if "scores" not in st.session_state:
    st.session_state.scores = {}
if "report" not in st.session_state:
    st.session_state.report = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================
# Top Navigation Tabs
# ==========================
tabs = st.tabs(["Home", "Chat", "Dashboard", "About Us"])
home_tab, chat_tab, dashboard_tab, about_tab = tabs

# ==========================
# PAGE 1: HOME (Personality Test)
# ==========================
with home_tab:
    # Hero Banner Section
    st.markdown(
        """
        <div style="
            position: relative;
            text-align: center;
            color: white;
            margin-bottom: 35px;
        ">
             <img src='data:image/jpeg;base64,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'
                 style='width:100%; border-radius:15px; max-height:400px; object-fit:cover;'/>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Generate Button
    if st.button("Start Personality Test", key="generate_test_home"):
        st.session_state.questions = generate_personality_questions()
        # initialize answers to None for each question
        st.session_state.answers = [None] * len(st.session_state.questions)
        st.session_state.show_test = True

    # If test is active
    if st.session_state.get("show_test", False) and st.session_state.questions:
        st.markdown(
            "<h4 style='color:#4a00e0'>Please answer every question to get the most accurate personality assessment.</h4>",
            unsafe_allow_html=True
        )

        # Render questions (no duplicate button inside loop)
        for i, q in enumerate(st.session_state.questions):
            # Default style for each question block
            question_style = "border: 1px solid #ddd; padding:10px; border-radius:10px; margin-bottom:10px;"
            # Show question text as HTML block
            st.markdown(f'<div style="{question_style}">{q["question"]}</div>', unsafe_allow_html=True)

            # Show radio options (we keep label collapsed because question is shown above)
            selected = st.radio(
                label="",
                options=q["options"],
                key=f"q{i}",
                label_visibility="collapsed"
            )
            st.session_state.answers[i] = selected

        # Single Submit button below all questions
        if st.button("Submit Test", key="submit_test_home"):
            # validate
            unanswered = [idx for idx, an in enumerate(st.session_state.answers) if an is None or an == ""]
            if unanswered:
                st.error(f"Please answer all questions. Unanswered: {', '.join(str(i+1) for i in unanswered)}")
            else:
                mapping = {
                    "Strongly Disagree": 1,
                    "Disagree": 2,
                    "Neutral": 3,
                    "Agree": 4,
                    "Strongly Agree": 5
                }

                collected = []
                for ans_value, q in zip(st.session_state.answers, st.session_state.questions):
                    trait = q.get("trait", "Openness")
                    collected.append({
                        "trait": trait,
                        "value": mapping.get(ans_value, 3)
                    })

                # Generate results
                st.session_state.scores = score_answers(collected)
                st.session_state.report = generate_report(st.session_state.scores)

                # Store data for vector DB
                store_in_faiss(st.session_state.report)

                st.success("Test submitted! You can view results on the Dashboard tab.")
                # Optionally reveal dashboard or instruct user to click Dashboard tab

# ==========================
# PAGE 2: CHAT
# ==========================
with chat_tab:
    st.subheader("💬 Chat with Personality Coach")

    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    # Callback to send message
    def send_message():
        user_msg = st.session_state.chat_input.strip()
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "message": user_msg})
            ai_msg = rag_chat(user_msg)
            st.session_state.chat_history.append({"role": "ai", "message": ai_msg})
        st.session_state.chat_input = ""

    # CSS
    st.markdown("""
        <style>
        .chat-container {
            max-height: 600px;
            overflow-y: auto;
            padding: 10px;
            background: linear-gradient(180deg, #f3e5ff 0%, #e0bbff 100%);
            border-radius: 15px;
        }
        .chat-message {
            padding: 10px 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            max-width: 70%;
            word-wrap: break-word;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            font-size: 16px;
        }
        .user-message {
            background: linear-gradient(90deg, #8e2de2, #4a00e0);
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .ai-message {
            background: linear-gradient(90deg, #d0b3ff, #a27eff);
            color: #1a1a1a;
            margin-right: auto;
            text-align: left;
        }
        </style>
    """, unsafe_allow_html=True)

    # Render chat history
    chat_html = "<div class='chat-container'>"
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            chat_html += f"<div class='chat-message user-message'>{chat['message']}</div>"
        else:
            chat_html += f"<div class='chat-message ai-message'>{chat['message']}</div>"
    chat_html += "</div>"

    st.markdown(chat_html, unsafe_allow_html=True)

    st.text_input(
        "Type your message here...",
        key="chat_input",
        on_change=send_message,
    )

# ==========================
# PAGE 3: DASHBOARD
# ==========================
with dashboard_tab:
    if not st.session_state.get('scores'):
        st.warning("Please complete the test first.")
    else:
        st.subheader("📊 Personality Dashboard")

        # --- Big Five Bar Chart ---
        traits = list(st.session_state.scores.keys())
        scores = list(st.session_state.scores.values())
        bar_colors = ['#ff7eb9','#ff758c','#ff9472','#ffcc70','#70d6ff']

        fig_bar = go.Figure([go.Bar(x=traits, y=scores, marker_color=bar_colors)])
        fig_bar.update_layout(title="Big Five Trait Scores", yaxis=dict(range=[0,100]))
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- Pie Chart Example ---
        fig_pie = go.Figure(data=[go.Pie(labels=traits, values=scores, hole=0.3)])
        fig_pie.update_traces(marker=dict(colors=bar_colors))
        fig_pie.update_layout(title="Trait Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

        # --- Stacked Area Chart Example ---
        df_area = pd.DataFrame({
            'Traits': traits,
            'Score1': [s*0.8 for s in scores],
            'Score2': [s*0.6 for s in scores],
            'Score3': [s*0.4 for s in scores]
        })
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(x=df_area['Traits'], y=df_area['Score1'], fill='tozeroy', name='Level 1'))
        fig_area.add_trace(go.Scatter(x=df_area['Traits'], y=df_area['Score2'], fill='tonexty', name='Level 2'))
        fig_area.add_trace(go.Scatter(x=df_area['Traits'], y=df_area['Score3'], fill='tonexty', name='Level 3'))
        fig_area.update_layout(title="Trait Progression Over Levels")
        st.plotly_chart(fig_area, use_container_width=True)

        # --- Donut Chart Example ---
        fig_donut = go.Figure(data=[go.Pie(labels=traits, values=scores, hole=0.5)])
        fig_donut.update_traces(marker=dict(colors=bar_colors))
        fig_donut.update_layout(title="Donut Chart of Traits")
        st.plotly_chart(fig_donut, use_container_width=True)

        # --- Optional: Stacked Bar Example ---
        df_stacked = pd.DataFrame({
            'Traits': traits,
            'Part1': [s*0.5 for s in scores],
            'Part2': [s*0.3 for s in scores],
            'Part3': [s*0.2 for s in scores]
        })
        fig_stacked = go.Figure()
        fig_stacked.add_trace(go.Bar(x=df_stacked['Traits'], y=df_stacked['Part1'], name='Part 1'))
        fig_stacked.add_trace(go.Bar(x=df_stacked['Traits'], y=df_stacked['Part2'], name='Part 2'))
        fig_stacked.add_trace(go.Bar(x=df_stacked['Traits'], y=df_stacked['Part3'], name='Part 3'))
        fig_stacked.update_layout(barmode='stack', title="Stacked Trait Components")
        st.plotly_chart(fig_stacked, use_container_width=True)

        # --- Personality Summary & Recommendations ---
        st.markdown("### Summary & Recommendations")
        st.markdown(st.session_state.report.replace("\n", "  \n"))

# ==========================
# PAGE 4: ABOUT US
# ==========================
with about_tab:
    st.subheader("About Us")

    # Header
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #7F00FF, #E100FF);
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        ">
        Meet Our Team
        </div>
        """,
        unsafe_allow_html=True
    )

    team_members = [
        {"name": "Rao Muhammad Noman Farooq", "role": "Developer", "image": "https://via.placeholder.com/100"},
        {"name": "Dr Sarfraz Bibi", "role": "Developer", "image": "https://via.placeholder.com/100"},
        {"name": "Rafia Kashif", "role": "Developer", "image": "https://via.placeholder.com/100"},
        {"name": "Ome Aiman Rasheed", "role": "Developer", "image": "https://via.placeholder.com/100"},
        {"name": "Marriam Imdad", "role": "Developer", "image": "https://via.placeholder.com/100"},
    ]

    # Build cards HTML
    cards_html = """
    <div style="
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: center;
        margin-top: 20px;
    ">
    """

    for member in team_members:
        cards_html += f"""
        <div style="
            background: linear-gradient(135deg, #B993D6, #8CA6DB);
            border-radius: 12px;
            padding: 15px;
            width: 230px;
            text-align: center;
            color: white;
            box-shadow: 3px 3px 15px rgba(0,0,0,0.2);
        ">
            <img src="{member['image']}" width="100" style="border-radius: 50%; margin-bottom: 10px;">
            <h3 style="margin: 5px; font-size: 18px;">{member['name']}</h3>
            <h4 style="margin: 5px; font-weight: normal; font-size: 15px;">{member['role']}</h4>
        </div>
        """

    cards_html += "</div>"

    # Use components.html to reliably render the markup
    import streamlit.components.v1 as components
    components.html(cards_html, height=420)
    
    # Project Description
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #A18CD1, #FBC2EB);
            padding: 20px;
            border-radius: 12px;
            margin-top: 30px;
            color: #1A1A1A;
        ">
            <h2 style="text-align:center;">About the Project</h2>
            <p style="font-size:16px; line-height:1.6;">
                The AI Personality Assessment & Insight System is a cutting-edge application that evaluates your Big Five personality traits using advanced AI models. 
                It provides personalized insights, actionable recommendations, and guidance for personal growth, career development, 
                emotional well-being, and social relationships. The interactive dashboard and chatbot allow users to explore their personality 
                traits in depth and receive customized advice.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
