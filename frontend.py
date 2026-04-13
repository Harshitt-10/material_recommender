import streamlit as st
import pandas as pd
import json
from groq import Groq
from dotenv import load_dotenv
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

st.set_page_config(
    page_title="AI Material Recommendation Tool",
    layout="wide"
)

with st.sidebar:
    st.header("About")
    st.write("""
    This AI-powered system acts as a **Virtual Materials Scientist**:.

    - Understands natural language engineering requirements
    - Extracts design constraints using LLMs
    - Searches material property databases
    - Recommends optimal materials
    """)
    st.markdown("---")
    st.caption("Built by Harshit Karnani")

st.markdown("""
<style>

[data-testid="stToolbar"] {
    display: none;
}
#MainMenu {
    visibility: hidden;
}
footer {
    visibility: hidden;
}
header {
    visibility: hidden;
}

.main {
    padding-top: 2rem;
}

.stTextArea textarea {
    font-size: 18px;
}

/* Main Recommendation Box */
.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #1e1e1e;
    border: 1px solid #444;
    margin-top: 20px;
    color: white;
    font-size: 16px;
    font-weight: 500;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

/* Alternative Material Boxes */
.alt-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #2b2b2b;
    border: 1px solid #555;
    margin-bottom: 10px;
    color: white;
    font-size: 15px;
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)
# ==============================
# LOAD ENV
# ==============================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)


# ==============================
# HELPER FUNCTION
# ==============================
def apply_constraints(dataframe, constraints):
    filtered_df = dataframe.copy()

    if constraints.get("Ro_max") is not None:
        filtered_df = filtered_df[filtered_df["Ro"] <= constraints["Ro_max"]]

    if constraints.get("Su_min") is not None:
        filtered_df = filtered_df[filtered_df["Su"] >= constraints["Su_min"]]

    if constraints.get("Sy_min") is not None:
        filtered_df = filtered_df[filtered_df["Sy"] >= constraints["Sy_min"]]

    if constraints.get("E_min") is not None:
        filtered_df = filtered_df[filtered_df["E"] >= constraints["E_min"]]

    if constraints.get("G_min") is not None:
        filtered_df = filtered_df[filtered_df["G"] >= constraints["G_min"]]

    return filtered_df


def generate_pdf_report(content):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    story = []

    for line in content.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 12))

    doc.build(story)

    buffer.seek(0)
    return buffer


# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("Data.csv")
df = df[["Material", "Su", "Sy", "E", "G", "mu", "Ro"]]

numeric_cols = ["Su", "Sy", "E", "G", "mu", "Ro"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

raw_df = df.copy()

for col in numeric_cols:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())


# ==============================
# UI
# ==============================
st.markdown("""
<h1 style='text-align: center; color: white;'>
AI Material Recommendation Tool
</h1>
""", unsafe_allow_html=True)
st.markdown("""
<h3 style='text-align: center; color: white;'>
Your Virtual Materials Scientist
</h3>
""", unsafe_allow_html=True)
st.markdown("""
Describe your engineering/material requirements below and get intelligent recommendations.
""")

user_requirement = st.text_area("Enter Requirements")


if st.button("Recommend Material"):

    if not user_requirement.strip():
        st.warning("Please enter requirements.")
        st.stop()

    with st.spinner("🔍 Analyzing engineering constraints and searching database..."):

        # ------------------------------
        # LLM EXTRACTION
        # ------------------------------
        prompt = f"""
        You are an expert materials engineering assistant.

        Analyze the user's requirement and return:

        1. Importance weights (-1 to 1) for:
        Su, Sy, E, G, mu, Ro

        2. Use hard constraints sparingly.
        Prefer weights over constraints unless requirement explicitly demands minimum thresholds.

        RULES:
        - Lightweight means Ro should ALWAYS have NEGATIVE weight.
        - Strong/heavy duty means Su/Sy POSITIVE.
        - Stiff means E POSITIVE.
        - Shear resistant means G POSITIVE.
        - Poisson ratio (mu) should almost always be 0 unless explicitly requested.
        - NEVER assign negative weight to Su/Sy/E/G unless explicitly asked.
        - NEVER assign positive Ro for lightweight.
        - Constraints should be MODERATE, not extreme.
        - Avoid using 1.0/0.0 unless absolutely necessary.
        - Return ONLY JSON.
        - No explanation.

        Format:
        {{
        "weights": {{
            "Su": value,
            "Sy": value,
            "E": value,
            "G": value,
            "mu": value,
            "Ro": value
        }},
        "constraints": {{
            "Ro_max": optional_value,
            "Su_min": optional_value,
            "Sy_min": optional_value,
            "E_min": optional_value,
            "G_min": optional_value
        }}
        }}

        User Requirement:
        {user_requirement}
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        output = response.choices[0].message.content
        output = output.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            st.error("Invalid LLM output")
            st.stop()

        weights = parsed["weights"]
        constraints = parsed["constraints"]

        # ------------------------------
        # VALIDATE
        # ------------------------------
        weights["Ro"] = min(weights["Ro"], 0)

        weights["Su"] = max(weights["Su"], 0)
        weights["Sy"] = max(weights["Sy"], 0)
        weights["E"] = max(weights["E"], 0)
        weights["G"] = max(weights["G"], 0)

        weights["mu"] = max(min(weights["mu"], 0.2), -0.2)

        for key in constraints:
            if constraints[key] is not None:
                constraints[key] = min(constraints[key], 0.6)

        # ------------------------------
        # FILTER + RELAX
        # ------------------------------
        original_constraints = constraints.copy()

        df_filtered = apply_constraints(df, constraints)

        relax_factor = 0.1
        relaxed = False

        while df_filtered.empty and relax_factor <= 0.8:
            relaxed = True

            relaxed_constraints = {}

            for key, value in original_constraints.items():

                if value is None:
                    relaxed_constraints[key] = None

                elif "_min" in key:
                    relaxed_constraints[key] = max(0, value - relax_factor)

                elif "_max" in key:
                    relaxed_constraints[key] = min(1, value + relax_factor)

            df_filtered = apply_constraints(df, relaxed_constraints)

            if not df_filtered.empty:
                constraints = relaxed_constraints

            relax_factor += 0.1

        working_df = df_filtered

        if working_df.empty:
            st.error("Still no materials found after relaxation.")
            st.stop()

        if relaxed:
            st.info("Found recommendations after relaxing constraints.")

        working_df = working_df.copy()

        # ------------------------------
        # SCORE
        # ------------------------------
        working_df["Score"] = 0

        for feature, weight in weights.items():
            working_df["Score"] += working_df[feature] * weight

        working_df = working_df.sort_values(by="Score", ascending=False)

        top_material = working_df.iloc[0]
        raw_top_material = raw_df[raw_df["Material"]
                                  == top_material["Material"]].iloc[0]

        # ------------------------------
        # EXPLANATION LLM
        # ------------------------------
        explanation_prompt = f"""
        You are a virtual materials scientist.

        Your role is to analyze engineering constraints and utilize material property databases
        to recommend the optimal material. Do NOT assume or invent missing values.
        Use ONLY provided properties.

        Given:

        USER REQUIREMENT:
        {user_requirement}

        TOP RECOMMENDED MATERIAL:
        {top_material['Material']}

        MATERIAL PROPERTIES:
        Ultimate Tensile Strength (Su): {raw_top_material['Su']} MPa
        Yield Strength (Sy): {raw_top_material['Sy']} MPa
        Elastic Modulus (E): {raw_top_material['E']} MPa
        Shear Modulus (G): {raw_top_material['G']} MPa
        Poisson Ratio (mu): {raw_top_material['mu']}
        Density (Ro): {raw_top_material['Ro']} kg/m³

        Return your response EXACTLY in this format:

        Recommendation: <material name>

        Properties Retrieved:
        - property 1
        - property 2
        - property 3

        Explanation:
        <Explain why this material best satisfies the user's requirements in 2-4 sentences>

        Be concise, professional, and engineering-focused.
        """

        explanation_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            messages=[{"role": "user", "content": explanation_prompt}]
        )

        final_output = explanation_response.choices[0].message.content

        # ------------------------------
        # DISPLAY
        # ------------------------------
        st.success("Recommendation Generated!")

        st.markdown("## Final Recommendation")
        st.markdown("---")

        st.markdown(
            f'<div class="result-box">{final_output}</div>',
            unsafe_allow_html=True
        )

        st.subheader("🔄 Other Strong Alternatives")

        for i in range(1, min(4, len(working_df))):
            st.markdown(
                f'<div class="alt-box">#{i}: {working_df.iloc[i]["Material"]}</div>',
                unsafe_allow_html=True
            )

        pdf_buffer = generate_pdf_report(final_output)

        st.download_button(
            label="📄 Download Recommendation PDF",
            data=pdf_buffer,
            file_name="material_recommendation_report.pdf",
            mime="application/pdf"
        )
