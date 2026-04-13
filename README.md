# AI Material Recommendation Tool

An AI-powered web application that acts as a **Virtual Materials Scientist**, helping engineers and designers choose the most suitable material based on natural language requirements.

## Live Demo

[Click Here to Try](https://materialrecommender-pfcqyzmocv2qbw78zmmjae.streamlit.app)

## Features

- Accepts engineering/material requirements in plain English
- Uses LLMs to extract material constraints and priorities
- Searches a materials property database
- Recommends the optimal material based on scoring/ranking
- Provides technical explanation for recommendation
- Suggests alternative material options
- Generates downloadable PDF recommendation reports

---

## Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **LLM Integration:** Groq API (LLaMA 3.1)  
- **Data Processing:** Pandas  
- **PDF Export:** ReportLab  

---

## How It Works

1. User enters requirements in natural language  
   Example:  
   > "I need a lightweight but strong material for aerospace use"

2. The LLM extracts:
   - Importance weights for material properties
   - Hard engineering constraints

3. The system:
   - Filters material database
   - Applies constraint relaxation if needed
   - Scores/ranks materials dynamically

4. Returns:
   - Best material recommendation
   - Retrieved properties
   - Explanation of why it was chosen

---

## Dataset

The application uses a materials dataset containing:

- Ultimate Tensile Strength (Su)
- Yield Strength (Sy)
- Elastic Modulus (E)
- Shear Modulus (G)
- Poisson Ratio (μ)
- Density (Ro)

---

## Installation / Local Setup

```bash
git clone <your-repo-link>
cd material-recommender
pip install -r requirements.txt
streamlit run frontend.py
