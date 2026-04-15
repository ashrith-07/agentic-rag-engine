import streamlit as st

def apply_minimal_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif !important;
        }
        
        .stButton>button {
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease-in-out;
        }
        
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }
        
        div[data-testid="stMetricValue"] {
            font-weight: 600 !important;
            color: #0f172a !important;
        }
        
        div[data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-weight: 500 !important;
        }
        
        div[data-testid="stSidebar"] {
            border-right: 1px solid #e2e8f0;
        }
        
        .stAlert {
            border-radius: 8px !important;
            border: none !important;
        }
        
        .stDataFrame {
            border-radius: 8px !important;
            overflow: hidden !important;
            border: 1px solid #e2e8f0 !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #0f172a !important;
            font-weight: 600 !important;
            letter-spacing: -0.025em !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
