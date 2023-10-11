import streamlit as st
import base64
from utils.constants import *

st.set_page_config(page_title='Template' ,layout="wide",initial_sidebar_state="auto", page_icon='👧🏻') # always show the sidebar

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        
local_css("styles/styles_main.css")
    
# App Sidebar
with st.sidebar:
    # Description
    st.markdown("""
                # Chat with my AI assistant
                """)

    with st.expander("Click here to see FAQs"):
        st.info(
            """
            - What are her strengths and weaknesses?
            - What is her expected salary?
            - What is her latest project?
            - When can she start to work?
            - Tell me about her professional background
            - What is her skillset?
            - What is her contact?
            - What are her achievements?
            """
        )
        
    st.caption("© Made by Vicky Kuo 2023. All rights reserved.")
 
st.title("📝 Resume")

st.write(f"[Click here if it's blocked by your browser]({info['Resume']})")

with open("images/resume.pdf","rb") as f:
      base64_pdf = base64.b64encode(f.read()).decode('utf-8')
      pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1000mm" height="1000mm" type="application/pdf"></iframe>'
      st.markdown(pdf_display, unsafe_allow_html=True)
        
