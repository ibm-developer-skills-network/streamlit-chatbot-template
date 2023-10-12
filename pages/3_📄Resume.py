import streamlit as st
import base64
from utils.constants import *

st.set_page_config(page_title='Template' ,layout="wide",initial_sidebar_state="auto", page_icon='👧🏻') # always show the sidebar

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        
local_css("styles/styles_main.css")
    
# get the variables from constants.py
pronoun = info['Pronoun']

# app sidebar
with st.sidebar:
    st.markdown("""
                # Chat with my AI assistant
                """)
    with st.expander("Click here to see FAQs"):
        st.info(
            f"""
            - What are {pronoun} strengths and weaknesses?
            - What is {pronoun} expected salary?
            - What is {pronoun} latest project?
            - When can {pronoun} start to work?
            - Tell me about {pronoun} professional background
            - What is {pronoun} skillset?
            - What is {pronoun} contact?
            - What are {pronoun} achievements?
            """
        )
        
    st.caption(f"© Made by Vicky Kuo 2023. All rights reserved.")
 
st.title("📝 Resume")

st.write(f"[Click here if it's blocked by your browser]({info['Resume']})")

with open("images/resume.pdf","rb") as f:
      base64_pdf = base64.b64encode(f.read()).decode('utf-8')
      pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1000mm" height="1000mm" type="application/pdf"></iframe>'
      st.markdown(pdf_display, unsafe_allow_html=True)
        
