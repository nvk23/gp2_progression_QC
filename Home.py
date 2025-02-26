import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import datetime


sys.path.append('utils')
from utils.app_setup import AppConfig, HY

app = AppConfig('Home', 'home')
app.config_page()

# Main title
st.markdown("<h2 style='text-align: center; color: #B8390E; font-family: Verdana; '>GP2 Progression Quality Control</h1>", unsafe_allow_html=True)

# Page formatting
sent1, sent2, sent3 = st.columns([1, 6, 1])  # holds brief overview sentences
exp1, exp2, exp3 = st.columns([1, 2, 1])  # holds expander for full description

sent2.markdown("<h5 style='text-align: center; '>Visualize and review your data prior to GP2 submission for downstream analyses.</h5>", unsafe_allow_html=True)
sent2.markdown("<h5 style='text-align: center; '>Please select a page marked with ⚕️ in the sidebar to begin.</h5>", unsafe_allow_html=True)
sent2.markdown("<h6 style='text-align: center; '>If an image or video does not load in the tabs below, please refresh the page.</h6>", unsafe_allow_html=True)

# Display expander with full project description
overview = exp2.expander("Available Metrics", expanded=False)
with overview:
    st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                padding-left:40px;
            }
            </style>
            ''', unsafe_allow_html=True)

    st.markdown("### _Hoehn and Yahr_")
    st.markdown('Hoehn and Yahr staging scale (HY scale) is one of the most widely accepted clinical staging \
                in Parkinson’s disease (PD) being used for about 50 years. The original version was a five-point scale \
                while the modified version later added 1.5 and 2.5. The comparison of the original vs modified versions is \
                shown in the following table:')
    st.image(
        'data/original_vs_modified_HY.png')
    
instructions = exp2.expander("Instructional Video", expanded=False)
with instructions:
    st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ul{
            padding-left:40px;
        }
        </style>
        ''', unsafe_allow_html=True)
    
    st.markdown("_Click to expand to full-screen in the bottom right corner:_")
    instruct_video = open('data/GP2_HY_app_tutorial.mov', 'rb')
    video_bytes = instruct_video.read()
    st.video(video_bytes)


# Customize text in Expander element
hvar = """ <script>
                var elements = window.parent.document.querySelectorAll('.streamlit-expanderHeader');
                elements[0].style.fontSize = 'large';
                elements[0].style.color = '#850101';
            </script>"""
components.html(hvar, height=0, width=0)