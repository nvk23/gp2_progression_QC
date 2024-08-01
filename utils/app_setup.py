import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO

# from google.cloud import storage

# config page with logo in browser tab
def config_page(title):
    st.set_page_config(
        page_title=title,
        page_icon='data/gp2_2-removebg.png',
        layout="wide",
    )