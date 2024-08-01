import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
from google.cloud import storage


def studycode_callback():
    st.session_state['keepcode'] = st.session_state['mycode']

def get_studycode(master_path):
    if 'studycodes' not in st.session_state:
        df = pd.read_csv(master_path)
        study_codes = df.iloc[:, 0].str.strip().dropna().unique().tolist()
        study_codes = sorted(study_codes, key=str.lower)
        st.session_state['studycodes'] = study_codes
    
    study_name = st.sidebar.selectbox(
        'Select your GP2 study code',
        st.session_state['studycodes'],
        key='mycode',
        index=None,
        on_change = studycode_callback)
    return study_name

def upload_data(bucket_name, data, destination):
    """Upload a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination)
    blob.upload_from_string(data)
    return "File successfully uploaded to GP2 storage system"

def read_file(data_file):
    if data_file.type == "text/csv":
        df = pd.read_csv(data_file, dtype={'clinical_id':'str', 'sample_id':'str'})
    elif data_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(data_file, sheet_name=0, dtype={'clinical_id':'str', 'sample_id':'str'})
    return (df)