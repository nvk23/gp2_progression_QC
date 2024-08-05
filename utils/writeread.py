import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
import io
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders
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

# can add Captcha confirmation to avoid automated emails
# can add function to customize file name to send
# can auto add date/time to file name
def send_email(studycode, activity, data_path = None):
    if activity == 'send_data':
        subject = f'{studycode} has finished QCing Clinical Data'
        body = f"Hey team, \n{studycode} has finished QCing their clinical data. See attachment."
    elif activity == 'upload':
        subject = f'{studycode} has uploaded Clinical Data to the bucket'
        body = f"Hey team, \n{studycode} has uploaded their QC'ed clinical data to the bucket."

    with open(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]) as f:
        get_json = json.load(f)
        sender = get_json['email_data']['secrets']['sender']
        receiver = get_json['email_data']['secrets']['receiver']
        pwd = get_json['email_data']['secrets']['pwd'] # may need to create Google app password

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ", ".join(receiver) # can be list of names
    msg['Subject'] = subject

    msg.attach(MIMEText(body))

    if data_path: 
        # can make for-loop if multiple files
        part = MIMEBase('application', "octet-stream")
        with open(data_path, 'rb') as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        f'attachment; filename={studycode}_clinical_qc.csv')
        msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender, pwd)
    server.sendmail(sender, receiver, msg.as_string())
    server.quit()

    os.remove('data/tmp/*.csv') # may replace with temp files in the future

def upload_data(bucket_name, data, destination):
    """Upload a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination)
    blob.upload_from_string(data)
    return "File successfully uploaded to GP2 storage system"