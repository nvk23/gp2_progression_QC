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
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


def studycode_callback():
    st.session_state['keepcode'] = st.session_state['mycode']


def get_master():
    # Can improve speed by adding Master Key to bucket
    with open(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]) as f:
        scopes = ['https://www.googleapis.com/auth/drive']

        # Pull necessary information from JSON file
        get_json = json.load(f)
        file_ID = get_json['connections']['gsheets']['master_key_ID']
        credentials = service_account.Credentials.from_service_account_file(
            filename=f.name,
            scopes=scopes)

        # Initialize the Google Drive API client
        drive_service = build('drive', 'v3', credentials=credentials)

        # Request the file
        request = drive_service.files().get_media(fileId=file_ID)
        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)

        # Download contents
        done = False
        while not done:
            status, done = downloader.next_chunk()

        # Move the pointer to the beginning of the file
        file_content.seek(0)

        # Load the CSV
        master_key = pd.read_csv(file_content, low_memory=False)
    return master_key


def get_studycode():
    if 'study_name' not in st.session_state:
        # Create study code options for dropdown
        study_codes = st.session_state.master_key.study.str.strip().dropna().unique().tolist()
        study_codes = sorted(study_codes, key=str.lower)
        st.session_state['study_name'] = study_codes # sort out if session state variable is necessary

    study_name = st.sidebar.selectbox(
        'Select your GP2 study code',
        st.session_state['study_name'],
        key='mycode',
        index=None,
        on_change=studycode_callback)
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
        df = pd.read_csv(data_file, dtype={
                         'clinical_id': 'str', 'sample_id': 'str'})
    elif data_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(data_file, sheet_name=0, dtype={
                           'clinical_id': 'str', 'sample_id': 'str'})
    return (df)


def to_excel(df, studycode):
    version = dt.datetime.today().strftime('%Y-%m-%d')
    ext = "xlsx"

    filename = f"{version}_{studycode}_HY_qc.{ext}"

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name='clinical_progression')

    # writer.save()
    processed_data = output.getvalue()
    return processed_data, filename

# can add Captcha confirmation to avoid automated emails
# can add function to customize file name to send
# can auto add date/time to file name


def send_email(studycode, activity, contact_info, data=None, modality='Clinical'):
    if activity == 'send_data':
        subject = f'{studycode} has Attached QCed {modality} Data'
        body = f'Hey team,\n\n{contact_info["name"]} has finished QCing their {studycode} {modality} Data for our Progression Project. You can contact them at {contact_info["email"]}. See attachment below.'
    elif activity == 'upload':
        subject = f'{studycode} has uploaded {modality} Data to the bucket'
        body = f"Hey team,\n\n{studycode} has uploaded their QC'ed {modality} data to the bucket."

    with open(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]) as f:
        get_json = json.load(f)
        sender = get_json['email_data']['secrets']['sender']
        receiver = get_json['email_data']['secrets']['receiver']
        pwd = get_json['email_data']['secrets']['pwd']

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ", ".join(receiver)  # can be list of names
    msg['Subject'] = subject

    msg.attach(MIMEText(body))

    if len(data) > 0:
        # Convert the DataFrame to CSV in memory (as text)
        csv_content = data.to_csv(index=False)  # Create CSV string in memory

        # Make CSV attachment-compatible
        part = MIMEBase('application', "octet-stream")
        # Encode CSV string to bytes
        part.set_payload(csv_content.encode('utf-8'))

        # Encode in base64
        encoders.encode_base64(part)

        version = dt.datetime.today().strftime('%Y-%m-%d')
        part.add_header('Content-Disposition',
                        f'attachment; filename={version}_{studycode}_HY_qc.csv')
        msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender, pwd)
    server.sendmail(sender, receiver, msg.as_string())
    server.quit()


def upload_data(bucket_name, data, destination):
    """Upload a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination)
    blob.upload_from_string(data)
    return "File successfully uploaded to GP2 storage system"
