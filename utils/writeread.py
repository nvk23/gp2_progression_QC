import pandas as pd
import numpy as np
import datetime as dt
from io import BytesIO
import streamlit as st
import xlsxwriter
from google.cloud import storage
from streamlit_gsheets import GSheetsConnection
import os
import json
import smtplib
from email.mime.text import MIMEText


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

def to_excelv2(df,clin, dct):
    """It returns an excel object sheet with the QC sample manifest
    and clinical data written in separate
    """
    today = dt.datetime.today()
    version = f'{today.year}{today.month}{today.day}'
    study_code = df.study.unique()[0]
    ext = "xlsx"
    filename = "{s}_sample_manifest_selfQC_{v}.{e}".format(s=study_code, v = version, e = ext)
    
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='sm')
    clin.to_excel(writer, index=False, sheet_name='clinical')
    dct.to_excel(writer, index=False, sheet_name='Dictionary')
    writer.save()
    processed_data = output.getvalue()
    return processed_data, filename

def to_excel(df, studycode, mv, datatype):
    """It returns an excel object sheet with the QC sample manifest
    and clinical data written in separate
    """
    today = dt.datetime.today()
    version = f'{today.year}{today.month}{today.day}'
    ext = "xlsx"

    if datatype == 'sm':
        filename = "{s}_sample_manifest_selfQCV2_{v}_{m}.{e}".format(s=studycode, v = version, m = mv, e = ext)
    else:
        filename = "{s}_clinial_data_selfQC__{v}.{e}".format(s=studycode, v = version, e = ext)
    
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='sample_manifest')
    writer.save()
    processed_data = output.getvalue()
    return processed_data, filename


def email_ellie(studycode, activity):
    if activity == 'qc':
        subject = f'{studycode} has finished QCing the manifest'
        body = 'Hey team, \n Someone has finished QCing the manifest. They should upload to the bucket soon. \n Keep an eye if the don\'t'
    elif activity == 'upload':
        subject = f'{studycode} has uploaded the data to the bucket'
        body = 'Hey team, \n Someone has uploaded the sm to the bucket'
    else:
        st.error(f'{activity} not detected')
        st.stop()

    with open(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]) as f:
        get_json = json.load(f)
        sender = get_json['email_data']['secrets']['sender']
        receiver = get_json['email_data']['secrets']['receiver']
        pwd = get_json['email_data']['secrets']['pwd']
    
    msg = MIMEText(body)
    msg['From'] = sender
    msg['To'] = ", ".join(receiver)
    msg['Subject'] = subject

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender, pwd)
    server.sendmail(sender, receiver, msg.as_string())
    server.quit()


def read_file(data_file):
    if data_file.type == "text/csv":
        df = pd.read_csv(data_file, dtype={'clinical_id':'str', 'sample_id':'str'})
    elif data_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(data_file, sheet_name=0, dtype={'clinical_id':'str', 'sample_id':'str'})
    return (df)

def read_filev2(data_file):
    #if data_file.type == "text/csv":
        #dfdemo = pd.read_csv(data_file)
    if data_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        dfdemo = pd.read_excel(data_file, sheet_name=0)
        dfclin = pd.read_excel(data_file, sheet_name=1)
        try:
            dfdict = pd.read_excel(data_file, sheet_name=2)
        except:
            dfdict = None
    else:
        st.error("Please make sure you upload the temaplate excel file")
        st.stop()
    return (dfdemo, dfclin, dfdict)


# def to_excel(df):
#     """It returns an excel object sheet with the QC sample manifest
#     written in Sheet1
#     """
#     output = BytesIO()
#     writer = pd.ExcelWriter(output, engine='xlsxwriter')
#     df.to_excel(writer, index=False, sheet_name='Sheet1')
#     workbook = writer.book
#     worksheet = writer.sheets['Sheet1']
#     format1 = workbook.add_format({'num_format': '0.00'})
#     writer.save()
#     processed_data = output.getvalue()
#     return processed_data


# def read_file(data_file):
#     if data_file.type == "text/csv":
#         df = pd.read_csv(data_file)
#     elif data_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
#         df = pd.read_excel(data_file, sheet_name=0)
#     return (df)


# def output_create(df, filetype = "CSV"):
#     """It returns a tuple with the file content and the name to
#     write through a download button
#     """
#     today = dt.datetime.today()
#     version = f'{today.year}{today.month}{today.day}'
#     study_code = df.study.unique()[0]

#     if filetype == "CSV":
#         file = df.to_csv(index=False).encode()
#         ext = "csv"
#     elif filetype == "TSV":
#         file = df.to_csv(index=False, sep="\t").encode()
#         ext = "tsv"
#     else:
#         file = to_excel(df)
#         ext = "xlsx"
#     filename = "{s}_sample_manifest_selfQC_{v}.{e}".format(s=study_code, v = version, e = ext)

#     return (file, filename)


# def output_create(df, clin, filetype = "CSV"):
#     """It returns a tuple with the file content and the name to
#     write through a download button
#     """
#     today = dt.datetime.today()
#     version = f'{today.year}{today.month}{today.day}'
#     study_code = df.study.unique()[0]
#     if filetype == "CSV":
#         file = df.to_csv(index=False).encode()
#         ext = "csv"
#     elif filetype == "TSV":
#         file = df.to_csv(index=False, sep="\t").encode()
#         ext = "tsv"
#     else:
#         #file = df.to_excel(df)
#         ext = "xlsx"
#         filename = "{s}_sample_manifest_selfQC_{v}.{e}".format(s=study_code, v = version, e = ext)
#         with pd.ExcelWriter(filename) as file:  
#             df.to_excel(file, sheet_name='demographics')
#             clin.to_excel(file, sheet_name='clinical')
#     return (file, filename)