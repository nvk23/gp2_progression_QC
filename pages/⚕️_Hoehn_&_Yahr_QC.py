import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from functools import reduce

sys.path.append('utils')
from app_setup import config_page
from writeread import read_file, get_studycode, send_email, upload_data
from qcutils import checkNull, subsetData, checkDup, create_survival_df
from plotting import plot_km_curve, plot_interactive_visit_month, plot_interactive_first_vs_last


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ''
bucket_name = ''
bucket_destination = ''

config_page('Hoehn & Yahr QC')

# Necessary paths - Update Template
# template_link = 'https://docs.google.com/spreadsheets/d/1tTkVcfP8l37uN09vGMNWiPQKBESRQGCrTZLdvR7rVQw/edit?usp=sharing' # old link
template_link = 'https://docs.google.com/spreadsheets/d/1qexD8xKUaORH-kZjUPWl-1duc_PEwbg0pvvlXQ0OPbY/edit?usp=sharing'
data_file = st.sidebar.file_uploader("Upload Your clinical data (CSV/XLSX)", type=['xlsx', 'csv'])
master_path = 'data/master_key.csv'
study_name = get_studycode(master_path) # initializes study_name to None
# cols = ['clinical_id', 'visit_month', 'clinical_state_on_medication', 'hoehn_and_yahr_stage']
required_cols =  ['clinical_id', 'visit_month']
optional_cols =  ['clinical_state_on_medication', 'hoehn_and_yahr_stage']
outcomes = ['Original HY Scale', 'Modified HY Scale']
outcomes_dict = {'Original HY Scale': 'hoehn_and_yahr_stage'} # add Modified here when figure out template col name

# Necessary session state initializiation/methods - can move to app_setup
if 'data_chunks' not in st.session_state:
    st.session_state['data_chunks'] = []
if 'btn' not in st.session_state:
    st.session_state['btn'] = False
if 'plot_val' not in st.session_state:
    st.session_state['plot_val'] = False
if 'send_email' not in st.session_state:
    st.session_state['send_email'] = False
if 'upload_bucket' not in st.session_state:
    st.session_state['upload_bucket'] = False
if 'variable' not in st.session_state:
    st.session_state['variable'] = outcomes

# can move to app_setup
def callback1():
    st.session_state['btn'] = True
def plot_callback1():
    st.session_state['plot_val'] = True
def email_callback1():
    st.session_state['send_email'] = True
def upload_callback1():
    st.session_state['upload_bucket'] = True
def callback2():
    st.session_state['btn'] = False
def email_callback2():
    st.session_state['send_email'] = False
def upload_callback2():
    st.session_state['upload_bucket'] = False

# App set-up
st.markdown('## Hoehn and Yahr QC')

instructions = st.expander("##### :red[Getting Started]", expanded=True)
with instructions:
    st.markdown(f'__①__ Please download [the data dictionary and template]({template_link}). Data dictionary can be found in the 2nd tab.', unsafe_allow_html=True)
    st.markdown('__②__ Upload your clinical data consistent to the template & required fields in the left sidebar.')
    st.markdown('__③__ Select your GP2 Study Code.') # add feature that suggests study code name if file name found in master key - correct if wrong
st.markdown('---------')


if data_file is not None and study_name is not None:
    st.markdown('### Your Data Overview')
    df = read_file(data_file)

    # Load GP2 Genotyping Data Master Key
    dfg = pd.read_csv(master_path, low_memory=False)
    dfg = dfg.drop_duplicates(subset='GP2ID', keep='first')
    dfg = dfg[dfg.study==study_name].copy()
    df = pd.merge(df, dfg[['GP2ID', 'clinical_id', 'GP2_phenotype', 'study_arm', 'study_type']], on='clinical_id', how='left')
    
    # Check counts and GP2IDs
    count1, count2, count3 = st.columns(3)
    n = len(df.clinical_id.unique())

    id_not_in_GP2 = df[df.GP2ID.isnull()]['clinical_id'].unique()
    if len(id_not_in_GP2) == n:
        st.error(f'None of the clinical IDs are in the GP2. Please check that your clinical IDs and selected GP2_study_code ({study_name}) are correct.')
        st.stop()
    elif len(id_not_in_GP2) > 0:
        st.warning(f'Warning: Some clinical IDs are not in the GP2 so the dataset. Dataset review will continue only with GP2 IDs.')
        df= df[df.GP2ID.notnull()].copy()
        n = len(df.clinical_id.unique())
        count1.metric(label="Unique GP2 Clinical IDs", value=n)
        count2.metric(label="Total GP2 Observations", value=len(df))
        count3.metric(label="Clinical IDs Not Found in GP2", value=len(id_not_in_GP2))

        view_missing_ids = st.button('Review IDs not Found in GP2')
        if view_missing_ids:
            st.markdown('_Non-GP2 Clinical IDs:_')
            st.dataframe(id_not_in_GP2)
    else:
        # all IDs in GP2
        count1.metric(label="Unique Clinical IDs", value=n)
        count2.metric(label="Total Observations", value=len(df))

    # Check for missing columns compared to template
    missing_optional = np.setdiff1d(optional_cols, df.columns)
    missing_req = np.setdiff1d(required_cols, df.columns)
    if len(missing_optional)>0:
        st.warning(f'Warning: The following optional columns are missing: {missing_optional}. \
                Please use the template sheet if you would like to add these values.')
    if len(missing_req) > 0:
        st.error(f'The following required columns are missing: {missing_req}. \
                Please use the template sheet to add these columns and re-upload.')
        st.stop()

    
    # Required columns checks
    df_non_miss_check = df[required_cols].copy()
    if df_non_miss_check.isna().sum().sum()>0:
        st.error(f'There are missing entries in the required columns {required_cols}. Please fill in the missing cells ')
        st.markdown('_Missing Values:_')
        st.dataframe(df_non_miss_check[df_non_miss_check.isna().sum(1)>0])
        st.stop()

    # Make sure visit_month and sample_ids are in the right format
    try:
        df['visit_month'] = df['visit_month'].astype(int)
        stopapp=False
    except:
        st.error(f'We could not convert visit month to integer')
        st.error(f'Please check visit month refers to numeric month from Baseline')
        st.error(f'First ~20 columns with visit_month not converted to integer')
        st.write(df[df['visit_month'].apply(lambda x: not x.isnumeric())].head(20)) # make into dataframe 
        stopapp=True
    
    # Make sure the clnical_id - visit_month combination is unique (warning if not unique)
    if df.duplicated(subset=['clinical_id', 'visit_month']).sum()>0:
        dup_warn1, dup_warn2 = st.columns([2, 0.5])
        dup_warn1.warning(f'Warning: We have detected duplicated visit months within samples. Please review data if this was unintended.') # change warning message
        if dup_warn2.button('View Duplicate Visits'):
            st.markdown('_Duplicate Visits:_')
            st.dataframe(df[df[['clinical_id', 'visit_month']].duplicated(keep=False)])

    if stopapp:
        st.stop()
    
    # Make sure clinical vars are non-negative integers
    for col in outcomes_dict.values():
        if not df[df[col] < 0].shape[0] == 0:
            st.error(f'We have detected negative values on column {col}')
            st.error(f' This is likely to be a mistake on the data. Please, go back to the sample manifest and check')
            st.stop()

    st.success('Your clinical data has passed all required up-front checks!')
    st.markdown('---------')

    hy_qc1, hy_qc2 = st.columns(2)
    hy_qc_col1, hy_qc_col2, hy_qc_col3 = st.columns(3)
    hy_qc_count1, hy_qc_count2, hy_qc_count3 = st.columns(3)
    
    if 'counter' not in st.session_state:
        hy_qc1.markdown('### HY-Specific Quality Control')
        st.session_state['counter'] = 0
        hy_qc_col1.selectbox("Choose an HY version", st.session_state['variable'], label_visibility='collapsed')
        b1 = hy_qc_col2.button("Continue", on_click=callback1)
        get_varname = None
    else:
        hy_qc1.markdown('### HY-Specific Quality Control')
        st.session_state['counter'] += 1
        if len(st.session_state['variable'])>=1:
            hy_version = hy_qc_col1.selectbox("Choose an HY version", st.session_state['variable'], on_change=callback2, label_visibility='collapsed')
            get_varname = outcomes_dict[hy_version]
            b1 = hy_qc_col2.button("Continue", on_click=callback1)
        else:
            st.markdown('<p class="medium-font"> THANKS. YOU HAVE SUCCESFULLY QCED all the CLINICAL VARIABLES</p>', unsafe_allow_html=True )

            final_df = reduce(lambda x, y: pd.merge(x, y, 
                                                    on = ['clinical_id', 'visit_month'],
                                                    how = 'outer'), st.session_state['data_chunks'])
            
            ## Move to the end if add final download button - remove aggrid dataframe
            # st.session_state['clinqc'] = final_df

            # aggridPlotter(final_df)
            # # df_builder = GridOptionsBuilder.from_dataframe(final_df)
            # # df_builder.configure_grid_options(alwaysShowHorizontalScroll = True,
            # #                                     enableRangeSelection=True,
            # #                                     pagination=True,
            # #                                     paginationPageSize=10000,
            # #                                     domLayout='normal')
            # # godf = df_builder.build()
            # # AgGrid(final_df,gridOptions=godf, theme='streamlit', height=300)

            # writeexcel = to_excel(final_df, st.session_state['keepcode'], datatype = 'clinical')
            # st.download_button(label='📥 Download your QC clinical data',
            #                     data = writeexcel[0],
            #                     file_name = writeexcel[1],)
            
            # st.stop()
       

    if st.session_state['btn']:
    #if st.button("Continue", on_click=callback1):
        vars = ['GP2ID', 'clinical_id', 'GP2_phenotype', 'study_arm', 'study_type', 'visit_month', get_varname]
        df_subset = df[vars].copy()

        nulls = checkNull(df_subset, get_varname)
        hy_qc_count1.metric(label="Null Values", value=len(nulls))

        dups = checkDup(df_subset, ['GP2ID', 'visit_month'])
        hy_qc_count2.metric(label="Duplicate Values", value=len(dups))

        # Add buttons to check null and duplicate samples
        if len(nulls) > 0:
            view_null = hy_qc_count3.button('Review Null Values')
            if view_null:
                st.markdown('_Null Values:_')
                st.dataframe(nulls)
        if len(dups) > 0:
            view_dup = hy_qc_count3.button('Review Duplicate Values')
            if view_dup:
                st.markdown('_Duplicate Values:_')
                st.dataframe(dups)

        st.markdown('---------')

        st.markdown('### Visualize Dataset')
        
        df_final =subsetData(df_subset, 
                            ['GP2ID', 'visit_month'],
                            method='less_na')

        with st.expander('###### _Subset of your data with minimal null values. :red[Click here to hide window]_', expanded = True):
            st.dataframe(df_final)
        
        st.markdown('Select a stratifying variable to plot:')
        plot1, plot2, plot3 = st.columns(3)
        strat_val = {'Study Arm': 'study_arm', 'GP2 Phenotype': 'GP2_phenotype'}
        strata = plot1.selectbox("Select a stratifying variable to plot:", strat_val.keys(), index = 0, label_visibility = 'collapsed')
        selected_strata = strat_val[strata]
        btn2 = plot2.button('Continue', key = 'continue_plot', on_click = plot_callback1)

        if st.session_state['plot_val']:
            plot_interactive_visit_month(df_final, get_varname, selected_strata)

            df_sv_temp = create_survival_df(df_final, 3, 'greater', get_varname)
            df_sv_temp = df_sv_temp.drop(columns=['event', 'censored_month'])
            plot_interactive_first_vs_last(df_sv_temp, df_final, selected_strata)

            # using df_sv, event and censored_months, generate the show KM curve stratifeid by strata
            # take a threshold input

            st.markdown('#### Kaplan-Meier Curve for Reaching the Threshold Score')
            thresh1, thresh2, thresh3 = st.columns([1, 0.5, 1])
            direction = thresh1.radio(label='Direction', horizontal = True, options=['Greater Than or Equal To', 'Less Than or Equal To'])
            threshold = thresh2.number_input(min_value=0, max_value=5, step=1, label='Threshold', value=3)
            st.write('###')
            df_sv = create_survival_df(df_final, threshold, direction, get_varname)
            plot_km_curve(df_sv, selected_strata, threshold, direction)

            st.markdown('---------')
            st.markdown('### Review Individual Samples')

            # Select a GP2ID from the list
            selected_gp2id = st.selectbox("Select GP2ID", df_final['GP2ID'].unique())

            if selected_gp2id:
                single_sample = df_final[df_final['GP2ID'] == selected_gp2id].drop(columns=df_final.filter(regex='_jittered$').columns)
                st.dataframe(single_sample)

            st.markdown('---------')
            st.markdown('### Data Submission')

            qc_yesno = st.selectbox("Does the variable QC look correct?", 
                                    ["YES", "NO"],
                                    index=None)
            if qc_yesno == 'YES':
                st.info('Thank you! Please review the following options:')
                st.session_state['data_chunks'].append(df_final)
                
                yes_col1, yes_col2, yes_col3 = st.columns(3) # add download button?
                if yes_col1.button("QC another variable", use_container_width = True):
                    st.session_state['variable'].remove(hy_version) # will not work until Modified HY Field is added
                    callback2()
                
                yes_col2.button("Email Data to GP2 Clinical Data Coordinator", use_container_width = True, on_click = email_callback1)
                if st.session_state['send_email']: # necessary because of nested form/button
                    tmp_path = f'data/tmp/{study_name}_final_qc.csv' # need bytes-like object to send email, not dataframe
                    df_final.to_csv(tmp_path, index = False) # no way to get path from file upload

                    # add form
                    with st.form("email_gp2"):
                        st.write("#### :red[Send the following email?]")
                        st.markdown("__TO:__ Lietsel Jones (Member of GP2's Cohort Integration Working Group)") # double check this title
                        
                        # can add "change message" option
                        st.markdown(f"__MESSAGE:__ Hi Lietsel, \nThe {study_name} team has finished QCing their clinical data. See attachment.")

                        st.markdown(f'__ATTACHMENT:__ {study_name}_clinical_qc.csv') # can add "change file name" option
                        st.dataframe(df_final)

                        send1, send2, send3 = st.columns(3)
                        submitted = send2.form_submit_button("Send", use_container_width = True)
                        if submitted:
                            send_email(study_name, 'send_data', tmp_path)
                            st.success('Email sent, thank you!')
                            email_callback2()

                yes_col3.button("Submit Data to GP2's Google Bucket", use_container_width = True, on_click = upload_callback1)
                if st.session_state['upload_bucket']:
                    # add form
                    with st.form("upload_gp2"):
                        st.write("#### :red[Upload the following data?]")

                        st.markdown(f"__GOOGLE CLOUD PROJECT NAME:__ ")
                        st.markdown("__BUCKET DESTINATION:__ ")

                        st.markdown(f'__FILE:__ {study_name}_clinical_qc.csv') # can add "change file name" option
                        st.dataframe(df_final)

                        send1, send2, send3 = st.columns(3)
                        submitted = send2.form_submit_button("Upload", use_container_width = True)
                        if submitted:
                            upload_data(bucket_name, df_final, bucket_destination) # currently do not have Google Bucket info
                            st.success('Data uploaded, thank you!')
                            upload_callback2()

            if qc_yesno == 'NO':
                st.error("Please change any unexpected values in your clinical data and reupload \
                         or get in touch with GP2's Cohort Integration Working Group (cohort@gp2.org) if needed.")
                st.stop()
