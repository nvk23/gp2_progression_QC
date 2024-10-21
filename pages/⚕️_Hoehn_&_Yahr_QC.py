import os
import sys
import subprocess
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from functools import reduce

from plotting import plot_km_curve, plot_interactive_visit_month, plot_interactive_first_vs_last
from qcutils import checkNull, subsetData, checkDup, create_survival_df
from writeread import read_file, get_master, get_studycode, send_email, to_excel, upload_data
from app_setup import config_page

sys.path.append('utils')


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secrets/secrets.json"

# Moving forward with email only for now
# bucket_name = ''
# bucket_destination = ''

config_page('Hoehn & Yahr QC')

# Necessary paths
template_link = 'https://docs.google.com/spreadsheets/d/1qexD8xKUaORH-kZjUPWl-1duc_PEwbg0pvvlXQ0OPbY/edit?usp=sharing'
data_file = st.sidebar.file_uploader(
    "Upload Your clinical data (CSV/XLSX)", type=['xlsx', 'csv'])

# Local master key access - replace with file in bucket
if 'master_key' not in st.session_state:
    st.session_state.master_key = get_master()
study_name = get_studycode()

# Establish necessary columns
required_cols = ['clinical_id', 'visit_month',
                 'age_at_baseline', 'age_of_onset', 'age_at_diagnosis']
# alert about any missing values
required_cols_check = ['clinical_id',
                       'visit_month', 'age_at_baseline', 'age_outcome']
age_cols = ['age_at_baseline', 'age_of_onset', 'age_at_diagnosis']

# can default all cols to None and delete dictionary if we prefer
optional_cols = {'clinical_state_on_medication': None,
                 'ledd_daily': None, 'medication_for_pd': None,  'dbs_status': None, }
med_vals = {'clinical_state_on_medication': ['ON', 'OFF', 'Unknown'],
            'medication_for_pd': ['Yes', 'No', 'Unknown'],
            'dbs_status': ['Yes', 'No', 'Unknown', 'Not applicable']}
outcomes_dict = {'Original HY Scale': 'hoehn_and_yahr_stage',
                 'Modified HY Scale': 'modified_hoehn_and_yahr_stage'}

# Necessary session state initializiation/methods - can move to app_setup
if 'data_chunks' not in st.session_state:
    st.session_state['data_chunks'] = []
if 'btn' not in st.session_state:
    st.session_state['btn'] = False
if 'plot_val' not in st.session_state:
    st.session_state['plot_val'] = False
if 'send_email' not in st.session_state:
    st.session_state['send_email'] = False
# if 'upload_bucket' not in st.session_state:
#     st.session_state['upload_bucket'] = False
if 'add_nulls' not in st.session_state:
    st.session_state['add_nulls'] = False
if 'variable' not in st.session_state:
    st.session_state['variable'] = list(outcomes_dict.keys())

# can move to app_setup


def callback1():
    st.session_state['btn'] = True


def null_callback1():
    st.session_state['add_nulls'] = True


def plot_callback1():
    st.session_state['plot_val'] = True


def email_callback1():
    st.session_state['send_email'] = True


def upload_callback1():
    st.session_state['upload_bucket'] = True


def callback2():
    st.session_state['btn'] = False


# App set-up
st.markdown('## Hoehn and Yahr QC')

instructions = st.expander("##### :red[Getting Started]", expanded=True)
with instructions:
    st.markdown(
        f'__①__ Please download [the data dictionary and template]({template_link}). Data dictionary can be found in the 2nd tab.', unsafe_allow_html=True)
    st.markdown(
        '__②__ Upload your clinical data consistent to the template & required fields in the left sidebar. If you recieve AxiosError 400, please re-upload until the issue resolves itself.')
    # Optional: add feature that suggests study code name if file name found in master key - correct if wrong
    st.markdown('__③__ Select your GP2 Study Code.')
st.markdown('---------')


if data_file is not None and study_name is not None:
    st.markdown('### Your Data Overview')
    df = read_file(data_file)

    # Load GP2 Genotyping Data Master Key
    dfg = st.session_state.master_key.drop_duplicates(
        subset='GP2ID', keep='first')
    dfg = dfg[dfg.study == study_name].copy()

    # make sure this is consistent among all manifest versions
    dfg.rename(columns={'age': 'age_at_baseline'}, inplace=True)
    df = pd.merge(df, dfg[['GP2ID', 'clinical_id', 'GP2_phenotype', 'age_at_baseline', 'age_of_onset',
                           'age_at_diagnosis', 'study_arm', 'study_type']], on='clinical_id', how='left', suffixes=('_uploaded', '_manifest'))

    # Check counts and GP2IDs
    count1, count2, count3 = st.columns(3)
    n = len(df.clinical_id.unique())

    id_not_in_GP2 = df[df.GP2ID.isnull()]['clinical_id'].unique()
    if len(id_not_in_GP2) == n:
        st.error(
            f'None of the clinical IDs are in the GP2. Please check that your clinical IDs and selected GP2 Study Code ({study_name}) are correct.')
        st.stop()
    elif len(id_not_in_GP2) > 0:
        st.warning(
            f'Warning: Some clinical IDs are not in the GP2 so the dataset. Dataset review will continue only with GP2 IDs.')
        df = df[df.GP2ID.notnull()].copy()
        n = len(df.clinical_id.unique())
        count1.metric(label="Unique GP2 Clinical IDs", value=n)
        count2.metric(label="Clinical IDs Not Found in GP2",
                      value=len(id_not_in_GP2))
        count3.metric(label="Total Observations for GP2 IDs", value=len(df))

        view_missing_ids = st.button('Review IDs not Found in GP2')
        if view_missing_ids:
            st.markdown('_Non-GP2 Clinical IDs:_')
            st.dataframe(id_not_in_GP2, use_container_width=True)
    else:
        # All IDs are in GP2
        count1.metric(label="Unique Clinical IDs", value=n)
        count2.metric(label="GP2 IDs Found", value=n)
        count3.metric(label="Total Observations", value=len(df))

    # Identify columns from manifest
    manifest_cols = [col for col in df.columns if col.endswith('_manifest')]

    # Identify equivalent columns from uploaded file
    uploaded_cols = [col.replace('_manifest', '_uploaded')
                     for col in manifest_cols]

    # Compare the corresponding columns
    unequal_cols = [col for x, y in zip(manifest_cols, uploaded_cols) if not (
        df[x] == df[y]).all() for col in (x, y)]

    # Need to fix any discrepancies between manifest and uploaded file before continuing
    if len(unequal_cols) > 0:
        st.error(
            f'Discrepancies were found between overlapping columns in the GP2 Manifest and your uploaded file in the following columns: {unequal_cols}. \
            __Would you like to continue with your uploaded values or the manifest values?__')
        unequal_cols.insert(0, 'clinical_id')
        unequal_cols.insert(1, 'GP2ID')
        st.dataframe(df[unequal_cols], use_container_width=True)
        uploaded1, uploaded2, uploaded3 = st.columns(3)
        continue_merge = uploaded2.selectbox('Continue with:', options=[
                                             '', 'Uploaded Values', 'Manifest Values'], index=0)

        if continue_merge == 'Uploaded Values':
            rename_uploaded = [col.split('_uploaded')[0]
                               for col in uploaded_cols]
            rename_cols = dict(zip(uploaded_cols, rename_uploaded))
            df.rename(columns=rename_cols, inplace=True)
            df.drop(columns=manifest_cols, inplace=True)
        elif continue_merge == 'Manifest Values':
            rename_manifest = [col.split('_manifest')[0]
                               for col in manifest_cols]
            rename_cols = dict(zip(manifest_cols, rename_manifest))
            df.rename(columns=rename_cols, inplace=True)
            df.drop(columns=uploaded_cols, inplace=True)
        else:
            st.stop()

    # Create column that combines values from age of diagnosis first then age of onset
    df['age_outcome'] = df['age_at_diagnosis'].combine_first(
        df['age_of_onset'])

    # Check for missing optional columns from template
    missing_optional = np.setdiff1d(list(optional_cols.keys()), df.columns)
    missing_req = np.setdiff1d(required_cols, df.columns)
    init_cols1, init_cols2 = st.columns([2, 0.5])
    if len(missing_optional) > 0:
        init_cols1.warning(f'Warning: The following optional columns are missing: {missing_optional}. \
                Please use the template sheet if you would like to add these values or initialize with null values.')
        add_nulls = init_cols2.button(
            'Fill Columns with Null Values', on_click=null_callback1)
        if st.session_state['add_nulls']:
            for col in missing_optional:
                df[col] = optional_cols[col]
        if add_nulls:
            st.markdown('_All Columns:_')
            st.dataframe(df, use_container_width=True)

    # Check for missing required columns from template
    if len(missing_req) > 0:
        init_cols1.error(f'The following required columns are missing: {missing_req}. \
                Please use the template sheet to add these columns and re-upload.')
        st.stop()

    # Check either/or HY-related required columns - may need to check if missing values in HY
    missing_HY = np.setdiff1d(list(outcomes_dict.values()), df.columns)
    if len(missing_HY) > 1:
        init_cols1.error(f'Both Hoehn and Yahr Stage columns are missing: {missing_HY}. \
                        Please use the template sheet to add at least one of these columns and re-upload.')
        st.stop()
    elif len(missing_HY) == 1:
        df[missing_HY] = None

    # Missing values in required columns - more efficient method may exist
    check_missing = pd.DataFrame(
        df[required_cols_check].value_counts(dropna=False).reset_index())

    # Checking for NaN values in each column and summing them
    missing_sums = {col: check_missing[check_missing[col].isna(
    )]['count'].sum() for col in required_cols_check}
    if sum(missing_sums.values()) > 0:
        st.error(
            f'There are missing entries in the required columns {required_cols}. Please fill in the missing cells.\
            __Reminder that age_of_onset is only required if age_at_diagnosis is unavailable and vice versa.__')
        st.markdown('_Missing Values (displayed first):_')

        # Sort the DataFrame so that NaN values appear first in each column
        missing_display = df[required_cols].sort_values(
            by=required_cols, key=lambda col: col.isna(), ascending=False)
        st.dataframe(missing_display, use_container_width=True)
        st.stop()

    # Make sure visit_month are in the right format
    try:
        df['visit_month'] = df['visit_month'].astype(int)
        stopapp = False
    except:
        st.error(f'We could not convert visit month to integer. Please check visit month refers to numeric month from Baseline.')
        st.markdown('_Non-Integer Visit Months:_')
        st.dataframe(df[df['visit_month'].apply(
            lambda x: not x.isnumeric())], use_container_width=True)
        stopapp = True

    # Make sure visit month is in the range -1200 <= y <= 1200
    df_range_check = df[(df.visit_month > 1200) | (df.visit_month < -1200)]
    if df_range_check.shape[0] > 0:
        st.error(
            f'Please keep the visit month greater than or equal to -1200 and less than or equal to 1200.')
        st.markdown('_Out-of-range Visit Months:_')
        st.dataframe(df_range_check, use_container_width=True)
        stopapp = True

    # Make sure the clnical_id - visit_month combination is unique (warning if not unique)
    if df.duplicated(subset=['clinical_id', 'visit_month']).sum() > 0:
        dup_warn1, dup_warn2 = st.columns([2, 0.5])
        dup_warn1.warning(
            f'Warning: We have detected duplicated visit months within samples. Please review data if this was unintended.')
        if dup_warn2.button('View Duplicate Visits'):
            st.markdown('_Duplicate Visits:_')
            if 'clinical_state_on_medication' in df.columns:
                st.dataframe(df[df[['clinical_id', 'visit_month',
                                    'clinical_state_on_medication']].duplicated(keep=False)], use_container_width=True)
            else:
                st.dataframe(
                    df[df[['clinical_id', 'visit_month']].duplicated(keep=False)], use_container_width=True)

    if stopapp:
        st.stop()

    # Make sure clinical vars are non-negative integers
    for col in outcomes_dict.values():
        if df[df[col] < 0].shape[0] > 0:
            st.error(
                f'We have detected negative values in the {col} column. This is likely to be a mistake in the data.')
            st.markdown(f'_Negative Values in column {col}:_')
            st.dataframe(df[df[col] < 0], use_container_width=True)
            st.stop()

    # Make sure ages are at least 25
    for col in age_cols:
        if df[df[col] < 25].shape[0] > 0:
            age_warn1, age_warn2 = st.columns([2, 0.5])
            age_warn1.warning(
                f'Warning: We have detected ages that are below 25 in the {col} column. Please check that this is correct.')
            if age_warn2.button('View Ages Below 25'):
                st.markdown(f'_Ages Below 25 in column {col}:_')
                st.dataframe(df[df[col] < 25], use_container_width=True)

    # Make sure LEDD dosage falls in specific range if column exists
    if 'ledd_daily' in df.columns:
        check_ledd = df[(df.ledd_daily < 0) | (df.ledd_daily > 10000)]
        if check_ledd.shape[0] > 0:
            st.warning(
                f'Warning: Pleaes keep Levadopa Equivalent Dosage values per day greater than or equal to 0 and less than or equal to 10,000.')
            st.markdown(f'_Daily LEDD Values out of this Range:_')
            st.dataframe(check_ledd, use_container_width=True)

    # Check that clinical variables have correct values if they're in the data
    optional_vars = [col for col in list(
        optional_cols.keys()) if col in df.columns]
    optional_vars.remove('ledd_daily')
    for col in optional_vars:
        wrong_med_vals = df[(~df[col].isin(med_vals[col]))
                            & (df[col].notnull())]
        if wrong_med_vals.shape[0] > 0:
            st.error(
                f'Please make sure {col} values are {med_vals[col]} to continue.')
            st.markdown(f'_Fix the following {col} values:_')
            st.dataframe(wrong_med_vals, use_container_width=True)
            st.stop()

    st.success('Your clinical data has passed all required up-front checks!')
    st.markdown('---------')

    hy_qc1, hy_qc2 = st.columns(2)
    hy_qc_col1, hy_qc_col2, hy_qc_col3 = st.columns(3)
    hy_qc_count1, hy_qc_count2, hy_qc_count3 = st.columns(3)

    if 'counter' not in st.session_state:
        hy_qc1.markdown('### HY-Specific Quality Control')
        st.session_state['counter'] = 0
        hy_qc_col1.selectbox(
            "Choose an HY version", st.session_state['variable'], label_visibility='collapsed')
        b1 = hy_qc_col2.button("Continue", on_click=callback1)
        get_varname = None
    else:
        hy_qc1.markdown('### HY-Specific Quality Control')
        st.session_state['counter'] += 1
        if len(st.session_state['variable']) >= 1:
            hy_version = hy_qc_col1.selectbox(
                "Choose an HY version", st.session_state['variable'], on_change=callback2, label_visibility='collapsed')
            get_varname = outcomes_dict[hy_version]
            b1 = hy_qc_col2.button("Continue", on_click=callback1)
        else:
            st.markdown(
                '<p class="medium-font"> You have successfully QCed all clinical variables, thank you!</p>', unsafe_allow_html=True)

            final_df = reduce(lambda x, y: pd.merge(x, y,
                                                    on=['clinical_id',
                                                        'visit_month'],
                                                    how='outer'), st.session_state['data_chunks'])

    if st.session_state['btn']:
        # if st.button("Continue", on_click=callback1):
        keep_vars = ['GP2ID', 'clinical_id', 'GP2_phenotype',
                     'study_arm', 'study_type', 'visit_month',  get_varname]
        keep_vars.extend(optional_vars)
        df_subset = df[keep_vars].copy()

        nulls = checkNull(df_subset, get_varname)
        hy_qc_count1.metric(label="Null Values", value=len(nulls))

        dups = checkDup(df_subset, ['GP2ID', 'visit_month'])
        hy_qc_count2.metric(label="Duplicate Values", value=len(dups))

        # Add buttons to check null and duplicate samples
        if len(nulls) > 0:
            view_null = hy_qc_count3.button('Review Null Values')
            if view_null:
                st.markdown('_Null Values:_')
                st.dataframe(nulls, use_container_width=True)
        if len(dups) > 0:
            view_dup = hy_qc_count3.button('Review Duplicate Values')
            if view_dup:
                st.markdown('_Duplicate Values:_')
                st.dataframe(dups, use_container_width=True)

        # Make sure either HY scale is in the range 0 <= y <= 5
        var_range_check = df[(df[get_varname] > 5) | (df[get_varname] < 0)]
        if var_range_check.shape[0] > 0:
            st.error(
                f'Please keep the {get_varname} value greater than or equal to 0 and less than or equal to 5.')
            st.markdown(f'_Out-of-range {get_varname} Values:_')
            st.dataframe(var_range_check, use_container_width=True)
            st.stop()

        st.markdown('---------')

        st.markdown('### Visualize Dataset')

        df_final = subsetData(df_subset,
                              ['GP2ID', 'visit_month'],
                              method='less_na')

        with st.expander('###### _Subset of your data with minimal null values. :red[Click here to hide window]_', expanded=True):
            st.dataframe(df_final, use_container_width=True)

        # may need to move this higher to the initial QC before HY-specific QC
        st.markdown('Select a stratifying variable to plot:')
        plot1, plot2, plot3 = st.columns(3)
        strat_val = {'Study Arm': 'study_arm', 'GP2 Phenotype': 'GP2_phenotype',
                     'Clinical State on Medication': 'clinical_state_on_medication',
                     'Medication for PD': 'medication_for_pd', 'DBS Stimulation': 'dbs_status'}
        strata = plot1.selectbox("Select a stratifying variable to plot:", strat_val.keys(
        ), index=0, label_visibility='collapsed')
        selected_strata = strat_val[strata]

        # Make sure selected stratifying variable is in the dataframe
        if selected_strata not in df_final.columns:
            st.error(
                'The selected stratifying variable is not in the data. Please select another variable to plot.')
            st.stop()

        # Make sure stratifying selections include required values to continue
        if selected_strata in med_vals.keys():
            wrong_med_vals = df_final[~df_final[selected_strata].isin(
                med_vals[selected_strata])]
            if wrong_med_vals.shape[0] > 0:
                st.error(
                    f'Please make sure {selected_strata} values are {med_vals[selected_strata]} to continue.')
                st.markdown(f'_Fix the following {selected_strata} values:_')
                st.dataframe(wrong_med_vals, use_container_width=True)
                st.stop()

        btn2 = plot2.button('Continue', key='continue_plot',
                            on_click=plot_callback1)
        if st.session_state['plot_val']:
            plot_interactive_visit_month(
                df_final, get_varname, selected_strata)

            df_sv_temp = create_survival_df(
                df_final, 3, 'greater', get_varname)
            df_sv_temp = df_sv_temp.drop(columns=['event', 'censored_month'])
            plot_interactive_first_vs_last(
                df_sv_temp, df_final, selected_strata)

            # using df_sv, event and censored_months, generate the show KM curve stratified by strata
            # take a threshold input

            st.markdown(
                '#### Kaplan-Meier Curve for Reaching the Threshold Score')
            thresh1, thresh2, thresh3 = st.columns([1, 0.5, 1])
            direction = thresh1.radio(label='Direction', horizontal=True, options=[
                                      'Greater Than or Equal To', 'Less Than or Equal To'])
            threshold = thresh2.number_input(
                min_value=0, max_value=5, step=1, label='Threshold', value=3)
            st.write('###')
            df_sv = create_survival_df(
                df_final, threshold, direction, get_varname)
            plot_km_curve(df_sv, selected_strata, threshold, direction)

            st.markdown('---------')
            st.markdown('### Review Individual Samples')

            # Select a GP2ID from the list
            selected_gp2id = st.selectbox(
                "Select GP2ID", df_final['GP2ID'].unique())

            if selected_gp2id:
                single_sample = df_final[df_final['GP2ID'] == selected_gp2id].drop(
                    columns=df_final.filter(regex='_jittered$').columns)
                st.dataframe(single_sample, use_container_width=True)

            st.markdown('---------')
            st.markdown('### Data Submission')

            qc_yesno = st.selectbox("Does the variable QC look correct?",
                                    ["YES", "NO"],
                                    index=None)

            # Will send df instead of df_final because df_final was subset to eliminate NAs
            if qc_yesno == 'YES':
                st.info('Thank you! Please review the following options:')
                st.session_state['data_chunks'].append(df_final)

                yes_col1, yes_col2, yes_col3 = st.columns(
                    3)  # add download button?
                if yes_col1.button("QC another variable", use_container_width=True):
                    # will not work until Modified HY Field is added
                    st.session_state['variable'].remove(hy_version)
                    callback2()

                yes_col2.button("Email Data to GP2 Clinical Data Coordinator",
                                use_container_width=True, on_click=email_callback1)

                # necessary because of nested form/button
                if st.session_state['send_email']:

                    email_form = st.empty()
                    with email_form.container(border=True):
                        st.write("#### :red[Send the following email?]")

                        st.markdown(
                            "__TO:__ Lietsel Jones (Member of GP2's Cohort Integration Working Group)")

                        version = dt.datetime.today().strftime('%Y-%m-%d')
                        st.markdown(
                            f'__ATTACHMENT:__ {version}_{study_name}_clinical_qc.csv')
                        st.dataframe(df, use_container_width=True)
                        st.markdown(
                            "_If you'd like, you can hover over the table above and click the :blue[Download] symbol in the top-right corner to save your QC'ed data as a :blue[CSV] with the filename above._")

                        st.markdown(f"__SUBMITTER CONTACT INFO:__ ")
                        submit1, submit2 = st.columns(2)
                        submitter_name = submit1.text_input(
                            label='Please provide your name:')
                        submitter_email = submit2.text_input(
                            label='Please provide your email:')  # verify if email format

                        send1, send2, send3 = st.columns(3)
                        if submitter_name and submitter_email:
                            submitted = send2.button(
                                "Send", use_container_width=True)
                        else:
                            submitted = send2.button(
                                "Send", use_container_width=True, disabled=True)
                    if submitted:
                        st.session_state.send_email = False
                        send_email(study_name, 'send_data', contact_info={
                            'name': submitter_name, 'email': submitter_email}, data=df)
                        email_form.empty()  # clear form from screen
                        st.success('Email sent, thank you!')

                excel_file, filename = to_excel(df=df,
                                                studycode=study_name)
                yes_col3.download_button("Download your Data", data=excel_file, file_name=filename,
                                         mime="application/vnd.ms-excel", use_container_width=True)

                # Currently moving forward with email-only submission
                # yes_col3.button("Submit Data to GP2's Google Bucket",
                #                 use_container_width=True, on_click=upload_callback1)
                # if st.session_state['upload_bucket']:
                #     upload_form = st.empty()

                #     # add form
                #     with upload_form.form("upload_gp2"):
                #         st.write("#### :red[Upload the following data?]")

                #         st.markdown(f"__GOOGLE CLOUD PROJECT NAME:__ ")
                #         st.markdown("__BUCKET DESTINATION:__ ")

                #         # can add "change file name" option
                #         st.markdown(f'__FILE:__ {study_name}_clinical_qc.csv')
                #         st.dataframe(df, use_container_width=True)

                #         send1, send2, send3 = st.columns(3)
                #         submitted = send2.form_submit_button(
                #             "Upload", use_container_width=True)
                #         if submitted:
                #             # currently do not have Google Bucket info
                #             upload_data(bucket_name, df,
                #                         bucket_destination)
                #             upload_form.empty()  # clear form from screen
                #             st.success('Data uploaded, thank you!')

            if qc_yesno == 'NO':
                st.error("Please change any unexpected values in your clinical data and reupload \
                         or get in touch with GP2's Cohort Integration Working Group if needed.")
                st.stop()
