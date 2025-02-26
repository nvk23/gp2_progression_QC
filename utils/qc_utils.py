import os
import sys
import subprocess
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from functools import reduce

from plotting import plot_km_curve, plot_interactive_visit_month, plot_interactive_first_vs_last, plot_baseline_scores, plot_duration_values
from data_prep import checkNull, subsetData, checkDup, mark_removed_rows, check_chronological_order, check_consistent, create_survival_df
from writeread import send_email, to_excel
from app_setup import AppConfig

def prep_merge(page, uploaded_df):
    # Make sure uploaded dataframe matches exact names to prep for merge
    missing_outcome, missing_req = page.check_required(uploaded_df)
    available_metrics = [item for item in page.OUTCOMES_DICT.values() if item not in set(missing_outcome)]

    if len(missing_outcome) > 0:
        st.session_state[f'{page.get_name()}_variable'] = [key for key, value in page.OUTCOMES_DICT.items() if value in available_metrics]
    if len(missing_outcome) == len(page.OUTCOMES_DICT.values()):
        st.error(f'All {page.get_name()} metric columns are missing: {missing_outcome}. \
                Please use the template sheet to add at least one of these columns and re-upload.')
        st.stop()
    if len(missing_req) > 0:
        st.error(f"You are currently missing {len(missing_req)} column(s) required to continue the QC process: __{', '.join(missing_req)}__.")
        st.stop()

    return missing_outcome, available_metrics

def compare_manifest(page, manifest):
    # Load GP2 Genotyping Data Master Key
    unequal_num, unequal_cat = manifest.compare_cols()
    unequal_cols = list(unequal_num + unequal_cat)

    # Need to fix any discrepancies between manifest and uploaded file before continuing
    if len(unequal_cols) > 0:
        st.error(
            f'Discrepancies were found between overlapping columns in the GP2 Manifest and your uploaded file in the following columns: {unequal_cols}. \
          __Would you like to continue with your uploaded values or the manifest values?__')

        diff_rows = manifest.find_diff(unequal_num, unequal_cat)
        st.dataframe(diff_rows, use_container_width=True)

        # User selection for which dataset to continue with when > 1 discrepancies
        uploaded1, uploaded2, uploaded3 = st.columns(3)
        st.session_state.continue_merge = uploaded2.selectbox('Continue with:', options=[
                                                              '', 'Uploaded Values', 'Manifest Values'], on_change=page.call_off, args=[f'{page.get_name()}_btn'])

        if st.session_state.continue_merge == '':
            st.stop()
        else:
            df = manifest.adjust_data(no_diff = False)
    else:
        df = manifest.adjust_data()
    return df

def gp2_ids(manifest, study_name):
    # Will print total count metrics at the top of the page
    count1, count2, count3 = st.columns(3)
    checkids1, checkids2 = st.columns([2, 0.5])

    # Check counts and GP2IDs
    all_ids, non_gp2, df = manifest.get_ids()

    if len(non_gp2) == all_ids:
        st.error(
            f'None of the clinical IDs are in GP2. Please check that your clinical IDs and selected GP2 Study Code ({study_name}) are correct.')
        st.stop()
    elif len(non_gp2) > 0:
        checkids1.warning(
            f'Warning: Some clinical IDs are not in the GP2 so the dataset. Dataset review will continue only with GP2 IDs.')
        count1.metric(label="Unique GP2 Clinical IDs", value = len(df.clinical_id.unique()))
        count2.metric(label="Clinical IDs Not Found in GP2",
                      value=len(non_gp2))
        count3.metric(label="Total Observations for GP2 IDs", value=len(df))

        view_missing_ids = checkids2.button('Review IDs not Found in GP2')
        if view_missing_ids:
            st.markdown('_Non-GP2 Clinical IDs:_')
            st.dataframe(non_gp2, use_container_width=True)
    else:
        # All IDs are in GP2
        count1.metric(label="Unique Clinical IDs", value=all_ids)
        count2.metric(label="GP2 IDs Found", value=all_ids)
        count3.metric(label="Total Observations", value=len(df))

def optional_cols(page, missing_outcome, df):
    # Check for missing optional columns from template
    init_cols1, init_cols2 = st.columns([2, 0.5])
    if len(missing_outcome) > 0:
        init_cols1.warning(f"Warning: The following optional columns are missing: _{', '.join(missing_outcome)}_. \
                Please use the template sheet if you would like to add these values or initialize with null values.")
        add_nulls = init_cols2.button(
            'Fill Columns with Null Values', on_click=page.call_on, args = [f'{page.get_name()}_add_nulls'])
        if st.session_state[f'{page.get_name()}_add_nulls']:
            page.nullify(df, missing_outcome)
        if add_nulls:
            st.markdown('_All Columns:_')
            st.dataframe(df, use_container_width=True)

def null_vals(page, available_metrics, df):
    # Checking for NaN values in each column and summing them
    df_nulls = page.check_nulls(df, available_metrics, df.columns)
    if len(df_nulls) > 0:
        st.error(
            f'There are missing entries in the following required columns. Please fill in the missing cells.')
        st.markdown('_Missing Required Values:_')

        # Display dataframe with missing value rows
        st.dataframe(df_nulls, use_container_width=True)
        st.stop()

def numeric_ranges(page, df):
    # Make sure numeric columns are in proper ranges - flag 25 and below for age columns & 0 for visit_month
    out_of_range = page.check_ranges(df)

    if out_of_range:
        st.error(f'We have detected values that are out of range in the column(s) _{", ".join(list(out_of_range.keys()))}_. \
                    Values in these columns should be numeric and between the following permitted ranges:')
        for col in out_of_range.keys():
            col_range = page.NUMERIC_RANGES.get(col, AppConfig.NUMERIC_RANGES.get(col, "Unknown Range"))
            st.markdown(f'* _{col} : {col_range}_')
            st.dataframe(df[df[col].isin(out_of_range[col])])
        st.stop()

def dup_med_vals(df):
    dup_cols = checkDup(df, ['clinical_id', 'visit_month', 'clinical_state_on_medication', 'dbs_status'])
    if len(dup_cols) > 0:
        dup_warn1, dup_warn2 = st.columns([2, 0.5])
        dup_warn1.warning(
            f'Warning: We have detected samples with duplicated visit months, clinical state on medication, and DBS status.\
            Please review data if this was unintended.')
        if dup_warn2.button('View Duplicate Visits'):
            st.markdown('_Duplicate Visits:_')
            st.dataframe(dup_cols, use_container_width=True)

def flag_ages(page, df):
    age_warn1, age_warn2 = st.columns([2, 0.5])
    age_flags = page.flag_ages(df, 25)
    if age_flags:
        age_warn1.warning(f'Warning: We have detected ages that are below 25 in the column(s) _{", ".join(list(age_flags.keys()))}_. \
                    Please check that this is correct.')
        if age_warn2.button('View Ages Below 25'):
            for col in age_flags.keys():
                st.markdown(f'* _{col}_')
                st.dataframe(df[df[col].isin(age_flags[col])])

def chronological_ages(df):
    not_chrono = check_chronological_order(df)

    if len(not_chrono) > 0:
        st.error(f'We have detected ages that are not in chronological order in PD entries.\
                The age values should be in the following increasing order: age_of_onset, age_at_diagnosis, age_at_baseline.')
        st.dataframe(not_chrono, use_container_width=True)
        st.stop()

def age_consistency(df):
    diff_baseline = check_consistent(df, 'age_at_baseline')
    diff_outcome = check_consistent(df, 'age_outcome')
    
    if len(diff_baseline) > 0:
        st.error(
                f'We have detected samples with inconsistent age_at_baseline values. Please correct this and re-upload.')
        st.markdown(f'_Samples with inconsistent age_at_baseline values:_')
        st.dataframe(diff_baseline, use_container_width=True)
        st.stop()
    elif len(diff_outcome) > 0:
        st.error(
                f'We have detected samples with inconsistent age_at_diagnosis or age_of_onset values. Please correct this and re-upload.')
        st.markdown(f'_Samples with inconsistent age_at_diagnosis or age_of_onset values:_')
        st.dataframe(diff_outcome, use_container_width=True)
        st.stop()

def med_values(page, df):
    invalid_med_vals = page.check_med_vals(df)
    if invalid_med_vals:
        st.error(f'Please make sure the following columns have the following permitted values to continue.')
        for col in invalid_med_vals.keys():
            st.error(f'* {col}: {page.MED_VALS[col]}')
            st.dataframe(invalid_med_vals[col])
        st.stop()
    return invalid_med_vals

def med_val_strata(page, invalid_med_vals, strata):
    if strata in page.MED_VALS.keys():
        if len(invalid_med_vals) > 0:
            st.error(
                f'Please make sure {strata} values are {page.MED_VALS[strata]} to continue.')
            st.markdown(f'_Fix the following {strata} values:_')
            st.dataframe(invalid_med_vals[strata], use_container_width=True)
            st.stop()

def visit_month_zero(page, df):
    # Warn if sample does not have visit_month = 0
    no_zero_month = page.check_visit_months(df)
    zero_warn1, zero_warn2 = st.columns([2, 0.5])

    if len(no_zero_month) > 0:
        zero_warn1.warning(
            f'Warning: We have detected samples with no visit month value of 0. Please review data if this was unintended.')
        if zero_warn2.button('View Samples'):
            st.markdown('_Samples Without Visit Month of 0:_')
            st.dataframe(no_zero_month, use_container_width=True)

def outcome_qc(df_subset, varname, qc_count1, qc_count2, qc_count3):
    nulls = checkNull(df_subset, varname)
    qc_count1.metric(label="Null Values", value=len(nulls))

    dups = checkDup(df_subset, list(df_subset.columns), drop_dup = False)
    qc_count2.metric(label="Duplicate Rows", value=len(dups))

    # Add buttons to check null and duplicate samples
    if len(nulls) > 0:
        view_null = qc_count3.button('Review Null Values')
        if view_null:
            st.markdown('_Null Values:_')
            st.dataframe(nulls, use_container_width=True)
    if len(dups) > 0:
        view_dup = qc_count3.button('Review Duplicate Rows')
        if view_dup:
            st.markdown('_Duplicate Rows:_')
            st.dataframe(dups, use_container_width=True)
            st.info('All duplicated rows will be merged, keeping the first unique entry.')

def cleaned_df(df_subset):
    # Merge on ID and visit_month to keep entries with least null values
    df_final = subsetData(df_subset,
                            ['GP2ID', 'visit_month'],
                            method='less_na')

    with st.expander('###### _Hover over the dataframe and search for values using the 🔎 in the top right. :red[Click here to hide window]_', expanded=True):
        st.dataframe(df_final, use_container_width=True)

        # Check for unequal duplicates that were removed and need user input
        unequal_dup_rows = checkDup(df_subset, ['clinical_id', 'visit_month'])

        # Highlight removed rows if any exist
        if len(unequal_dup_rows) > 0:
            st.info('Please note that the following rows, :red[highlighted in red], will be removed from the dataframe above to handle duplicate visit month entries per sample ID. \
                    Rows with less null values were prioritized, where applicable. If you disagree with the dropped column, please make adjustments to your data or add a note in the \
                    "comments" column. __All rows will still be sent to us, but some analyses require duplicated visit month removal.__')
            
            # Style dataframes to highlight deleted rows in red
            df_subset, styled_duplicates = mark_removed_rows(df_final, df_subset, unequal_dup_rows)
            st.dataframe(styled_duplicates)

    return df_final, df_subset

def check_strata(df_final, selected_strata):
    null_strata = checkNull(df_final, selected_strata)
    if selected_strata not in df_final.columns:
        st.error(
            'The selected stratifying variable is not in the data. Please select another variable to plot.')
        st.stop()
    elif len(null_strata) == len(df_final):
        st.error(
            'The selected stratifying variable only has null input values. Please select another variable to plot.')
        st.stop()

def plot_outcomes(page, df_final, varname, out_version, strata):
    if st.session_state[f'{page.get_name()}_plot_val']:
        # Check if only cross-sectional data
        if (df_final.visit_month == 0).all():
            # Cross-sectional bar plot at baseline
            plot_baseline_scores(df_final, varname, out_version, strata)
            plot_duration_values(df_final, varname, out_version)
        else:
            plot_interactive_visit_month(
                df_final, varname, strata)

            df_sv_temp = create_survival_df(
                df_final, 3, 'greater', varname, strata)
            df_sv_temp = df_sv_temp.drop(columns=['event', 'censored_month'])

            plot_interactive_first_vs_last(df_sv_temp, strata)

            if page.get_name() in ['cisi']:
                min_value = page.NUMERIC_RANGE[0]
                max_value = page.NUMERIC_RANGE[1]
            else:
                min_value = page.NUMERIC_RANGES[varname][0]
                max_value = page.NUMERIC_RANGES[varname][1]

            st.markdown(
                '#### Kaplan-Meier Curve for Reaching the Threshold Score')
            thresh1, thresh2, thresh3 = st.columns([1, 0.5, 1])
            direction = thresh1.radio(label='Direction', horizontal=True, options=[
                                        'Greater Than or Equal To', 'Less Than or Equal To'])
            threshold = thresh2.number_input(
                min_value=min_value, max_value=max_value, step=1, label='Threshold', value=3)
            st.write('###')
            df_sv = create_survival_df(
                df_final, threshold, direction, varname, strata)
            
            plot_km_curve(df_sv, strata, threshold, direction) # new interactive method

def review_sample(page, df_final):
    if st.session_state[f'{page.get_name()}_plot_val']:
        st.markdown('---------')
        st.markdown('### Review Individual Samples')

        # Select a GP2ID from the list
        selected_sample= st.selectbox(
            "Select GP2ID", df_final['GP2ID'].unique())
        # Select a GP2ID from the list
        if selected_sample:
            single_sample = df_final[df_final['GP2ID'] == selected_sample].drop(columns=df_final.filter(regex='_jittered$').columns)
            st.markdown('###### _Hover over the dataframe and search for values using the 🔎 in the top right._')
            st.dataframe(single_sample, use_container_width=True)

def qc_submit(page, df_final, df_subset, out_version, study_name):
    if st.session_state[f'{page.get_name()}_plot_val']:
        st.markdown('---------')
        st.markdown('### Data Submission')
        
        qc_yesno = st.selectbox("Does the variable QC look correct?",
                                        ["YES", "NO"],
                                        index=None)
        if qc_yesno == 'YES':
            st.info('Thank you! Please review the following options:')
            st.session_state[f'{page.get_name()}_data_chunks'].append(df_final)

            yes_col1, yes_col2, yes_col3 = st.columns(
                3)  # add download button?
            if yes_col1.button("QC another variable", use_container_width=True):
                st.session_state[f'{page.get_name()}_variable'].remove(out_version)
                page.call_off(f'{page.get_name()}_btn')

            yes_col2.button("Email Data to GP2 Clinical Data Coordinator",
                            use_container_width=True, on_click=page.call_on, args = [f'{page.get_name()}_send_email'])

            # necessary because of nested form/button
            if st.session_state[f'{page.get_name()}_send_email']:

                email_form = st.empty()
                with email_form.container(border=True):
                    st.write("#### :red[Send the following email?]")

                    st.markdown(
                        "__TO:__ Lietsel Jones (Member of GP2's Cohort Integration Working Group)")

                    version = dt.datetime.today().strftime('%Y-%m-%d')
                    st.markdown(
                        f'__ATTACHMENT:__ {version}_{study_name}_{page.get_name()}_qc.csv')
                    
                        # Submit full dataset with markers for rows removed in unequal duplicate removal
                    df_subset.sort_values(by=['GP2ID']).reset_index(drop = True, inplace = True)
                    st.dataframe(df_subset, use_container_width=True)

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
                    st.session_state[f'{page.get_name()}_send_email'] = False
                    send_email(study_name, 'send_data', contact_info={
                        'name': submitter_name, 'email': submitter_email}, data=df_subset, modality=f'{page.get_name()}')
                    email_form.empty()  # clear form from screen
                    st.success('Email sent, thank you!')

            excel_file, filename = to_excel(df=df_subset,
                                            studycode=study_name)
            yes_col3.download_button("Download your Data", data=excel_file, file_name=filename,
                                        mime="application/vnd.ms-excel", use_container_width=True)

        if qc_yesno == 'NO':
            st.error("Please change any unexpected values in your clinical data and reupload \
                        or get in touch with GP2's Cohort Integration Working Group if needed.")
            st.stop()