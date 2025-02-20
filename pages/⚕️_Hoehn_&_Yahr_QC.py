import os
import sys
import subprocess
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from functools import reduce

from utils.plotting import plot_km_curve, plot_interactive_visit_month, plot_interactive_first_vs_last
from utils.data_prep import checkNull, subsetData, checkDup, mark_removed_rows, check_chronological_order, check_consistent, create_survival_df
from utils.writeread import read_file, get_master, get_studycode, send_email, to_excel
from utils.app_setup import AppConfig, HY
from utils.manifest_setup import ManifestConfig

# Page set-up
hy = HY('Hoehn & Yahr QC')
hy.config_page()
hy.config_HY()
hy.google_connect()

# Uploader set-up
data_file = hy.config_data_upload(hy.TEMPLATE_LINK)

# Google Drive Master Key access
if 'master_key' not in st.session_state:
    st.session_state.master_key = get_master()
study_name = get_studycode()

# Required inputs passed
if data_file is not None and study_name is not None:
    st.markdown('### Your Data Overview')
    uploaded_df = read_file(data_file)

    # Make sure uploaded dataframe matches exact names to prep for merge
    incorrect_cols, checked_cols = hy.missing_HY(uploaded_df)
    missing_outcome, missing_req = hy.check_required(uploaded_df)

    if len(incorrect_cols) > 0:
        st.warning(f"Please correct the column(s) __{', '.join(incorrect_cols)}__ in your uploaded data to match the following template options: _{', '.join(checked_cols)}_ to ensure proper data QC.")
    if len(missing_outcome) > 1:
        st.error(f'Both Hoehn and Yahr Stage columns are missing: {missing_outcome}. \
                Please use the template sheet to add at least one of these columns and re-upload.')
        st.stop()
    if len(missing_req) > 0:
        st.error(f"You are currently missing {len(missing_req)} column(s) required to continue the QC process: __{', '.join(missing_req)}__.")
        st.stop()
    
    # Load GP2 Genotyping Data Master Key
    manifest = ManifestConfig(uploaded_df, study_name)
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
                                                              '', 'Uploaded Values', 'Manifest Values'], on_change=hy.call_off, args=['btn'])

        if st.session_state.continue_merge == '':
            st.stop()
        else:
            df = manifest.adjust_data(no_diff = False)
    else:
        df = manifest.adjust_data()
    
    # Will print total count metrics at the top of the page
    count1, count2, count3 = st.columns(3)

    # Check counts and GP2IDs
    all_ids, non_gp2, df = manifest.get_ids()

    checkids1, checkids2 = st.columns([2, 0.5])
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

    # Check for missing optional columns from template
    missing_optional = hy.missing_optional(df)
    init_cols1, init_cols2 = st.columns([2, 0.5])
    if len(missing_optional) > 0:
        init_cols1.warning(f"Warning: The following optional columns are missing: _{', '.join(missing_optional)}_. \
                Please use the template sheet if you would like to add these values or initialize with null values.")
        add_nulls = init_cols2.button(
            'Fill Columns with Null Values', on_click=hy.call_on, args = ['add_nulls'])
        if st.session_state['add_nulls']:
            hy.nullify(df, missing_optional)
        if add_nulls:
            st.markdown('_All Columns:_')
            st.dataframe(df, use_container_width=True)

    # Regardless of user selection add "clinical_state_on_medication" column if don't exist already
    hy.nullify(df, ['clinical_state_on_medication', 'dbs_status'])

    # Create column that combines values from age of diagnosis first then age of onset
    hy.add_age_outcome(df) 

    # Checking for NaN values in each column and summing them
    df_nulls = hy.check_nulls(df)
    if len(df_nulls) > 0:
        st.error(
            f'There are missing entries in the following required columns. Please fill in the missing cells.\
            __Reminder that age_of_onset is only required if age_at_diagnosis is unavailable.__')
        st.markdown('_Missing Required Values:_')

        # Display dataframe with rows only missing both age_at_diagnosis and age_of_onset
        st.dataframe(df_nulls, use_container_width=True)
        st.stop()

    # Make sure the clnical_id - visit_month combination is unique (warning if not unique)
    dup_cols = checkDup(df, ['clinical_id', 'visit_month', 'clinical_state_on_medication', 'dbs_status'])
    if len(dup_cols) > 0:
        dup_warn1, dup_warn2 = st.columns([2, 0.5])
        dup_warn1.warning(
            f'Warning: We have detected samples with duplicated visit months, clinical state on medication, and DBS status.\
            Please review data if this was unintended.')
        if dup_warn2.button('View Duplicate Visits'):
            st.markdown('_Duplicate Visits:_')
            st.dataframe(dup_cols, use_container_width=True)

    # Make sure numeric columns are in proper ranges - flag 25 and below for age columns & 0 for visit_month
    out_of_range = hy.check_ranges(df)
    age_flags = hy.flag_ages(df, 25)

    if out_of_range:
        st.error(f'We have detected values that are out of range in the column(s) _{", ".join(list(out_of_range.keys()))}_. \
                    Values in these columns should be numeric and between the following permitted ranges:')
        for col in out_of_range.keys():
            col_range = HY.NUMERIC_RANGES.get(col, AppConfig.NUMERIC_RANGES.get(col, "Unknown Range"))
            st.markdown(f'* _{col} : {col_range}_')
            st.dataframe(df[df[col].isin(out_of_range[col])])
        st.stop()

    # Warn if sample does not have visit_month = 0
    no_zero_month = hy.check_visit_months(df)
    zero_warn1, zero_warn2 = st.columns([2, 0.5])

    if len(no_zero_month) > 0:
        zero_warn1.warning(
            f'Warning: We have detected samples with no visit month value of 0. Please review data if this was unintended.')
        if zero_warn2.button('View Samples'):
            st.markdown('_Samples Without Visit Month of 0:_')
            st.dataframe(no_zero_month, use_container_width=True)

    # Warn if sample has age values < 25
    age_warn1, age_warn2 = st.columns([2, 0.5])
    if age_flags:
        age_warn1.warning(f'Warning: We have detected ages that are below 25 in the column(s) _{", ".join(list(age_flags.keys()))}_. \
                    Please check that this is correct.')
        if age_warn2.button('View Ages Below 25'):
            for col in age_flags.keys():
                st.markdown(f'* _{col}_')
                st.dataframe(df[df[col].isin(age_flags[col])])

    # Make sure age cols are in chronological order unless Prodromal and non-PD Genetically Enriched
    not_chrono = check_chronological_order(df)

    if len(not_chrono) > 0:
        st.error(f'We have detected ages that are not in chronological order in PD entries.\
                The age values should be in the following increasing order: age_of_onset, age_at_diagnosis, age_at_baseline.')
        st.dataframe(not_chrono, use_container_width=True)
        st.stop()

    # Make sure age_at_baseline is consistent for the same clinical IDs
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

    # Check that clinical variables have correct values if they're in the data
    invalid_med_vals = hy.check_med_vals(df)
    if invalid_med_vals:
        st.error(f'Please make sure the following columns have the following permitted values to continue.')
        for col in invalid_med_vals.keys():
            st.error(f'* {col}: {hy.MED_VALS[col]}')
            st.dataframe(invalid_med_vals[col])
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
        b1 = hy_qc_col2.button("Continue", on_click=hy.call_on, args = ['btn'])
        get_varname = None
    else:
        hy_qc1.markdown('### HY-Specific Quality Control')
        st.session_state['counter'] += 1
        if len(st.session_state['variable']) >= 1:
            hy_version = hy_qc_col1.selectbox(
                "Choose an HY version", st.session_state['variable'], on_change=hy.call_off, args = ['btn'], label_visibility='collapsed')
            get_varname = hy.OUTCOMES_DICT[hy_version]
            hy_qc_col2.button("Continue", on_click = hy.call_on, args = ['btn'])
        else:
            st.markdown(
                '<p class="medium-font"> You have successfully QCed all clinical variables, thank you!</p>', unsafe_allow_html=True)

            final_df = reduce(lambda x, y: pd.merge(x, y,
                                                    on=['clinical_id',
                                                        'visit_month'],
                                                    how='outer'), st.session_state['data_chunks'])

    if st.session_state['btn']:

        # reorder columns for display
        df_subset = hy.reorder_cols(df)

        nulls = checkNull(df_subset, get_varname)
        hy_qc_count1.metric(label="Null Values", value=len(nulls))

        dups = checkDup(df_subset, list(df_subset.columns), drop_dup = False)
        hy_qc_count2.metric(label="Duplicate Rows", value=len(dups))

        # Add buttons to check null and duplicate samples
        if len(nulls) > 0:
            view_null = hy_qc_count3.button('Review Null Values')
            if view_null:
                st.markdown('_Null Values:_')
                st.dataframe(nulls, use_container_width=True)
        if len(dups) > 0:
            view_dup = hy_qc_count3.button('Review Duplicate Rows')
            if view_dup:
                st.markdown('_Duplicate Rows:_')
                st.dataframe(dups, use_container_width=True)
                st.info('All duplicated rows will be merged, keeping the first unique entry.')

        st.markdown('---------')

        st.markdown('### Visualize Cleaned Dataset')

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

        # may need to move this higher to the initial QC before HY-specific QC
        st.markdown('Select a stratifying variable to plot:')
        plot1, plot2, plot3 = st.columns(3)
        strata = plot1.selectbox("Select a stratifying variable to plot:", hy.STRAT_VALS.keys(
        ), index=0, label_visibility='collapsed', on_change=hy.call_off, args=['plot_val'])
        selected_strata = hy.STRAT_VALS[strata]

        # Make sure selected stratifying variable is in the dataframe
        null_strata = checkNull(df_final, selected_strata)
        if selected_strata not in df_final.columns:
            st.error(
                'The selected stratifying variable is not in the data. Please select another variable to plot.')
            st.stop()
        elif len(null_strata) == len(df_final):
            st.error(
                'The selected stratifying variable only has null input values. Please select another variable to plot.')
            st.stop()

        # Make sure stratifying selections include required values to continue
        if selected_strata in hy.MED_VALS.keys():
            if len(invalid_med_vals) > 0:
                st.error(
                    f'Please make sure {selected_strata} values are {hy.MED_VALS[selected_strata]} to continue.')
                st.markdown(f'_Fix the following {selected_strata} values:_')
                st.dataframe(invalid_med_vals[selected_strata], use_container_width=True)
                st.stop()

        plot2.button('Continue', key='continue_plot', on_click = hy.call_on, args = ['plot_val'])
            
        if st.session_state['plot_val']:
            plot_interactive_visit_month(
                df_final, get_varname, selected_strata)

            df_sv_temp = create_survival_df(
                df_final, 3, 'greater', get_varname, selected_strata)
            df_sv_temp = df_sv_temp.drop(columns=['event', 'censored_month'])

            plot_interactive_first_vs_last(df_sv_temp, selected_strata)

            # using df_sv, event and censored_months, generate the show KM curve stratified by strata
            # take a threshold input

            min_value = hy.NUMERIC_RANGES[get_varname][0]
            max_value = hy.NUMERIC_RANGES[get_varname][1]

            st.markdown(
                '#### Kaplan-Meier Curve for Reaching the Threshold Score')
            thresh1, thresh2, thresh3 = st.columns([1, 0.5, 1])
            direction = thresh1.radio(label='Direction', horizontal=True, options=[
                                        'Greater Than or Equal To', 'Less Than or Equal To'])
            threshold = thresh2.number_input(
                min_value=min_value, max_value=max_value, step=1, label='Threshold', value=3)
            st.write('###')
            df_sv = create_survival_df(
                df_final, threshold, direction, get_varname, selected_strata)
            
            plot_km_curve(df_sv, selected_strata, threshold, direction) # new interactive method

            st.markdown('---------')
            st.markdown('### Review Individual Samples')

            # Select a GP2ID from the list
            selected_gp2id = st.selectbox(
                "Select GP2ID", df_final['GP2ID'].unique())

            if selected_gp2id:
                single_sample = df_final[df_final['GP2ID'] == selected_gp2id].drop(columns=df_final.filter(regex='_jittered$').columns)
                st.markdown('###### _Hover over the dataframe and search for values using the 🔎 in the top right._')
                st.dataframe(single_sample, use_container_width=True)

            st.markdown('---------')
            st.markdown('### Data Submission')

            qc_yesno = st.selectbox("Does the variable QC look correct?",
                                    ["YES", "NO"],
                                    index=None)

            if qc_yesno == 'YES':
                st.info('Thank you! Please review the following options:')
                st.session_state['data_chunks'].append(df_final)

                yes_col1, yes_col2, yes_col3 = st.columns(
                    3)  # add download button?
                if yes_col1.button("QC another variable", use_container_width=True):
                    # will not work until Modified HY Field is added
                    st.session_state['variable'].remove(hy_version)
                    hy.call_off('btn')

                yes_col2.button("Email Data to GP2 Clinical Data Coordinator",
                                use_container_width=True, on_click=hy.call_on, args = ['send_email'])

                # necessary because of nested form/button
                if st.session_state['send_email']:

                    email_form = st.empty()
                    with email_form.container(border=True):
                        st.write("#### :red[Send the following email?]")

                        st.markdown(
                            "__TO:__ Lietsel Jones (Member of GP2's Cohort Integration Working Group)")

                        version = dt.datetime.today().strftime('%Y-%m-%d')
                        st.markdown(
                            f'__ATTACHMENT:__ {version}_{study_name}_HY_qc.csv')
                        
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
                        st.session_state.send_email = False
                        send_email(study_name, 'send_data', contact_info={
                            'name': submitter_name, 'email': submitter_email}, data=df_subset, modality='HY')
                        email_form.empty()  # clear form from screen
                        st.success('Email sent, thank you!')

                excel_file, filename = to_excel(df=df,
                                                studycode=study_name)
                yes_col3.download_button("Download your Data", data=excel_file, file_name=filename,
                                            mime="application/vnd.ms-excel", use_container_width=True)

            if qc_yesno == 'NO':
                st.error("Please change any unexpected values in your clinical data and reupload \
                            or get in touch with GP2's Cohort Integration Working Group if needed.")
                st.stop()
