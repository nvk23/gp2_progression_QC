import os
import sys
import subprocess
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from functools import reduce

from utils.plotting import plot_baseline_scores, plot_km_curve, plot_interactive_visit_month, plot_interactive_first_vs_last
from utils.data_prep import checkNull, subsetData, checkDup, mark_removed_rows, check_chronological_order, check_consistent, create_survival_df
from utils.writeread import read_file, get_master, get_studycode, send_email, to_excel
from utils.app_setup import AppConfig, CISI
from utils.manifest_setup import ManifestConfig

# Page set-up
cisi = CISI('CISI-PD QC')
cisi.config_page()
cisi.config_CISI()
cisi.google_connect()

# Uploader set-up
data_file = cisi.config_data_upload(cisi.TEMPLATE_LINK)

# Google Drive Master Key access
if 'master_key' not in st.session_state:
    st.session_state.master_key = get_master()
study_name = get_studycode()

# Required inputs passed
if data_file is not None and study_name is not None:
    st.markdown('### Your Data Overview')
    uploaded_df = read_file(data_file)

    # Make sure uploaded dataframe matches exact names to prep for merge
    missing_outcome, missing_req = cisi.check_required(uploaded_df)
    available_metrics = [item for item in cisi.OUTCOMES_DICT.values() if item not in set(missing_outcome)]

    if len(missing_outcome) > 0:
        st.session_state['cisi_variable'] = [key for key, value in cisi.OUTCOMES_DICT.items() if value in available_metrics]
    if len(missing_outcome) == len(cisi.OUTCOMES_DICT.values()):
        st.error(f'All CISI-PD metric columns are missing: {missing_outcome}. \
                Please use the template sheet to add at least one of these columns and re-upload.')
        st.stop()
    if len(missing_req) > 0:
        st.error(f"You are currently missing {len(missing_req)} column(s) required to continue the QC process: __{', '.join(missing_req)}__.")
        st.stop()
    
    # Load GP2 Genotyping Data Master Key
    manifest = ManifestConfig(uploaded_df, study_name)
    unequal_num, unequal_cat = manifest.compare_cols()
    unequal_cols = set(unequal_num + unequal_cat)

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
                                                              '', 'Uploaded Values', 'Manifest Values'], on_change=cisi.call_off, args=['cisi_btn'])

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
    init_cols1, init_cols2 = st.columns([2, 0.5])
    if len(missing_outcome) > 0:
        init_cols1.warning(f"Warning: The following optional columns are missing: _{', '.join(missing_outcome)}_. \
                Please use the template sheet if you would like to add these values or initialize with null values.")
        add_nulls = init_cols2.button(
            'Fill Columns with Null Values', on_click=cisi.call_on, args = ['cisi_add_nulls'])
        if st.session_state['cisi_add_nulls']:
            cisi.nullify(df, missing_outcome)
        if add_nulls:
            st.markdown('_All Columns:_')
            st.dataframe(df, use_container_width=True)

    # Checking for NaN values in each column and summing them
    df_nulls = cisi.check_nulls(df, available_metrics, df.columns)
    if len(df_nulls) > 0:
        st.error(
            f'There are missing entries in the following required columns. Please fill in the missing cells.')
        st.markdown('_Missing Required Values:_')

        # Display dataframe with missing value rows
        st.dataframe(df_nulls, use_container_width=True)
        st.stop()

    # Make sure numeric columns are in proper ranges - flag 25 and below for age columns & 0 for visit_month
    out_of_range = cisi.check_ranges(df)

    if out_of_range:
        st.error(f'We have detected values that are out of range in the column(s) _{", ".join(list(out_of_range.keys()))}_. \
                    Values in these columns should be numeric and between the following permitted ranges:')
        for col in out_of_range.keys():
            col_range = cisi.NUMERIC_RANGES.get(col, AppConfig.NUMERIC_RANGES.get(col, "Unknown Range"))
            st.markdown(f'* _{col} : {col_range}_')
            st.dataframe(df[df[col].isin(out_of_range[col])])
        st.stop()

    # Warn if sample does not have visit_month = 0
    no_zero_month = cisi.check_visit_months(df)
    zero_warn1, zero_warn2 = st.columns([2, 0.5])

    if len(no_zero_month) > 0:
        zero_warn1.warning(
            f'Warning: We have detected samples with no visit month value of 0. Please review data if this was unintended.')
        if zero_warn2.button('View Samples'):
            st.markdown('_Samples Without Visit Month of 0:_')
            st.dataframe(no_zero_month, use_container_width=True)

    st.success('Your clinical data has passed all required up-front checks!')
    st.markdown('---------')

    cisi_qc1, cisi_qc2 = st.columns(2)
    cisi_qc_col1, cisi_qc_col2, cisi_qc_col3 = st.columns(3)
    cisi_qc_count1, cisi_qc_count2, cisi_qc_count3 = st.columns(3)

    if 'cisi_counter' not in st.session_state:
        cisi_qc1.markdown('### CISI-PD Quality Control')
        st.session_state['cisi_counter'] = 0
        cisi_qc_col1.selectbox(
            "Choose a CISI-PD metric", st.session_state['cisi_variable'], label_visibility='collapsed')
        b1 = cisi_qc_col2.button("Continue", on_click=cisi.call_off, args = ['cisi_btn'])
        get_varname = None
    else:
        cisi_qc1.markdown('### CISI-PD Quality Control')
        st.session_state['cisi_counter'] += 1
        if len(st.session_state['variable']) >= 1:
            cisi_version = cisi_qc_col1.selectbox(
                "Choose an cisi version", st.session_state['cisi_variable'], on_change=cisi.call_off, args = ['cisi_btn'], label_visibility='collapsed')
            get_varname = cisi.OUTCOMES_DICT[cisi_version]
            cisi_qc_col2.button("Continue", on_click = cisi.call_on, args = ['cisi_btn'])
        else:
            st.markdown(
                '<p class="medium-font"> You have successfully QCed all clinical variables, thank you!</p>', unsafe_allow_html=True)

            final_df = reduce(lambda x, y: pd.merge(x, y,
                                                    on=['clinical_id',
                                                        'visit_month'],
                                                    how='outer'), st.session_state['cisi_data_chunks'])
    if st.session_state['cisi_btn']:

        # reorder columns for display
        df_subset = cisi.reorder_cols(df)

        nulls = checkNull(df_subset, get_varname)
        cisi_qc_count1.metric(label="Null Values", value=len(nulls))

        dups = checkDup(df_subset, list(df_subset.columns), drop_dup = False)
        cisi_qc_count2.metric(label="Duplicate Rows", value=len(dups))

        # Add buttons to check null and duplicate samples
        if len(nulls) > 0:
            view_null = cisi_qc_count3.button('Review Null Values')
            if view_null:
                st.markdown('_Null Values:_')
                st.dataframe(nulls, use_container_width=True)
        if len(dups) > 0:
            view_dup = cisi_qc_count3.button('Review Duplicate Rows')
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

        # may need to move this higher to the initial QC before cisi-specific QC
        st.markdown('Select a stratifying variable to plot:')
        plot1, plot2, plot3 = st.columns(3)
        strata = plot1.selectbox("Select a stratifying variable to plot:", cisi.STRAT_VALS.keys(
        ), index=0, label_visibility='collapsed', on_change=cisi.call_off, args=['cisi_plot_val'])
        selected_strata = cisi.STRAT_VALS[strata]

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

        plot2.button('Continue', key='continue_plot', on_click = cisi.call_on, args = ['cisi_plot_val'])
            
        if st.session_state['cisi_plot_val']:
            # Cross-sectional bar plot at baseline
            plot_baseline_scores(df_final, get_varname, cisi_version, selected_strata)


            # plot_interactive_visit_month(
            #     df_final, get_varname, selected_strata)

            # df_sv_temp = create_survival_df(
            #     df_final, 3, 'greater', get_varname, selected_strata)
            # df_sv_temp = df_sv_temp.drop(columns=['event', 'censored_month'])

            # plot_interactive_first_vs_last(df_sv_temp, selected_strata)

            # min_value = cisi.NUMERIC_RANGE[0]
            # max_value = cisi.NUMERIC_RANGE[1]

            # st.markdown(
            #     '#### Kaplan-Meier Curve for Reaching the Threshold Score')
            # thresh1, thresh2, thresh3 = st.columns([1, 0.5, 1])
            # direction = thresh1.radio(label='Direction', horizontal=True, options=[
            #                             'Greater Than or Equal To', 'Less Than or Equal To'])
            # threshold = thresh2.number_input(
            #     min_value=min_value, max_value=max_value, step=1, label='Threshold', value=3)
            # st.write('###')
            # df_sv = create_survival_df(
            #     df_final, threshold, direction, get_varname, selected_strata)
            
            # plot_km_curve(df_sv, selected_strata, threshold, direction) # new interactive method

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
                st.session_state['cisi_data_chunks'].append(df_final)

                yes_col1, yes_col2, yes_col3 = st.columns(
                    3)  # add download button?
                if yes_col1.button("QC another variable", use_container_width=True):
                    # will not work until Modified cisi Field is added
                    st.session_state['cisi_variable'].remove(cisi_version)
                    cisi.call_off('cisi_btn')

                yes_col2.button("Email Data to GP2 Clinical Data Coordinator",
                                use_container_width=True, on_click=cisi.call_on, args = ['cisi_send_email'])

                # necessary because of nested form/button
                if st.session_state['cisi_send_email']:

                    email_form = st.empty()
                    with email_form.container(border=True):
                        st.write("#### :red[Send the following email?]")

                        st.markdown(
                            "__TO:__ Lietsel Jones (Member of GP2's Cohort Integration Working Group)")

                        version = dt.datetime.today().strftime('%Y-%m-%d')
                        st.markdown(
                            f'__ATTACHMENT:__ {version}_{study_name}_cisi_qc.csv')
                        
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
                        st.session_state.cisi_send_email = False
                        send_email(study_name, 'send_data', contact_info={
                            'name': submitter_name, 'email': submitter_email}, data=df_subset, modality='cisi')
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
