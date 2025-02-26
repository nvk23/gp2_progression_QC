import os
import sys
import subprocess
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from functools import reduce

from utils.writeread import read_file, get_master, get_studycode
from utils.app_setup import CISI
from utils.manifest_setup import ManifestConfig
from utils.qc_utils import (prep_merge, compare_manifest, gp2_ids, optional_cols, null_vals, numeric_ranges, flag_ages,
                            chronological_ages, age_consistency, visit_month_zero, outcome_qc, cleaned_df,
                            check_strata, plot_outcomes, review_sample, qc_submit)

# Page set-up
cisi = CISI('CISI-PD QC', 'cisi')
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
    missing_outcome, available_metrics = prep_merge(cisi, uploaded_df)

    # Load GP2 Genotyping Data Master Key
    manifest = ManifestConfig(uploaded_df, study_name)

    # Need to fix any discrepancies between manifest and uploaded file before continuing
    df = compare_manifest(cisi, manifest)
    
    # Check counts and GP2IDs
    gp2_ids(manifest, study_name)
        
    # Check for missing optional columns from template
    optional_cols(cisi, missing_outcome, df)

    # Create column that combines values from age of diagnosis first then age of onset
    df = cisi.add_age_outcome(df) 

    # Checking for NaN values in each column and summing them
    null_vals(cisi, available_metrics, df)

    # Make sure numeric columns are in proper ranges - flag 25 and below for age columns & 0 for visit_month
    numeric_ranges(cisi, df)

    # Warn if sample does not have visit_month = 0
    visit_month_zero(cisi, df)

    # Warn if sample has age values < 25
    flag_ages(cisi, df)

    # Make sure age cols are in chronological order unless Prodromal and non-PD Genetically Enriched
    chronological_ages(df)

    # Make sure age_at_baseline is consistent for the same clinical IDs
    age_consistency(df)

    st.success('Your clinical data has passed all required up-front checks!')
    st.markdown('---------')

    qc1, qc2 = st.columns(2)
    qc_col1, qc_col2, qc_col3 = st.columns(3)
    qc_count1, qc_count2, qc_count3 = st.columns(3)

    if 'cisi_counter' not in st.session_state:
        qc1.markdown('### CISI-PD Quality Control')
        st.session_state['cisi_counter'] = 0
        qc_col1.selectbox(
            "Choose a CISI-PD metric", st.session_state['cisi_variable'], label_visibility='collapsed')
        b1 = qc_col2.button("Continue", on_click=cisi.call_on, args = ['cisi_btn'])
        get_varname = None
    else:
        qc1.markdown('### CISI-PD Quality Control')
        st.session_state['cisi_counter'] += 1
        if len(st.session_state['cisi_variable']) >= 1:
            cisi_version = qc_col1.selectbox(
                "Choose an cisi version", st.session_state['cisi_variable'], on_change=cisi.call_off, args = ['cisi_btn'], label_visibility='collapsed')
            get_varname = cisi.OUTCOMES_DICT[cisi_version]
            qc_col2.button("Continue", on_click = cisi.call_on, args = ['cisi_btn'])
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

        outcome_qc(df_subset, get_varname, qc_count1, qc_count2, qc_count3)

        st.markdown('---------')

        st.markdown('### Visualize Cleaned Dataset')

        # Merge on ID and visit_month to keep entries with least null values
        df_final, df_subset = cleaned_df(df_subset)

        # may need to move this higher to the initial QC before cisi-specific QC
        st.markdown('Select a stratifying variable to plot:')
        plot1, plot2, plot3 = st.columns(3)
        strata = plot1.selectbox("Select a stratifying variable to plot:", cisi.STRAT_VALS.keys(
        ), index=0, label_visibility='collapsed', on_change=cisi.call_off, args=['cisi_plot_val'])
        selected_strata = cisi.STRAT_VALS[strata]

        # Make sure selected stratifying variable is in the dataframe
        check_strata(df_final, selected_strata)

        plot2.button('Continue', key='continue_plot', on_click = cisi.call_on, args = ['cisi_plot_val'])
        plot_outcomes(cisi, df_final, get_varname, cisi_version, selected_strata)
        review_sample(cisi, df_final)
        qc_submit(cisi, df_final, df_subset, cisi_version, study_name)
