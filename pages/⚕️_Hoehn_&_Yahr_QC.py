import os
import sys
import subprocess
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from functools import reduce

from utils.writeread import read_file, get_master, get_studycode
from utils.app_setup import HY
from utils.manifest_setup import ManifestConfig
from utils.qc_utils import (prep_merge, compare_manifest, gp2_ids, optional_outcomes, optional_cols, null_vals, numeric_ranges,
                            dup_med_vals, flag_ages, chronological_ages, age_consistency, med_values, visit_month_zero, outcome_qc,
                            cleaned_df, med_val_strata, check_strata, plot_outcomes, review_sample, qc_submit)

# Page set-up
hy = HY('Hoehn & Yahr QC', 'hy')
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
    missing_outcome, available_metrics = prep_merge(hy, uploaded_df)
    
    # Load GP2 Genotyping Data Master Key
    manifest = ManifestConfig(uploaded_df, study_name)
    unequal_num, unequal_cat = manifest.compare_cols()
    unequal_cols = list(unequal_num + unequal_cat)

    # Need to fix any discrepancies between manifest and uploaded file before continuing
    df = compare_manifest(hy, manifest)

    # Check counts and GP2IDs
    df = gp2_ids(manifest, study_name)

    # Check for missing optional outcome columns from template
    optional_outcomes(hy, missing_outcome, df)

    # Check for missing optional outcome columns from template
    optional_cols(hy, df)

    # Add "clinical_state_on_medication" and "dbs_status" column if don't exist already
    for col in ['clinical_state_on_medication', 'dbs_status']:
        if col not in df.columns:
            hy.nullify(df, col)

    # Create column that combines values from age of diagnosis first then age of onset
    df = hy.add_age_outcome(df) 

    # Checking for NaN values in each column and summing them
    null_vals(hy, available_metrics, df)
    
    # Make sure the clnical_id - visit_month combination is unique (warning if not unique)
    dup_med_vals(df)

    # Make sure numeric columns are in proper ranges
    numeric_ranges(hy, df)

    # Warn if sample does not have visit_month = 0
    df = visit_month_zero(hy, df)

    # Warn if sample has age values < 25
    flag_ages(hy, df)

    # Make sure age cols are in chronological order unless Prodromal and non-PD Genetically Enriched
    chronological_ages(df)

    # Make sure age_at_baseline is consistent for the same clinical IDs
    age_consistency(df)

    # Check that clinical variables have correct values if they're in the data
    invalid_med_vals = med_values(hy, df)

    st.success('Your clinical data has passed all required up-front checks!')
    st.markdown('---------')

    # Create method
    qc1, qc2 = st.columns(2)
    qc_col1, qc_col2, qc_col3 = st.columns(3)
    qc_count1, qc_count2, qc_count3 = st.columns(3)

    if 'hy_counter' not in st.session_state:
        qc1.markdown('### HY-Specific Quality Control')
        st.session_state['hy_counter'] = 0
        qc_col1.selectbox(
            "Choose an HY version", st.session_state['hy_variable'], label_visibility='collapsed')
        b1 = qc_col2.button("Continue", on_click=hy.call_on, args = ['hy_btn'])
        get_varname = None
    else:
        qc1.markdown('### HY-Specific Quality Control')
        st.session_state['hy_counter'] += 1
        if len(st.session_state['hy_variable']) >= 1:
            hy_version = qc_col1.selectbox(
                "Choose an HY version", st.session_state['hy_variable'], on_change=hy.call_off, args = ['hy_btn'], label_visibility='collapsed')
            get_varname = hy.OUTCOMES_DICT[hy_version]
            qc_col2.button("Continue", on_click = hy.call_on, args = ['hy_btn'])
        else:
            st.markdown(
                '<p class="medium-font"> You have successfully QCed all clinical variables, thank you!</p>', unsafe_allow_html=True)

            final_df = reduce(lambda x, y: pd.merge(x, y,
                                                    on=['clinical_id',
                                                        'visit_month'],
                                                    how='outer'), st.session_state['hy_data_chunks'])

    if st.session_state['hy_btn']:

        # reorder columns for display
        df_subset = hy.reorder_cols(df)

        outcome_qc(df_subset, get_varname, qc_count1, qc_count2, qc_count3)

        st.markdown('---------')

        st.markdown('### Visualize Cleaned Dataset')

        # Merge on ID and visit_month to keep entries with least null values
        df_final, df_subset = cleaned_df(df_subset)

        # may need to move this higher to the initial QC before HY-specific QC
        st.markdown('Select a stratifying variable to plot:')
        plot1, plot2, plot3 = st.columns(3)
        strata = plot1.selectbox("Select a stratifying variable to plot:", hy.STRAT_VALS.keys(
        ), index=0, label_visibility='collapsed', on_change=hy.call_off, args=['hy_plot_val'])
        selected_strata = hy.STRAT_VALS[strata]

        # Make sure selected stratifying variable is in the dataframe
        med_val_strata(hy, invalid_med_vals, selected_strata)
        check_strata(df_final, selected_strata)

        # Make sure stratifying selections include required values to continue
        if selected_strata in hy.MED_VALS.keys():
            if len(invalid_med_vals) > 0:
                st.error(
                    f'Please make sure {selected_strata} values are {hy.MED_VALS[selected_strata]} to continue.')
                st.markdown(f'_Fix the following {selected_strata} values:_')
                st.dataframe(invalid_med_vals[selected_strata], use_container_width=True)
                st.stop()

        plot2.button('Continue', key='continue_plot', on_click = hy.call_on, args = ['hy_plot_val'])
            
        plot_outcomes(hy, df_final, get_varname, hy_version, selected_strata)
        review_sample(hy, df_final)
        qc_submit(hy, df_final, df_subset, study_name)
