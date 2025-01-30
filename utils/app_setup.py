import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO

# from google.cloud import storage

class AppConfig():
    # Necessary paths and variables for each page
    GOOGLE_APP_CREDS = "secrets/secrets.json"
    REQUIRED_COLS = ['clinical_id', 'visit_month']
    NUMERIC_RANGES = {'visit_month': [-1200, 1200]}

    def __init__(self, page_title):
        self.page_title = page_title

    def config_page(self):
        # Config page with logo in browser tab
        st.set_page_config(page_title=self.page_title, page_icon='data/gp2_2-removebg.png', layout="wide")

    def google_connect(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = AppConfig.GOOGLE_APP_CREDS

    def config_variables(self, ss_dict):
        for var in ss_dict:
            self._config_session_state(var, ss_dict[var])

    def _config_session_state(self, variable, value):
        if variable not in st.session_state:
            st.session_state[variable] = value

    def call_on(self, var_name):
        st.session_state[var_name] = True

    def call_off(self, var_name):
        st.session_state[var_name] = False

    def nullify(self, df, cols):
        for col in cols:
            df[col] = None

    def missing_required(self, df, outcome_cols):
        missing_outcome = np.setdiff1d(outcome_cols, df.columns)
        missing_req = np.setdiff1d(AppConfig.REQUIRED_COLS, df.columns)
        return missing_outcome, missing_req
    
    def check_data_types(self, df):
        invalid_types = {}
        for col in self.NUMERIC_RANGES.keys():
            if col in df.columns:
                non_numeric_values = df[~df[col].apply(lambda x: isinstance(x, (int, float, np.number)) | pd.isna(x))][col]
                if not non_numeric_values.empty:
                    invalid_types[col] = non_numeric_values.tolist()
        return invalid_types
    
    def check_ranges(self, df):
        out_of_range = {}
        for col, (lower, upper) in self.NUMERIC_RANGES.items():
            if col in df.columns:
                invalid_values = df[(df[col] < lower) | (df[col] > upper)][col]
                if not invalid_values.empty:
                    out_of_range[col] = invalid_values.tolist()
        return out_of_range


class HY(AppConfig):
    TEMPLATE_LINK = 'https://docs.google.com/spreadsheets/d/1qexD8xKUaORH-kZjUPWl-1duc_PEwbg0pvvlXQ0OPbY/edit?usp=sharing'
    OPTIONAL_COLS = ['clinical_state_on_medication', 'medication_for_pd', 'dbs_status',
                    'ledd_daily', 'comments']
    MED_VALS = {'clinical_state_on_medication': ['ON', 'OFF', 'Unknown'],
                'medication_for_pd': ['Yes', 'No', 'Unknown'],
                'dbs_status': ['Yes', 'No', 'Unknown', 'Not applicable']}
    OUTCOMES_DICT = {'Original HY Scale': 'hoehn_and_yahr_stage',
                    'Modified HY Scale': 'modified_hoehn_and_yahr_stage'}
    SESSION_STATES = {'data_chunks': [], 'btn': False, 'plot_val': False, 'send_email': False, 'add_nulls': False,
                      'variable': list(OUTCOMES_DICT.keys()), 'continue_merge': ''}
    AGE_COLS = ['age_at_baseline', 'age_of_onset', 'age_at_diagnosis'] # see if you can combine with required_cols
    NUMERIC_RANGES = {'ledd_daily': [0, 10000], 'age_at_baseline': [25, 125], 'age_of_onset': [25, 120], 'age_at_diagnosis': [25, 120],
                      'hoehn_and_yahr_stage': [0, 5], 'modified_hoehn_and_yahr_stage': [0, 5]}
    
    def config_HY(self):
        super().config_variables(HY.SESSION_STATES)
    
    def missing_HY(self, df):
        hy_all  = list(HY.OUTCOMES_DICT.values())
        hy_all.extend(list(AppConfig.REQUIRED_COLS))
        hy_all.extend(HY.AGE_COLS)
        hy_all.extend(HY.OPTIONAL_COLS)
        no_req = np.setdiff1d(df.columns, hy_all)
        return no_req, hy_all
    
    def check_required(self, df):
        return super().missing_required(df, list(HY.OUTCOMES_DICT.values()))

    def check_ranges(self, df):
        out_of_range = super().check_ranges(df)

        # If data type errors exist, return them instead of checking ranges
        if 'Invalid Data Types' in out_of_range:
            return out_of_range

        for col, (lower, upper) in self.NUMERIC_RANGES.items():
            if col in df.columns: # not all columns are required
                invalid_values = df[(df[col] < lower) | (df[col] > upper)][col]
                if not invalid_values.empty:
                    out_of_range[col] = invalid_values.tolist()
        return out_of_range
    
    def check_med_values(self, df):
        invalid_med_values = {}
        for col, valid_values in self.MED_VALS.items():
            if col in df.columns:
                invalid_entries = df[~df[col].isin(valid_values) & df[col].notna()][col]
                if not invalid_entries.empty:
                    invalid_med_values[col] = invalid_entries.tolist()
        return invalid_med_values