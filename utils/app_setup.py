import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO


class AppConfig():
    # Necessary paths and variables for each page
    GOOGLE_APP_CREDS = "secrets/secrets.json"
    REQUIRED_COLS = ['clinical_id', 'visit_month']
    AGE_COLS = ['age_at_baseline', 'age_of_onset', 'age_at_diagnosis']
    NUMERIC_RANGES = {'visit_month': [-1200, 1200], 'age_at_baseline': [0, 125], 'age_of_onset': [0, 120], 'age_at_diagnosis': [0, 120]}
    SESSION_STATES = {'data_chunks': [], 'btn': False, 'plot_val': False, 'send_email': False, 'add_nulls': False, 
                     'add_opts': False, 'continue_merge': ''}

    def __init__(self, page_title, page_name):
        self.page_title = page_title
        self.page_name = page_name

    def get_name(self):
        return self.page_name

    def config_page(self):
        # Config page with logo in browser tab
        st.set_page_config(page_title=self.page_title, page_icon='data/gp2_2-removebg.png', layout="wide")
        if self.page_title != 'Home':
            st.markdown(f'## {self.page_title}')

    def config_data_upload(self, template_link):
        instructions = st.expander("##### :red[Getting Started]", expanded=True)
        with instructions:
            st.markdown(
                f'__①__ Please download [the data dictionary and template]({template_link}). Data dictionary can be found in the 2nd tab.', unsafe_allow_html=True)
            st.markdown(
                '__②__ Upload your clinical data consistent to the template & required fields in the left sidebar. If you recieve AxiosError 400, please re-upload until the issue resolves itself.')
            st.markdown('__③__ Select your GP2 Study Code.')
        st.markdown('---------')

        # Data Uploader
        data_file = st.sidebar.file_uploader("Upload Your clinical data (CSV/XLSX)", type=['xlsx', 'csv'])
        return data_file

    def google_connect(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = AppConfig.GOOGLE_APP_CREDS

    def config_variables(self, ss_dict):
        for var in ss_dict:
            self._config_session_state(f'{self.page_name}_{var}', ss_dict[var])

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

    def add_age_outcome(self, df):
        df['age_outcome'] = df['age_at_diagnosis'].combine_first(df['age_of_onset'])
        df['disease_duration'] = df['age_at_baseline'] - df['age_outcome']
        return df

    def flag_ages(self, df, age):
        flagged_vals = {}
        
        for col in self.AGE_COLS:
            flag_ages = df[df[col] < age][col]
            if not flag_ages.empty:
                flagged_vals[col] = flag_ages.tolist()
        return flagged_vals

    def reorder_cols(self, df):
        cols_order = ['GP2ID'] + [col for col in df.columns if col != 'GP2ID']
        return df[cols_order]

    def missing_required(self, df, extra_cols):
        check_req = AppConfig.REQUIRED_COLS.copy()
        check_req.extend(AppConfig.AGE_COLS)
        missing_extra = np.setdiff1d(extra_cols, df.columns)
        missing_req = np.setdiff1d(check_req, df.columns)
        return missing_extra, missing_req
    
    def check_nulls(self, df, extra_cols):
        # Missing values in required columns
        check_cols = AppConfig.REQUIRED_COLS.copy()
        check_cols.extend(extra_cols)
        check_cols = np.unique(check_cols)

        show_cols = AppConfig.REQUIRED_COLS.copy()
        show_cols.append('GP2ID')
        show_cols.extend(AppConfig.AGE_COLS)
        show_cols = np.unique(show_cols)
        df_nulls = df[show_cols][df[check_cols].isna().any(axis=1)]
        
        return df_nulls
    
    def check_data_types(self, df):
        invalid_types = {}
        for col in self.NUMERIC_RANGES.keys():
            if col in df.columns:
                non_numeric_values = df[~df[col].apply(lambda x: isinstance(x, (int, float, np.number)) | pd.isna(x))][col]
                if not non_numeric_values.empty:
                    invalid_types[col] = non_numeric_values.tolist()
        return invalid_types
    
    def check_visit_months(self, df):
        df['visit_month'] = df['visit_month'].astype(int)
        month_subset = df.groupby('clinical_id')['visit_month'].apply(lambda x: 0 in x.values).reset_index()
        month_subset.columns = ['clinical_id', 'has_zero_month']
        no_zero_month = month_subset[~month_subset.has_zero_month]
        return df, df[df.clinical_id.isin(no_zero_month.clinical_id)]
    
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
    STRAT_VALS = {'GP2 Phenotype': 'GP2_phenotype', 'GP2 PHENO': 'GP2_PHENO', 'Study Arm': 'study_arm',
                'Clinical State on Medication': 'clinical_state_on_medication',
                'Medication for PD': 'medication_for_pd', 'DBS Stimulation': 'dbs_status'}
    OUTCOMES_DICT = {'Original HY Scale': 'hoehn_and_yahr_stage',
                    'Modified HY Scale': 'modified_hoehn_and_yahr_stage'}
    NUMERIC_RANGES = {'ledd_daily': [0, 10000], 'hoehn_and_yahr_stage': [0, 5], 'modified_hoehn_and_yahr_stage': [0, 5]}
    
    def config_HY(self):
        hy_ss = AppConfig.SESSION_STATES.copy()
        hy_ss['variable'] = list(HY.OUTCOMES_DICT.keys())
        super().config_variables(hy_ss)
    
    def missing_optional(self, df):
        missing_optional, missing_req = super().missing_required(df, HY.OPTIONAL_COLS)
        return missing_optional
    
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
    
    def check_med_vals(self, df):
        invalid_med_values = {}
        for col, valid_values in self.MED_VALS.items():
            if col in df.columns:
                invalid_entries = df[~df[col].isin(valid_values) & df[col].notna()][col]
                if not invalid_entries.empty:
                    invalid_med_values[col] = invalid_entries.tolist()
        return invalid_med_values
    
class CISI(AppConfig):
    TEMPLATE_LINK = 'https://docs.google.com/spreadsheets/d/1WD-YPYHUfk5SwS2WDJHq-VG18a5a0JnNIFeainwvBbs/edit?gid=0#gid=0'
    STRAT_VALS = {'GP2 Phenotype': 'GP2_phenotype', 'GP2 PHENO': 'GP2_PHENO', 'Study Arm': 'study_arm'}
    OUTCOMES_DICT = {'CISI-PD Motor Signs': 'code_cisi_pd_motor',
                    'CISI-PD Disability': 'code_cisi_pd_disability',
                    'CISI-PD Motor Complications': 'code_cisi_pd_motor_complications',
                    'CISI-PD Cognitive Status': 'code_cisi_pd_cognitive'}
    NUMERIC_RANGE = [0, 6]
    
    def config_CISI(self):
        cisi_ss = AppConfig.SESSION_STATES.copy()
        cisi_ss['variable'] = list(CISI.OUTCOMES_DICT.keys())
        super().config_variables(cisi_ss)
    
    def check_required(self, df):
        return super().missing_required(df, list(CISI.OUTCOMES_DICT.values()))

    def check_ranges(self, df):
        out_of_range = super().check_ranges(df)

        # If data type errors exist, return them instead of checking ranges
        if 'Invalid Data Types' in out_of_range:
            return out_of_range

        for col in CISI.OUTCOMES_DICT.values():
            if col in df.columns: # not all columns are required
                invalid_values = df[(df[col] < CISI.NUMERIC_RANGE[0]) | (df[col] > CISI.NUMERIC_RANGE[1])][col]
                if not invalid_values.empty:
                    out_of_range[col] = invalid_values.tolist()
        return out_of_range
    
class MDS_UPDRS_PT1(AppConfig):
    TEMPLATE_LINK = 'https://docs.google.com/spreadsheets/d/1sRpbvlmHB0rtBMIuGW6YI7st3eWK-XvXPBe149pY4v8/edit?gid=869233872#gid=869233872'
    OPTIONAL_COLS = ['mds_updrs_part_i_primary_info_source', 'mds_updrs_part_i_pat_quest_primary_info_source']
    STRAT_VALS = {'GP2 Phenotype': 'GP2_phenotype', 'GP2 PHENO': 'GP2_PHENO', 'Study Arm': 'study_arm'}
    OUTCOMES_DICT = {'Cognitive Impairment (UPD2101)': 'code_upd2101_cognitive_impairment',
                    'Hallucinations and Psychosis (UPD2102)': 'code_upd2102_hallucinations_and_psychosis',
                    'Depressed Mood (UPD2103)': 'code_upd2103_depressed_mood',
                    'Anxious Mood (UPD2104)': 'code_upd2104_anxious_mood',
                    'Apathy (UPD2105)': 'code_upd2105_apathy',
                    'Features of Dopamine Dysregulation Syndrome (UPD2106)': 'code_upd2106_dopamine_dysregulation_syndrome_features',
                    'Sleep Problems (UPD2107)': 'code_upd2107_pat_quest_sleep_problems',
                    'Daytime Sleepiness (UPD2108)': 'code_upd2108_pat_quest_daytime_sleepiness',
                    'Pain And Other Sensations (UPD2109)': 'code_upd2109_pat_quest_pain_and_other_sensations',
                    'Urinary Problems (UPD2110)': 'code_upd2110_pat_quest_urinary_problems',
                    'Constipation Problems (UPD2111)': 'code_upd2111_pat_quest_constipation_problems',
                    'Lightheadedness on Standing (UPD2112)': 'code_upd2112_pat_quest_lightheadedness_on_standing',
                    'Fatigue (UPD2113)': 'code_upd2113_pat_quest_fatigue',
                    'MDS-UPDRS Part I Questions 1-6 Summary Sub-Score': 'mds_updrs_part_i_sub_score', 
                    'MDS-UPDRS Part I Patient Questionnaire Questions 7-13 Summary  Sub-Score': 'mds_updrs_part_i_pat_quest_sub_score',
                    'MDS-UPDRS Part I Summary Score': 'mds_updrs_part_i_summary_score'}
    NUMERIC_RANGE = [0, 4]
    SUM_RANGES = {'mds_updrs_part_i_sub_score': [0, 24], 'mds_updrs_part_i_pat_quest_sub_score': [0, 28], 'mds_updrs_part_i_summary_score': [0, 52]}

    def config_MDS_UPDRS_PT1(self):
        pt1_ss = AppConfig.SESSION_STATES.copy()
        var_list = list(MDS_UPDRS_PT1.OUTCOMES_DICT.keys())
        pt1_ss['variable'] = var_list
        super().config_variables(pt1_ss)

    def check_required(self, df):
        return super().missing_required(df, list(MDS_UPDRS_PT1.OUTCOMES_DICT.values()))
    
    def missing_optional(self, df):
        missing_optional, missing_req = super().missing_required(df, MDS_UPDRS_PT1.OPTIONAL_COLS)
        return missing_optional

    def check_ranges(self, df):
        out_of_range = super().check_ranges(df)

        # If data type errors exist, return them instead of checking ranges
        if 'Invalid Data Types' in out_of_range:
            return out_of_range

        # Only focus on individual input cols
        indiv_cols = list(MDS_UPDRS_PT1.OUTCOMES_DICT.values())[:-3]
        for col in indiv_cols:
            if col in df.columns: # not all columns are required
                invalid_values = df[(df[col] < MDS_UPDRS_PT1.NUMERIC_RANGE[0]) | (df[col] > MDS_UPDRS_PT1.NUMERIC_RANGE[1])][col]
                if not invalid_values.empty:
                    out_of_range[col] = invalid_values.tolist()

        # Sum scores only if provided
        for col, (lower, upper) in MDS_UPDRS_PT1.SUM_RANGES.items():
            if col in df.columns: # not all columns are required
                invalid_values = df[(df[col] < lower) | (df[col] > upper)][col]
                if not invalid_values.empty:
                    out_of_range[col] = invalid_values.tolist()

        return out_of_range
    
    def calc_sub_scores(self, df):
        sub1_cols = list(MDS_UPDRS_PT1.OUTCOMES_DICT.values())[:5]
        sub2_cols = list(MDS_UPDRS_PT1.OUTCOMES_DICT.values())[6:]
        sum_cols = list(MDS_UPDRS_PT1.SUM_RANGES.keys())
        
        # Calculate sum but overwrite with null if any cols not provided
        df[sum_cols[0]] = df[sub1_cols].sum(axis=1)
        df.loc[df[sub1_cols].isna().any(axis=1), sum_cols[0]] = np.nan

        df[sum_cols[1]] = df[sub2_cols].sum(axis=1)
        df.loc[df[sub2_cols].isna().any(axis=1), sum_cols[1]] = np.nan
        return df
    
    def calc_sum(self, df):
        sum_cols = list(MDS_UPDRS_PT1.SUM_RANGES.keys())
        df[sum_cols[2]] = df[sum_cols[0]] + df[sum_cols[1]]
        df.loc[df[sum_cols[0]].isna().any(axis=1), sum_cols[2]] = np.nan
        df.loc[df[sum_cols[1]].isna().any(axis=1), sum_cols[2]] = np.nan
        return df