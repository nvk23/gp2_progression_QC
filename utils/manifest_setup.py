import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO


class ManifestConfig():
    def __init__(self, df, study_name):
        dfg = st.session_state.master_key.drop_duplicates(
            subset='GP2ID', keep='first')
        dfg = dfg[dfg.study == study_name].copy()

        # Consistent columns across manifests - will need to update if changes
        dfg.rename(columns={'age': 'age_at_baseline'}, inplace=True)        
        self.df = pd.merge(df, dfg[['GP2ID', 'clinical_id', 'GP2_phenotype', 'GP2_PHENO', 'diagnosis', 'age_at_baseline', 'age_of_onset',
                                    'age_at_diagnosis', 'study_arm', 'study_type']], on='clinical_id', how='left', suffixes=('_uploaded', '_manifest'))

    def _get_cols(self):
        # Identify columns from manifest
        manifest_cols = [
            col for col in self.df.columns if col.endswith('_manifest')]

        # Identify equivalent columns from uploaded file
        uploaded_cols = [col.replace('_manifest', '_uploaded')
                         for col in manifest_cols]
        return manifest_cols, uploaded_cols

    def get_ids(self):
        all_ids = len(self.df.clinical_id.unique())
        non_gp2 = self.df[self.df.GP2ID.isnull()]['clinical_id'].unique()
        gp2_df = self.df[self.df.GP2ID.notnull()].copy()

        return all_ids, non_gp2, gp2_df

    def compare_cols(self):
        manifest_cols, uploaded_cols = self._get_cols()

        # Prevents null value replacement needed for comparison from persisting for df
        review_df = self.df.copy()

        manifest_num = [col for col in manifest_cols if pd.api.types.is_numeric_dtype(review_df[col])]
        manifest_cat = [col for col in manifest_cols if not pd.api.types.is_numeric_dtype(review_df[col])]
        
        uploaded_num = [col for col in uploaded_cols if pd.api.types.is_numeric_dtype(review_df[col])]
        uploaded_cat = [col for col in uploaded_cols if not pd.api.types.is_numeric_dtype(review_df[col])]

        # Solves issue where None != None returns True in Pandas 
        review_df.fillna(-9, inplace = True)

        # Compare the corresponding columns and only flag for > 1 difference
        unequal_num = [col for x, y in zip(manifest_num, uploaded_num) if not (abs(
            review_df[x] - review_df[y]) <= 1).all() for col in (x, y)]
        unequal_cat = [col for x, y in zip(manifest_cat, uploaded_cat) if not (
            review_df[x] == review_df[y]).all() for col in (x, y)]

        return unequal_num, unequal_cat
    
    def find_diff(self, unequal_num, unequal_cat):
        unequal_cols = list(unequal_num + unequal_cat)

        # Add ID columns for more complete dataset display
        if 'clinical_id_manifest' not in unequal_cols:
            unequal_cols.insert(0, 'clinical_id')
        if 'GP2ID_manifest' not in unequal_cols:
            unequal_cols.insert(1, 'GP2ID')

        # Only include relevant columns for comparison
        unequal_df = self.df[list(unequal_cols)]

        # Find rows where numerical differences exceed 1
        diff_num = unequal_df.apply(lambda row: any(
            abs(row[unequal_num[i]] - row[unequal_num[i + 1]]) > 1 
            for i in range(0, len(unequal_num), 2)
        ), axis=1) if unequal_num else False
        
        # Find rows where categorical values differ
        diff_cat = unequal_df.apply(lambda row: any(
            row[unequal_cat[i]] != row[unequal_cat[i + 1]] 
            for i in range(0, len(unequal_cat), 2)
        ), axis=1) if unequal_cat else False

        # Display only rows where values differ
        diff_rows = unequal_df[diff_num | diff_cat]
        diff_rows.drop_duplicates(inplace=True)
        return diff_rows

    def adjust_data(self, no_diff=True):
        manifest_cols, uploaded_cols = self._get_cols()

        # Need to fix any discrepancies between manifest and uploaded file before continuing
        if no_diff:
            # Continue with uploaded data columns
            original_cols = [col.replace('_uploaded', '')
                             for col in uploaded_cols]
            rename_cols = dict(zip(uploaded_cols, original_cols))
            self.df.drop(columns=manifest_cols, inplace=True)
            self.df.rename(columns=dict(
                zip(uploaded_cols, original_cols)), inplace=True)
        else:
            if st.session_state.continue_merge == 'Uploaded Values':
                rename_uploaded = [col.split('_uploaded')[0]
                                   for col in uploaded_cols]
                rename_cols = dict(zip(uploaded_cols, rename_uploaded))
                self.df.rename(columns=rename_cols, inplace=True)
                self.df.drop(columns=manifest_cols, inplace=True)
            elif st.session_state.continue_merge == 'Manifest Values':
                rename_manifest = [col.split('_manifest')[0]
                                   for col in manifest_cols]
                rename_cols = dict(zip(manifest_cols, rename_manifest))
                self.df.rename(columns=rename_cols, inplace=True)
                self.df.drop(columns=uploaded_cols, inplace=True)
    
        return self.df
