import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from functools import reduce

from utils.app_setup import config_page
from utils.writeread import read_file, get_studycode
from utils.qcutils import checkNull, subsetData, checkDup, create_survival_df
from utils.plotting import plot_km_curve, plot_interactive_visit_month, plot_interactive_first_vs_last


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ''

config_page('Hoehn & Yahr QC')

# Necessary paths - Update Template
# template_link = 'https://docs.google.com/spreadsheets/d/1tTkVcfP8l37uN09vGMNWiPQKBESRQGCrTZLdvR7rVQw/edit?usp=sharing'
template_link = 'https://docs.google.com/spreadsheets/d/1qexD8xKUaORH-kZjUPWl-1duc_PEwbg0pvvlXQ0OPbY/edit?usp=sharing'
data_file = st.sidebar.file_uploader("Upload Your clinical data (CSV/XLSX)", type=['xlsx', 'csv'])
master_path = 'data/master_key.csv'
study_name = get_studycode(master_path) # initializes study_name to None
cols = ['clinical_id', 'visit_month', 'visit_name', 'hoehn_and_yahr_stage']
required_cols =  ['clinical_id', 'visit_month']
outcomes = ['Original HY Scale', 'Modified HY Scale']
outcomes_dict = {'Original HY Scale': 'hoehn_and_yahr_stage'} # add Modified here when figure out template col name

# Necessary session state initializiation/methods
if 'data_chunks' not in st.session_state:
    st.session_state['data_chunks'] = []
if 'btn' not in st.session_state:
    st.session_state['btn'] = False
if 'plot_val' not in st.session_state:
    st.session_state['plot_val'] = False
if 'variable' not in st.session_state:
    st.session_state['variable'] = outcomes

def callback1():
    st.session_state['btn'] = True
def plot_callback1():
    st.session_state['plot_val'] = True
def callback2():
    st.session_state['btn'] = False

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
        st.error(f'None of the clinical IDs are in the GP2. Please check the clinical IDs and your GP2_study_code ({study_name}) are correct')
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
    missing_cols = np.setdiff1d(cols, df.columns)
    if len(missing_cols)>0:
        st.error(f'{missing_cols} are missing. Please use the template sheet')
        st.stop()
    else:
        df_non_miss_check = df[required_cols].copy()

    # Required columns checks
    if df_non_miss_check.isna().sum().sum()>0:
        st.error(f'There are some missing entries in the required columns {required_cols}. Please fill in the missing cells ')
        st.write('First ~20 columns with missing data in any required fields')
        st.write(df_non_miss_check[df_non_miss_check.isna().sum(1)>0].head(20)) # make into dataframe
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
        st.warning(f'Warning: We have detected duplicated visit months with different visit names. Please review data if this was unintended.') # change warning message
        # df.loc[df.duplicated(subset='visit_month', keep=False), 'visit_month'] = df['visit_month'].astype(str) + '_' + df['visit_name']
        if df[['visit_month', 'visit_name']].nunique().values[0] == 1:
            st.error(f' To allow duplicated visit_month, please fill the visit_name')
            stopapp=True

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
            
            ## Move to the end
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
                st.info('Thank you!')
                st.session_state['data_chunks'].append(df_final)
                st.session_state['variable'].remove(hy_version) # will not work until Modified HY Field is added
                st.button("QC another variable", on_click=callback2)
            if qc_yesno == 'NO':
                st.error("Please change any unexpected values in your clinical data and reupload \
                         or get in touch with GP2's Cohort Integration Working Group (cohort@gp2.org) if needed.")
                st.stop()
