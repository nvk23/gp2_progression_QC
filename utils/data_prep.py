import streamlit as st
import pandas as pd
import numpy as np


def checkNull(df, voi):
    """
    This will check for null values with a given variable of interest and allow you
    to observe null values within context
    """
    nulls = df[df.loc[:, voi].isnull()]
    return (nulls)


def subsetData(df, key, method='less_na'):
    """
    This function takes the obs with less NAs when duplicated.
    If method=='ffill' forward fill the missingness and take the last entry.
    Do the opposite if method=='bfill'
    """
    if method == 'less_na':
        df_new = df.copy()
        df_new['n_missing'] = pd.isna(df_new).sum(axis=1)
        df_new = df_new.sort_values(key+['n_missing']).copy()
        df_new = df_new.drop_duplicates(subset=key, keep='first')
        df_new = df_new.drop(columns=['n_missing']).copy()
    else:
        print('FFILL on process: DO NOT FORGET to sort before using this function!!')
        df.update(df.groupby(key).fillna(method=method))
        df = df.reset_index(drop=True)
        if method == 'ffill':
            df = df.drop_duplicates(subset=key, keep='last').copy()
        if method == 'bfill':
            df = df.drop_duplicates(subset=key, keep='first').copy()

    return (df_new)

def highlight_removed_rows(row):
    # Check if the row of one dataframe exists in the other
    if row.row_kept == 'remove':
        # Highlight removed row in red
        return ['background-color: #ffcccc'] * len(row)
    return [''] * len(row)  # No highlight for other rows

def mark_removed_rows(df_final, df_subset, unequal_dup_rows):
    check_exist = pd.merge(unequal_dup_rows, df_final, how='left', indicator='row_kept')
    check_exist.replace({'both': 'save', 'left_only': 'remove'}, inplace = True)
    styled_duplicates = check_exist.style.apply(highlight_removed_rows, axis=1)

    final_dataset = pd.concat([df_subset, check_exist])
    final_dataset.drop_duplicates(subset = final_dataset.columns[:-1], keep = 'last', inplace = True)
    final_dataset.sort_values(by=['GP2ID'], inplace = True)
    final_dataset.reset_index(drop = True, inplace = True)

    return final_dataset, styled_duplicates

def checkDup(df, keys, drop_dup = True):
    """
    This will check the duplicated observations by keys
    Keys should be provided as a list if they are multiple
    e.g. ['PATNO', 'EVENT_ID']
    If there are duplicated observation, returns these obs.
    """
    t = df[keys]
    t_dup = t[t.duplicated()]
    n_dup = len(t_dup)
    if n_dup == 0:
        return []
    elif n_dup > 0:
        dup_df = df[df.duplicated(keep=False, subset=keys)].sort_values(keys)
        if drop_dup:
          dup_df.drop_duplicates(keep = False, inplace = True)
        return (dup_df)


def data_naproc(df):
    navals = df.isna().sum().to_dict()
    cleancols = []
    for k, v in navals.items():
        if (v / df.shape[0] > 0.6):
            continue
        cleancols.append(k)
    df = df.fillna(999)
    return (df[cleancols], cleancols)


def check_chronological_order(df):
    # Invalid rows are not chronological and have difference > 1
    invalid_rows = (
        (df['age_at_diagnosis'] - df['age_at_baseline'] > 1) &
        (df['age_at_baseline'] - df['age_of_onset'] > 1) &
        ~((df['age_of_onset'] <= df['age_at_baseline']) &
          (df['age_at_baseline'] <= df['age_at_diagnosis']))
    )

    # Return invalid dataframe
    non_chronological =df[invalid_rows]
    non_prodromal = non_chronological[non_chronological.study_type != 'Prodromal']
    PD_cases = non_prodromal[~((non_prodromal.study_type == 'Genetically Enriched') & (non_prodromal.GP2_phenotype != 'PD'))]

    chrono_subset = ['GP2ID', 'clinical_id', 'study_type', 'GP2_phenotype', 'diagnosis', 'age_of_onset', 'age_at_diagnosis', 'age_at_baseline']
    not_chrono = PD_cases[chrono_subset]
    not_chrono.drop_duplicates(inplace = True)
    return not_chrono

def check_consistent(df, col):
    inconsistent = df.groupby('clinical_id')[col].nunique()
    inconsistent = inconsistent[inconsistent > 1].index.tolist()

    diff_vals = df[df.clinical_id.isin(inconsistent)]

    return diff_vals

def detect_multiple_clindups(df):
    st.error(f'There seems to be a problem with this sample manifest')
    groupids = df.groupby(['clinical_id']).size().sort_values(ascending=False)
    groupids_problems = list(groupids[groupids > 3].items())
    for problem_tuple in groupids_problems:
        repid = problem_tuple[0]
        n_reps = problem_tuple[1]
        st.error(
            f'We have detected a total of {n_reps} repetitions for the clinical id code {repid} ')
        show_problemchunk = df[df.clinical_id == repid][[
            'study', 'sample_id', 'clinical_id']]
        st.dataframe(
            show_problemchunk.style.set_properties(
                **{"background-color": "brown", "color": "lawngreen"})
        )
    st.stop()


def sample_type_fix(df, allowed_samples, col):
    allowed_samples_strp = [samptype.strip().replace(" ", "")
                            for samptype in allowed_samples]

    sampletype = df[col].copy()
    sampletype = sampletype.str.replace(" ", "")
    map_strip_orig = dict(zip(sampletype.unique(), df[col].unique()))
    not_allowed_v2 = list(np.setdiff1d(
        sampletype.unique(), allowed_samples_strp))

    if len(not_allowed_v2) > 0:
        st.text(f'WE have found unknown {col}')
        st.text(f'Writing entries with {col} not allowed')
        all_unknwown = []
        for stripval, origval in map_strip_orig.items():
            if stripval in not_allowed_v2:
                all_unknwown.append(origval)
        st.error(f'We could not find the following codes {all_unknwown}')
        st.error(
            f'Printing the list of allowed samples types for reference {allowed_samples}')
        st.stop()
    else:
        st.text('We have found some undesired whitespaces in some sample type values')
        st.text('Processing whitespaces found in certain sample_type entries')
        stype_map = dict(zip(allowed_samples_strp, allowed_samples))
        newsampletype = sampletype.replace(stype_map)
        df[col] = newsampletype
        st.text('sample type count after removing undesired whitespaces')
        st.write(df[col].astype('str').value_counts())


def create_survival_df(df, thres, direction, outcome, strata):
    # Create the event column based on the threshold
    if direction == 'Greater Than or Equal To' or direction == 'greater':
        df['event'] = (df[outcome] >= thres).astype(int)
    elif direction == 'Less Than or Equal To':
        df['event'] = (df[outcome] <= thres).astype(int)

    # Select columns to subset dataframe
    subset_cols = ['GP2ID', 'clinical_id',
                   'GP2_phenotype', 'study_arm', 'study_type']
    if not strata in (subset_cols):
        subset_cols.append(strata)

    df_cs = df[subset_cols].drop_duplicates()

    # Get the first occurrence of the event if it occurred
    df_event = df[df['event'] == 1].sort_values(
        'visit_month').drop_duplicates(subset=['GP2ID'])
    df_event['visit_month_event'] = df_event['visit_month']

    # One survival observation per person approach
    df_sv = df.groupby(['GP2ID'])['visit_month'].agg(
        visit_month_first='min', visit_month_last='max', n_obs='count').reset_index()
    df_sv = df_sv.merge(
        df_event[['GP2ID', 'visit_month_event']], on='GP2ID', how='left')
    df_sv['event'] = pd.notna(df_sv['visit_month_event']).astype(int)
    df_sv['visit_month_censored'] = df_sv['visit_month_last'].where(
        pd.isna(df_sv['visit_month_event']), df_sv['visit_month_event'])
    df_sv['censored_month'] = df_sv['visit_month_censored'] - \
        df_sv['visit_month_first']
    df_sv['follow_up_month'] = df_sv['visit_month_last'] - \
        df_sv['visit_month_first']

    # Get the outcome score at the minimum and maximum visit_month
    df_min_visit_month = df.groupby('GP2ID')[outcome].first(
    ).reset_index().rename(columns={outcome: 'score_first'})
    df_max_visit_month = df.groupby('GP2ID')[outcome].last(
    ).reset_index().rename(columns={outcome: 'score_last'})
    df_sv = df_sv.merge(df_min_visit_month, on='GP2ID', how='left')
    df_sv = df_sv.merge(df_max_visit_month, on='GP2ID', how='left')
    df_sv_return = pd.merge(df_cs,
                            df_sv[['GP2ID', 'n_obs', 'follow_up_month', 'visit_month_first',
                                   'visit_month_last', 'score_first', 'score_last', 'event', 'censored_month']],
                            on='GP2ID', how='left')

    return df_sv_return