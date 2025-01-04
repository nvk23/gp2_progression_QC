import streamlit as st
import pandas as pd
import numpy as np

def checkNull(df, voi):
    """
    This will check for null values with a given variable of interest and allow you
    to observe null values within context
    """
    nulls = df[df.loc[:,voi].isnull()]
    # n_null = len(nulls)
  
    # if n_null==0:
    #     st.write(f'{len(df)} entries: No null values')
    # if n_null>0:
    #     st.write(f'{len(df)} entries: {n_null} null entries were found')
    #     st.dataframe(nulls)

    return(nulls)

def subsetData(df, key, method='less_na'):
  """
  This function takes the obs with less NAs when duplicated.
  If method=='ffill' forward fill the missingness and take the last entry.
  Do the opposite if method=='bfill'
  """
  if method=='less_na':
    df['n_missing'] = pd.isna(df).sum(axis=1)
    df = df.sort_values(key+['n_missing']).copy()
    df = df.drop_duplicates(subset=key, keep='first')
    df = df.drop(columns=['n_missing']).copy()

  else:
      print('FFILL on process: DO NOT FORGET to sort before using this function!!')
      df.update(df.groupby(key).fillna(method=method))
      df=df.reset_index(drop=True)
      if method=='ffill':
        df = df.drop_duplicates(subset=key, keep='last').copy()
      if method=='bfill':
        df = df.drop_duplicates(subset=key, keep='first').copy()

  return(df)


def checkDup(df, keys):
  """
  This will check the duplicated observations by keys
  Keys should be provided as a list if they are multiple
  e.g. ['PATNO', 'EVENT_ID']
  If there are duplicated observation, returns these obs.
  """
  t = df[keys]
  t_dup = t[t.duplicated()]
  n_dup = len(t_dup)
  if n_dup==0:
    return []
  elif n_dup>0:
    d_dup2 = df[df.duplicated(keep=False, subset=keys)].sort_values(keys)
    return(d_dup2)


def data_naproc(df):
  navals = df.isna().sum().to_dict()
  cleancols = []
  for k, v in navals.items():
      if (v / df.shape[0] > 0.6):
          continue
      cleancols.append(k)
  df = df.fillna(999)
  return(df[cleancols], cleancols)

def evaluate_ages(row):
    ages = row[['age_of_onset', 'age_at_diagnosis', 'age_at_baseline']].dropna().tolist()
    return ages == sorted(ages)

def check_chronological_order(df):
    result = df.apply(evaluate_ages, axis=1)
    non_passing_entries = df[~result]
    return non_passing_entries

def detect_multiple_clindups(df):
  st.error(f'There seems to be a problem with this sample manifest')
  groupids = df.groupby(['clinical_id']).size().sort_values(ascending=False)
  groupids_problems = list(groupids[groupids > 3].items())
  for problem_tuple in groupids_problems:
      repid = problem_tuple[0]
      n_reps = problem_tuple[1]
      st.error(f'We have detected a total of {n_reps} repetitions for the clinical id code {repid} ')
      show_problemchunk = df[df.clinical_id == repid][['study','sample_id','clinical_id']]
      st.dataframe(
              show_problemchunk.style.set_properties(**{"background-color": "brown", "color": "lawngreen"})
              )
  st.stop()

def sample_type_fix(df, allowed_samples, col):
  allowed_samples_strp = [samptype.strip().replace(" ", "") for samptype in allowed_samples]
  
  sampletype = df[col].copy()
  sampletype = sampletype.str.replace(" ", "")
  map_strip_orig = dict(zip(sampletype.unique(), df[col].unique()))
  not_allowed_v2 = list(np.setdiff1d(sampletype.unique(), allowed_samples_strp))
  
  if len(not_allowed_v2)>0:
    st.text(f'WE have found unknown {col}')
    st.text(f'Writing entries with {col} not allowed')
    all_unknwown = []
    for stripval, origval in map_strip_orig.items():
      if stripval in not_allowed_v2:
        all_unknwown.append(origval)
    st.error(f'We could not find the following codes {all_unknwown}') 
    st.error(f'Printing the list of allowed samples types for reference {allowed_samples}')
    st.stop()
  else:
    st.text('We have found some undesired whitespaces in some sample type values')
    st.text('Processing whitespaces found in certain sample_type entries')
    stype_map = dict(zip(allowed_samples_strp, allowed_samples))
    newsampletype = sampletype.replace(stype_map)
    df[col] = newsampletype
    st.text('sample type count after removing undesired whitespaces')
    st.write(df[col].astype('str').value_counts())

  

def create_survival_df(df, thres, direction, outcome):
    # Create the event column based on the threshold
    if direction == 'Greater Than or Equal To' or direction == 'greater':
        df['event'] = (df[outcome] >= thres).astype(int)
    elif direction == 'Less Than or Equal To':
        df['event'] = (df[outcome] <= thres).astype(int)

    df_cs = df[['GP2ID', 'clinical_id', 'GP2_phenotype', 'study_arm', 'study_type']].drop_duplicates()

    # Get the first occurrence of the event if it occurred
    df_event = df[df['event'] == 1].sort_values('visit_month').drop_duplicates(subset=['GP2ID'])
    df_event['visit_month_event'] = df_event['visit_month']

    # One survival observation per person approach
    df_sv = df.groupby(['GP2ID'])['visit_month'].agg(visit_month_first='min', visit_month_last='max', n_obs='count').reset_index()
    df_sv = df_sv.merge(df_event[['GP2ID', 'visit_month_event']], on='GP2ID', how='left')
    df_sv['event'] = pd.notna(df_sv['visit_month_event']).astype(int)
    df_sv['visit_month_censored'] = df_sv['visit_month_last'].where(pd.isna(df_sv['visit_month_event']), df_sv['visit_month_event'])
    df_sv['censored_month'] = df_sv['visit_month_censored'] - df_sv['visit_month_first']
    df_sv['follow_up_month'] = df_sv['visit_month_last'] - df_sv['visit_month_first']
    
    # Get the outcome score at the minimum and maximum visit_month
    df_min_visit_month = df.groupby('GP2ID')[outcome].first().reset_index().rename(columns={outcome: 'score_first'})
    df_max_visit_month = df.groupby('GP2ID')[outcome].last().reset_index().rename(columns={outcome: 'score_last'})
    df_sv = df_sv.merge(df_min_visit_month, on='GP2ID', how='left')
    df_sv = df_sv.merge(df_max_visit_month, on='GP2ID', how='left')
    df_sv_return = pd.merge(df_cs, 
                            df_sv[['GP2ID', 'n_obs', 'follow_up_month', 'visit_month_first',  'visit_month_last', 'score_first', 'score_last', 'event', 'censored_month']],
                            on='GP2ID', how='left')

    return df_sv_return