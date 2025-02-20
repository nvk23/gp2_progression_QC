import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def plot_km_curve(df_sv, strata, threshold, direction):
    """
    Plots an interactive Kaplan-Meier survival curve using Plotly.
    
    Parameters:
        df_sv: pandas.DataFrame
            The DataFrame containing survival data.
        strata: str
            The column name for stratification.
        threshold: float
            Threshold value to display in the title.
        direction: str
            Direction of the event (e.g., "greater than" or "less than").
    """
    kmf = KaplanMeierFitter()
    fig = go.Figure()

    # Plot the KM curve for each group in strata
    for name, grouped_df in df_sv.groupby(strata):
        kmf.fit(
            durations=grouped_df['censored_month'], 
            event_observed=grouped_df['event'], 
            label=name
        )
        
        # Dynamically access the survival probability column
        kmf_vals = kmf.survival_function_
        survival_column = kmf_vals.columns[0]  # Get the column value dynamically
        kmf_vals.reset_index(inplace = True)

        # Repeat timeline values to match style of lifelines built-in function for KM plots
        kmf_dup = pd.DataFrame({
            'timeline': kmf_vals['timeline'].repeat(2).reset_index(drop=True),
            survival_column: kmf_vals[survival_column].repeat(2).reset_index(drop=True)
        })

        # Shift the timeline index and drop null values
        kmf_dup.loc[1::2, 'timeline'] = kmf_dup.loc[1::2, 'timeline'].shift(-1)
        kmf_dup.dropna(inplace = True)

        # Reset index with timeline as index
        kmf_dup['timeline'] = kmf_dup['timeline'].astype(int)
        kmf_dup = kmf_dup.set_index('timeline')
        
        # Add survival function to the plot with modified dataframe
        fig.add_trace(go.Scatter(
            x=kmf_dup.index,
            y=kmf_dup[survival_column],
            mode='lines',
            name=f"{name}",
            line=dict(width=4)
        ))

    # Update layout for the plot
    fig.update_layout(
        title={
            'text': f'Kaplan-Meier Survival Curve: Event = {direction} a Score of {threshold}',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 20
            }
        },
        xaxis_title='Time (Months)',
        yaxis_title='Survival Probability',
        xaxis=dict(showgrid=True),
        yaxis=dict(range=[0, 1.01], showgrid=True),
        showlegend = True,
        legend_title = strata,
        legend=dict(font=dict(size=14)),
        template='plotly_white',
        height=700 
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

def add_jitter(arr, scale=0.1):
    return arr + np.random.normal(scale=scale, size=len(arr))

def plot_interactive_first_vs_last(df_sf, strata):
    st.markdown('#### Comparison of First vs Last Score per Sample')

    # Add jitter to scores
    df_sf['First Score (Jittered)'] = add_jitter(df_sf['score_first'])
    df_sf['Last Score (Jittered)'] = add_jitter(df_sf['score_last'])

    # Create the plot
    fig = px.scatter(
        df_sf, 
        x='First Score (Jittered)', 
        y='Last Score (Jittered)', 
        color=strata,
        hover_data=['GP2ID'],
        # title='Score First vs Score Last with Jitter'
    )

    # Render the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def plot_interactive_visit_month(df, outcome, strata):
    st.markdown('#### Visit Month vs. Hoehn and Yahr Stage')

    # Add jitter
    df[f'{outcome}_jittered'] = add_jitter(df[outcome])
    df['visit_month_jittered'] = add_jitter(df['visit_month'], 1)

    # Create the Plotly figure
    fig = px.scatter(
        df,
        x='visit_month_jittered',
        y=f'{outcome}_jittered',
        color=strata,
        hover_data=['GP2ID',  'GP2_phenotype', 'study_arm'],
        # title='Visit Month vs. Hoehn and Yahr Stage (Jittered)',
        labels={'visit_month_jittered': 'Visit Month (Jittered)', 'hoehn_and_yahr_stage_jittered': 'Hoehn and Yahr Stage (Jittered)'}
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_baseline_scores(df, metric, metric_name, strata):
    fig = px.histogram(df, x=metric, color=strata, title=f"{metric_name} Scores per Sample")
    fig.update_xaxes(title_text=metric_name, tickmode='array', tickvals=[0, 1, 2, 3, 4, 5, 6])

    st.plotly_chart(fig, use_container_width=True)

def plot_duration_values(df, metric, metric_name):
    # Define bin edges
    bin_edges = np.arange(0, df["disease_duration"].max() + 3, 3)
    bin_labels = [f"{int(b)}-{int(b+3)}" for b in bin_edges[:-1]]

    # Assign bins to a new column
    df["duration_bins"] = pd.cut(df["disease_duration"], bins=bin_edges, labels=bin_labels, include_lowest=True)

    # Count occurrences per bin
    bin_counts = df["duration_bins"].value_counts()

    # Separate bins with only one value
    single_value_bins = bin_counts[bin_counts == 1].index
    multi_value_bins = bin_counts[bin_counts > 1].index

    df_single = df[df["duration_bins"].isin(single_value_bins)]
    df_multi = df[df["duration_bins"].isin(multi_value_bins)]

    # Creates violin plots - turned off outlier points
    fig = px.violin(df_multi, x="duration_bins", y=metric, box=True, points=False,
                    title=f"{metric_name} Distribution Across Disease Duration Bins",
                    category_orders={"duration_bins": bin_labels})

    # Add scatter plot for bins with only one value
    fig.add_trace(px.scatter(df_single, x="duration_bins", y=metric).data[0])

    # Update axes labels
    fig.update_xaxes(title_text="Disease Duration (Years)")
    fig.update_yaxes(title_text=metric_name)

    st.plotly_chart(fig, use_container_width=True)
