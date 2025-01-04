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
    Plots Kaplan-Meier survival curve using lifelines package.
    
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
    plt.figure(figsize=(10, 6))

    ax = plt.subplot(111)

    # Plot the KM curve for each group in strata
    for name, grouped_df in df_sv.groupby(strata):
        kmf.fit(durations=grouped_df['censored_month'], event_observed=grouped_df['event'], label=name)
        # st.dataframe(kmf.survival_function_)
        kmf.plot_survival_function(ax=ax, ci_show=False)
    
    # Manual adding at risk numbers
    event_tables = {}
    for name, grouped_df in df_sv.groupby(strata):
        kmf.fit(durations=grouped_df['censored_month'], event_observed=grouped_df['event'], label=name)
        event_tables[name] = kmf.event_table['at_risk']
    
    # Add at risk numbers to the plot - currently not displaying properly
    # for i, (name, event_table) in enumerate(event_tables.items()):
    #     times = event_table.index
    #     at_risk_counts = event_table.values
    #     for j, (time, count) in enumerate(zip(times, at_risk_counts)):
    #         ax.annotate(f'{count}', xy=(time, 0.1 - i*0.05), xytext=(time, 0.1 - i*0.05),
    #                     textcoords='offset points', fontsize=8, color='black')

    plt.title(f'Kaplan-Meier Survival Curve: Event = {direction} a Score of {threshold}')
    plt.xlabel('Time (Months)')
    plt.ylabel('Survival Probability')
    plt.ylim([0,1])
    plt.legend(title=strata)
    plt.grid(True)
    st.pyplot(plt)

# def plot_km_curve_plotly(df_sv, strata, threshold, direction):
#     """
#     Plots an interactive Kaplan-Meier survival curve using Plotly.
    
#     Parameters:
#         df_sv: pandas.DataFrame
#             The DataFrame containing survival data.
#         strata: str
#             The column name for stratification.
#         threshold: float
#             Threshold value to display in the title.
#         direction: str
#             Direction of the event (e.g., "greater than" or "less than").
#     """
#     kmf = KaplanMeierFitter()
#     fig = go.Figure()

#     # Plot the KM curve for each group in strata
#     for name, grouped_df in df_sv.groupby(strata):
#         kmf.fit(
#             durations=grouped_df['censored_month'], 
#             event_observed=grouped_df['event'], 
#             label=name
#         )
        
#         # Dynamically access the survival probability column
#         # st.dataframe(kmf.survival_function_)
#         survival_column = kmf.survival_function_.columns[0]  # Get the column dynamically
        
#         # Add survival function to the plot
#         fig.add_trace(go.Scatter(
#             x=kmf.survival_function_.index,
#             y=kmf.survival_function_[survival_column],
#             mode='lines',
#             name=f"{name}",
#             line=dict(width=2)
#         ))

#     # Update layout for the plot
#     fig.update_layout(
#         title={
#             'text': f'Kaplan-Meier Survival Curve: Event = {direction} a Score of {threshold}',
#             'x': 0.5,
#             'xanchor': 'center',
#             'yanchor': 'top',
#             'font': {
#                 'size': 20
#             }
#         },
#         xaxis_title='Time (Months)',
#         yaxis_title='Survival Probability',
#         xaxis=dict(showgrid=True),
#         yaxis=dict(range=[0, 1], showgrid=True),
#         showlegend = True,
#         legend_title = strata,
#         legend=dict(font=dict(size=14)),
#         template='plotly_white',
#         height=700 
#     )

#     # Display the plot
#     st.plotly_chart(fig, use_container_width=True)

def add_jitter(arr, scale=0.1):
    return arr + np.random.normal(scale=scale, size=len(arr))

def plot_interactive_first_vs_last(df_sf, df, strata):
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
