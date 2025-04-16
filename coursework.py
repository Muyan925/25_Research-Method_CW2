import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os
# ------------------- Data Loading -------------------

file_path = "Results_21Mar2022.csv"
df = pd.read_csv(file_path)
print(df.columns)
# Basic data checks
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
print("Data Preview:")
print(df.head())
print("Data Types:")
print(df.dtypes)
print("Missing Values (sorted by percentage):")
missing_values = df.isnull().sum() / len(df) * 100
print(missing_values[missing_values > 0].sort_values(ascending=False))
print("Unique Values Per Column (first 10 columns):")
print(df.nunique().head(10))


# Note: No missing values detected in dataset (checked via missing_values report)
indicators = ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut']
# ------------------- Helper Functions -------------------

def detect_outliers_iqr(df, column):
    """Detect outliers in a column using the IQR method.
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to analyze.
    Returns:
        pd.DataFrame: Rows identified as outliers.
    """"""Detect outliers using IQR method with 1.5x threshold (Tukey's rule)."""

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def get_all_outliers(df, indicators):
    outliers = pd.concat([detect_outliers_iqr(df, col) for col in indicators]).drop_duplicates()
    # Merge diet_metadata to add diet_category column only
    outliers = pd.merge(outliers, diet_metadata[['diet_group', 'diet_category']], 
                        on=['diet_group', 'diet_category'], how='left')
    return outliers



def categorize_diet(diet):
    """Map diet groups to categories.
    Args:
        diet (str): Diet group name.
    Returns:
        str: Diet category.
    """
    mapping = {
        'vegan': 'plant_based',
        'veggie': 'vegetarian',
        'fish': 'pescatarian',
        'meat100': 'high_meat',
        'meat50': 'medium_meat',
        'meat': 'low_meat'
    }
    return mapping.get(diet, 'low_meat')  # Default to low_meat for unrecognized diets



# ------------------- PCA-Based Weight Calculation -------------------

def calculate_pca_weights(data, indicators):
    """Calculate PCA-based weights for indicators.
    Args:
        data (pd.DataFrame): Input data.
        indicators (list): Columns for PCA.
    Returns:
        dict: Indicator weights.
        float: Explained variance ratio of first component.
    """
    X = data[indicators]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run PCA with all components to check variance
    pca = PCA()
    pca.fit(X_scaled)
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Use one component for weights (assumes first component captures most variance)
    pca = PCA(n_components=1)
    pca.fit(X_scaled)
    loadings = pca.components_[0]
    weights = loadings**2 / sum(loadings**2)
    return dict(zip(indicators, weights)), pca.explained_variance_ratio_[0]

weights, explained_variance = calculate_pca_weights(df, indicators)
print("PCA Weights:")
for indicator, weight in weights.items():
    print(f"{indicator}: {weight:.3f}")
print(f"Explained variance ratio by PC1: {explained_variance:.3f}")

# ------------------- Data Processing -------------------

# Normalize indicators (min-max scaling) once for reuse
for col in indicators:
    denom = df[col].max() - df[col].min()
    df[f'{col}_norm'] = (df[col] - df[col].min()) / denom if denom != 0 else 0

# Calculate total impact score
df['total_impact_score'] = sum(df[f'{col}_norm'] * weights[col] for col in indicators)
print("Total Impact Score Preview:")
print(df[['diet_group', 'sex', 'age_group', 'total_impact_score']].head())

# Add diet category
df['diet_category'] = df['diet_group'].apply(categorize_diet)



# Define diet metadata (aligned with dataset)
diet_metadata = pd.DataFrame({
    'diet_group': ['vegan', 'veggie', 'fish', 'meat50', 'meat', 'meat100'],
    'diet_category': ['plant_based', 'vegetarian', 'pescatarian', 'medium_meat', 'low_meat', 'high_meat']
})

# Validate diet groups
if not set(df['diet_group'].unique()).issubset(diet_metadata['diet_group']):
    print("Warning: Some diet groups in data not covered by metadata.")


# ------------------- Aggregations and Trends -------------------

# Combined aggregation for dashboard and trends
agg_df = df.groupby(['diet_category', 'age_group', 'sex']).agg({
    'total_impact_score': 'mean',
    'n_participants': 'sum',
    'mean_ghgs': 'mean',
    'mean_land': 'mean',
    'mean_watscar': 'mean',
    'mean_eut': 'mean'
}).reset_index()



# Normalize indicators for visualization
for col in indicators:
    denom = agg_df[col].max() - agg_df[col].min()
    agg_df[f'{col}_norm'] = (agg_df[col] - agg_df[col].min()) / denom if denom != 0 else 0




# Summary statistics
# Compute weighted average for diet categories using n_participants as weights
impact_summary = (agg_df.groupby('diet_category')
                 .apply(lambda x: np.average(x['total_impact_score'], weights=x['n_participants']))
                 .reset_index(name='total_impact_score'))
best_diet = impact_summary.loc[impact_summary['total_impact_score'].idxmin()]
worst_diet = impact_summary.loc[impact_summary['total_impact_score'].idxmax()]
disparity_ratio = (worst_diet['total_impact_score'] / best_diet['total_impact_score']
                   if best_diet['total_impact_score'] != 0 else np.nan)

# Compute weighted average for age groups using n_participants as weights
age_summary = (agg_df.groupby('age_group')
               .apply(lambda x: np.average(x['total_impact_score'], weights=x['n_participants']))
               .reset_index(name='total_impact_score'))
best_age = age_summary.loc[age_summary['total_impact_score'].idxmin()]
# ------------------- Save Data -------------------
df.to_csv("processed_data.csv", index=False)
agg_df.to_csv("aggregated_data.csv", index=False)
print("processed_data.csv columns:", df.columns)
print("aggregated_data.csv columns:", agg_df.columns)


# ------------------- Dash Application -------------------

app = dash.Dash(__name__)
# Define colors
custom_colors = [[0, '#5CB85C'], [0.5, '#F5E050'], [1, '#D9534F']]

# Define layout
app.layout = html.Div([
    html.H1("Diet, Demographics & Environmental Impact Analyzer",
            style={'textAlign': 'center', 'fontFamily': 'Arial', 'color': '#2c3e50', 'fontSize': '28px'}),
    dcc.Checklist(
        id='show-outliers',
        options=[{'label': 'Display outliers', 'value': 'show'}],
        value=[],
        style={'margin': '10px'}
    ),
    html.Div([
        html.Label("Select Diet Category:", style={'fontFamily': 'Arial', 'marginRight': '10px', 'fontSize': '16px'}),
        dcc.Checklist(
            id='diet-checklist',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': cat.capitalize(), 'value': cat} for cat in agg_df['diet_category'].unique()],
            value=['All'],
            inline=True,
            style={'width': '30%', 'display': 'inline-block', 'fontSize': '14px', 'marginRight': '20px'}
        ),
        html.Label("Select Age Group:", style={'fontFamily': 'Arial', 'marginRight': '10px', 'fontSize': '16px'}),
        dcc.Checklist(
            id='age-checklist',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': age, 'value': age} for age in agg_df['age_group'].unique()],
            value=['All'],
            inline=True,
            style={'width': '30%', 'display': 'inline-block', 'fontSize': '14px', 'marginRight': '20px'}
        ),
        html.Label("Select Gender:", style={'fontFamily': 'Arial', 'marginRight': '10px', 'fontSize': '16px'}),
        dcc.Dropdown(
            id='sex-dropdown',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': sex.capitalize(), 'value': sex} for sex in agg_df['sex'].unique()],
            value='All',
            style={'width': '30%', 'display': 'inline-block', 'fontSize': '14px'}
        ),
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderBottom': '1px solid #dee2e6', 'display': 'flex'}),
    html.Div([
        html.Label("Select View:", style={'fontFamily': 'Arial', 'marginRight': '10px', 'fontSize': '16px'}),
        dcc.RadioItems(
            id='view-toggle',
            options=[
                {'label': 'Treemap', 'value': 'treemap'},
                {'label': 'Parallel Coordinates', 'value': 'parcoords'},
                {'label': 'Combined View', 'value': 'combined'}
            ],
            value='treemap',
            labelStyle={'display': 'inline-block', 'marginRight': '20px', 'fontSize': '14px'},
            style={'padding': '10px'}
        ),
    ], style={'padding': '10px', 'backgroundColor': '#f8f9fa'}),
    html.Div(id='graph-container'),
    html.Div([
        html.H3("Key Statistics", style={'fontFamily': 'Arial', 'color': '#2c3e50', 'fontSize': '20px'}),
        html.P(f"🥦 Best Diet: {best_diet['diet_category']} (Score={best_diet['total_impact_score']:.3f})",
               style={'color': '#28a745', 'fontSize': '16px'}),
        html.P(f"🥩 Worst Diet: {worst_diet['diet_category']} (Score={worst_diet['total_impact_score']:.3f})",
               style={'color': '#dc3545', 'fontSize': '16px'}),
        html.P(f"Disparity: {disparity_ratio:.2f}x", style={'color': '#007bff', 'fontSize': '16px'}),
        html.P(f"👶 Best Age Group: {best_age['age_group']} (Score={best_age['total_impact_score']:.3f})",
               style={'color': '#17a2b8', 'fontSize': '16px'}),
    ], style={'padding': '20px', 'backgroundColor': '#e9ecef', 'borderRadius': '5px', 'margin': '10px'}),
    html.Div(id='insight-text', style={'padding': '20px', 'fontSize': '16px', 'fontFamily': 'Arial',
                                      'color': '#6c757d', 'backgroundColor': '#f1f3f5', 'borderRadius': '5px'})
])
# Define callback function
@app.callback(
    [Output('graph-container', 'children'),
     Output('insight-text', 'children')],
    [Input('diet-checklist', 'value'),  
     Input('age-checklist', 'value'),  
     Input('sex-dropdown', 'value'),
     Input('view-toggle', 'value'),
     Input('show-outliers', 'value')]
)
def update_dashboard(selected_diets, selected_age, selected_sex, view_type, show_outliers):
    df_filtered = agg_df.copy()  # Use a copy to avoid modifying the original data
    title = "Hierarchical Distribution of Environmental Impact Across Diet Groups"

    # Process diet-checklist's All logic
    all_diets = list(agg_df['diet_category'].unique())
    if 'All' in selected_diets:
        selected_diets = all_diets  # Select All to include all diet categories
    elif not selected_diets:
        selected_diets = []

    # Process age-checklist's All logic
    all_ages = list(agg_df['age_group'].unique())
    if 'All' in selected_age:
        selected_age = all_ages  # Select All to include all age groups
    elif not selected_age:
        selected_age = []

    # Define filter conditions
    filters = {
        'diet_category': selected_diets,
        'age_group': selected_age,
        'sex': selected_sex
    }

    # Apply filtering to df_filtered
    for column, value in filters.items():
        if value != 'All' and value:
            if isinstance(value, list):
                df_filtered = df_filtered[df_filtered[column].isin(value)]
            else:
                df_filtered = df_filtered[df_filtered[column] == value]

    # Handle outliers
    if 'show' in show_outliers:
        # Create a filtered dataset for outlier detection
        df_for_outliers = df.copy()
        for column, value in filters.items():
            if value != 'All' and value:
                if isinstance(value, list):
                    df_for_outliers = df_for_outliers[df_for_outliers[column].isin(value)]
                else:
                    df_for_outliers = df_for_outliers[df_for_outliers[column] == value]

        # Compute outliers on the filtered dataset
        all_outliers = get_all_outliers(df_for_outliers, indicators)
        # Ensure no NaNs in required columns
        all_outliers = all_outliers.dropna(subset=['diet_category', 'age_group', 'sex'])
        # Aggregate outliers to match df_filtered structure
        all_outliers_agg = all_outliers.groupby(['diet_category', 'age_group', 'sex']).agg({
            'total_impact_score': 'mean',
            'n_participants': 'sum',
            'mean_ghgs': 'mean',
            'mean_land': 'mean',
            'mean_watscar': 'mean',
            'mean_eut': 'mean'
        }).reset_index()
        # Normalize indicators for outliers
        for col in indicators:
            denom = all_outliers_agg[col].max() - all_outliers_agg[col].min()
            all_outliers_agg[f'{col}_norm'] = (all_outliers_agg[col] - all_outliers_agg[col].min()) / denom if denom != 0 else 0
        # Concatenate filtered data and outliers
        df_filtered = pd.concat([df_filtered, all_outliers_agg], ignore_index=True).drop_duplicates()

    # Ensure df_filtered has no NaNs
    df_filtered = df_filtered.dropna(subset=['diet_category', 'age_group', 'sex'])

    # Debugging: Check filtered data
    print("df_filtered NaN 检查：")
    print(df_filtered[['diet_category', 'age_group', 'sex']].isnull().sum())
    print("df_filtered total_impact_score 统计（显示异常值={}）：".format('show' in show_outliers))
    print(df_filtered['total_impact_score'].describe())

    # Create treemap
    fig_treemap = px.treemap(
        df_filtered,
        path=['diet_category', 'age_group', 'sex'],
        values='n_participants',
        color='total_impact_score',
        color_continuous_scale=custom_colors,
        title=title
    )
    fig_treemap.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        coloraxis_colorbar=dict(title="Total Impact Rating")
    )

    # Create parallel coordinates plot
    fig_parcoords = go.Figure(data=go.Parcoords(
        line=dict(
            color=df_filtered['total_impact_score'],
            colorscale=custom_colors,
            showscale=False
        ),
        dimensions=[
            dict(label="Greenhouse Gas Emissions", values=df_filtered['mean_ghgs_norm'], range=[0, 1]),
            dict(label="Land Use", values=df_filtered['mean_land_norm'], range=[0, 1]),
            dict(label="Water Scarcity", values=df_filtered['mean_watscar_norm'], range=[0, 1]),
            dict(label="Eutrophication", values=df_filtered['mean_eut_norm'], range=[0, 1])
        ]
    ))
    fig_parcoords.update_layout(
        margin=dict(t=90, l=25, r=25, b=25),
        title="Parallel Coordinates of Environmental Impacts",
        title_y=0.94
    )

    # Dynamic insights
    insight_map = {
        'plant_based': "Vegans typically have the lowest environmental impact (score ~0.35). Try different age groups to explore variations.",
        'vegetarian': "Vegetarians show low impacts, especially in greenhouse gas emissions. Compare with pescatarians.",
        'pescatarian': "Pescatarians have moderate impacts, but water use varies by age. Try selecting 60-79 years.",
        'low_meat': "Low-meat diets are better than high-meateat but higher than plant-based. Check land use patterns.",
        'medium_meat': "Medium-meat diets show increased impacts, particularly in land use. Explore gender differences.",
        'high_meat': "High-meat diets have the highest impact (score ~0.85). See how age affects this."
    }

    if not selected_diets:
        insight = "Select at least one diet category to see key environmental impact insights."
    elif len(selected_diets) == 1:
        insight = insight_map.get(selected_diets[0], "Explore the data to uncover patterns.")
    else:
        insight = "Multiple diet categories selected. Compare their environmental impacts using the treemap or parallel coordinates plot."

    # Return the chart based on view type
    if view_type == 'treemap':
        return dcc.Graph(figure=fig_treemap), insight
    elif view_type == 'parcoords':
        return dcc.Graph(figure=fig_parcoords), insight
    else:  # Combined view
        return html.Div([
            dcc.Graph(figure=fig_treemap, style={'width': '70%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_parcoords, style={'width': '30%', 'display': 'inline-block'})
        ]), insight
# Run the application
if __name__ == '__main__':
    app.run(debug=True)
