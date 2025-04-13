import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import squarify
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pandas.plotting import parallel_coordinates
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from scipy.stats import ttest_ind


# 1. load dataset
file_path = "Results_21Mar2022.csv" 
df = pd.read_csv(file_path)

# 2. Check data structure
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
print("Data Preview:")
print(df.head())

# Display data types without excessive output
print("Data Types:")
print(df.dtypes)

# Check missing values (sorted by percentage)
print("Missing Values (sorted by percentage):")
missing_values = df.isnull().sum() / len(df) * 100  # Calculate missing value percentage
print(missing_values[missing_values > 0].sort_values(ascending=False))  # Show only columns with missing values

# Check unique values for categorical data (first 10 columns)
print("Unique Values Per Column (first 10 columns):")
print(df.nunique().head(10))

# 3. Data processing
# Calculating PCA-based weights
indicators = ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut']
X = df[indicators]

# normalized data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#print(f"The shape of the data after normalization: {X_scaled.shape}")
# Run PCA
pca = PCA(n_components=1)
pca.fit(X_scaled)
loadings = pca.components_[0]
weights_pca = loadings**2 / sum(loadings**2)  # The normalized sum of squares of the loadings is the weight
df_pca = pca.fit_transform(X_scaled)
#print(f"Shape of data after PCA: {df_pca.shape}")
# Define weight dictionary
weights = dict(zip(indicators, weights_pca))
#print("Calculated PCA Weights:")
for indicator, weight in weights.items():
    print(f"{indicator}: {weight:.3f}")
print(f"Explained variance ratio by PC1: {pca.explained_variance_ratio_[0]:.3f}")



# Aggregation 
diet_gender_impact = df.groupby(['diet_group', 'sex']).agg({
    'mean_ghgs': ['mean', 'std'],
    'mean_land': ['mean', 'std'],
    'mean_watscar': ['mean', 'std'], 
    'mean_eut': ['mean', 'std'], 
    'n_participants': 'sum'
}).reset_index()

# Modify the column name (remove MultiIndex) and convert the multi-layer index to a single layer
diet_gender_impact.columns = ['_'.join(col).strip('_') for col in diet_gender_impact.columns]
## View the modified column names
#print("Modified column name：")
#print(diet_gender_impact.columns.tolist())  # Output column names as a list

## View the first few rows of a DataFrame 
#print("\nFirst 5 rows of DataFrame：")
#print(diet_gender_impact.head())

# Add total environmental impact score (weights calculated using PCA)
for col in indicators:
    df[col + '_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
df['total_impact_score'] = (df['mean_ghgs_norm'] * weights['mean_ghgs'] +
                            df['mean_land_norm'] * weights['mean_land'] +
                            df['mean_watscar_norm'] * weights['mean_watscar'] +
                            df['mean_eut_norm'] * weights['mean_eut'])


# Check the calculation of total_impact_score
print("Total Impact Score Preview:")
print(df[['diet_group', 'sex', 'age_group', 'total_impact_score']].head())


# Detecting outliers using the IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Modify the get_all_outliers function to include description
def get_all_outliers(df, indicators):
    outliers = pd.concat([detect_outliers_iqr(df, col) for col in indicators]).drop_duplicates()
    # Merge diet_metadata to add description column
    outliers = pd.merge(outliers, diet_metadata[['diet_group', 'diet_category', 'description']], 
                        on=['diet_group', 'diet_category'], how='left')
    return outliers
# Analyze trends by age group and diet type
age_diet_trend = df.groupby(['age_group', 'diet_group'])['total_impact_score'].mean().unstack()
print(age_diet_trend.head())


# View unique values ​​to understand categorical variables
print("Diet groups:", df['diet_group'].unique())
print("Age groups:", df['age_group'].unique())

# # Calculate the relative impact of each dietary group 
vegan_ghgs = df[df['diet_group'] == 'vegan']['mean_ghgs'].mean()
df['ghgs_relative_to_vegan'] = df['mean_ghgs'] / vegan_ghgs

# Add diet type category
def categorize_diet(diet):
    if 'vegan' in diet:
        return 'plant_based'
    elif 'veggie' in diet:
        return 'vegetarian'
    elif 'fish' in diet:
        return 'pescatarian'
    elif 'meat100' in diet:
        return 'high_meat'
    elif 'meat50' in diet:
        return 'medium_meat'
    else:
        return 'low_meat'

df['diet_category'] = df['diet_group'].apply(categorize_diet)

# Add median age for age group
age_mapping = {
    '20-29': 25,
    '30-39': 35,
    '40-49': 45,
    '50-59': 55,
    '60-69': 65,
    '70-79': 75
}
df['mid_age'] = df['age_group'].map(age_mapping)

diet_metadata = pd.DataFrame({
    'diet_group': ['vegan', 'veggie', 'fish', 'meat50', 'meat', 'meat100'],
    'diet_category': ['plant_based', 'vegetarian', 'pescatarian', 'low_meat', 'medium_meat', 'high_meat'],
    'meat_consumption_mean': [0.3, 0.4, 2.0, 28.3, 74.0, 140],  # Table 1 Actual mean values ​​of the paper (Vegans, vegetarians, fish-eaters and meat-eaters in the UK show discrepant environmental impacts | Nature Food)
    'description': [
        'No animal products ',
        'Includes dairy and eggs, no meat ',
        'Includes fish, theoretically no meat ',
        'Average meat intake 28.3 g/day (<50 g/day)',
        'Average meat intake 74.0 g/day (50-99 g/day)',
        'Average meat intake 140 g/day (≥100 g/day)'
    ]
})

# Print the results to check
# print("Updated diet groups in df:", df['diet_group'].unique())
# print("Diet categories:", df['diet_category'].unique())
# print("Mid age values:", df['mid_age'].unique())
# print("\nDiet Metadata:\n", diet_metadata)


# # Calculate the average emissions for the meat50 group
# meat50_ghgs = df[df['diet_group'] == 'meat50']['mean_ghgs'].mean()
# print("meat50 GHG emissions:", meat50_ghgs)

# # Calculate the average emissions for the meat100 group
# meat100_ghgs = df[df['diet_group'] == 'meat100']['mean_ghgs'].mean()
# print("meat100 GHG emissions:", meat100_ghgs)  # 修正：打印 meat100_ghgs

# # Calculate the average emissions for the meat group
# meat_ghgs = df[df['diet_group'] == 'meat']['mean_ghgs'].mean()
# print("meat GHG emissions:", meat_ghgs)  # 修正：打印 meat_ghgs
# #meat50 ➜ low_meat（<50 g）=5.80~5.37,meat ➜ medium_meat（50–99 g）=7.55~7.04,meat100 ➜ high_meat（≥100 g）=11.43~10.24(根据论文，Table3)

# Environmental indicators table
impact_metadata = pd.DataFrame({
    'metric': ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut'],
    'full_name': [
        'Greenhouse gas emissions (kg CO2-eq/day)',
        'Land use (m²/year)',
        'Water scarcity (liters/day)',
        'Eutrophication (g PO4-eq/day)'
    ],
    'importance': ['High', 'High', 'Medium', 'Medium'],
    'data_source': ['IPCC', 'FAO', 'WaterFootprint', 'EPA']
})

# Calculate the environmental impact percentile for each dietary category
for metric in ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut']:
    df[f'{metric}_percentile'] = df.groupby('diet_category')[metric].rank(pct=True)

# Preparing data for trend analysis
trend_data = df.groupby(['mid_age', 'diet_category']).agg({
    'mean_ghgs': 'mean',
    'mean_land': 'mean',
    'mean_watscar': 'mean',  
    'n_participants': 'sum'
}).reset_index()

# Calculate the diet distribution for each age group
age_diet_dist = pd.crosstab(df['age_group'], df['diet_category'],
                            values=df['n_participants'], aggfunc='sum', normalize='index')

# Save data
df.to_csv('enhanced_environmental_impact.csv', index=False)
diet_metadata.to_csv('diet_metadata.csv', index=False)
impact_metadata.to_csv('impact_metadata.csv', index=False)
trend_data.to_csv('age_trend_data.csv', index=False)
age_diet_dist.to_csv('age_diet_distribution.csv', index=False)


print("Data preprocessing completed!")

#Merge df and diet_metadata to keep the description information of each sample
df_merged = pd.merge(df, diet_metadata, on=['diet_group', 'diet_category'], how='left')


# Aggregate data, including multivariate indicators
treemap_df = df_merged.groupby(['diet_category', 'description', 'age_group', 'sex']).agg({
    'total_impact_score': 'mean',
    'n_participants': 'sum',
    'mean_ghgs': 'mean',
    'mean_land': 'mean',
    'mean_watscar': 'mean',
    'mean_eut': 'mean'
}).reset_index()

# Standardize multivariate indicators
for col in ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut']:
    treemap_df[f'{col}_norm'] = (treemap_df[col] - treemap_df[col].min()) / (treemap_df[col].max() - treemap_df[col].min())

treemap_df['total_impact_score'] = treemap_df['total_impact_score'].round(3)

# Calculate the average score for each age group and display it
age_group_summary = df.groupby('age_group')['total_impact_score'].mean().reset_index()

# # Calculate the best and worst diet scores and the difference
impact_summary = df.groupby('diet_category')['total_impact_score'].mean().reset_index()
best = impact_summary.loc[impact_summary['total_impact_score'].idxmin()]
worst = impact_summary.loc[impact_summary['total_impact_score'].idxmax()]
disparity_ratio = worst['total_impact_score'] / best['total_impact_score']

# Find the best age group
best_age_group = age_group_summary.loc[age_group_summary['total_impact_score'].idxmin()]

# Initialize the Dash application
app = dash.Dash(__name__)

# define color
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
            html.Label("Select Diet Category:", style={'fontFamily': 'Arial', 'marginRight': '10px', 'fontSize': '16px', 'display': 'inline-block'}),
            dcc.Checklist(
                id='diet-checklist',
                options=[{'label': 'All', 'value': 'All'}] + 
                        [{'label': cat.capitalize(), 'value': cat} for cat in treemap_df['diet_category'].unique()],
                value=['All'],  # Select all by default, including All
                inline=True,
                style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'middle', 'fontSize': '14px', 'marginRight': '20px'}
            ),
            html.Label("Select Age Group:", style={'fontFamily': 'Arial', 'marginRight': '10px', 'fontSize': '16px', 'display': 'inline-block'}),
            dcc.Checklist(
                id='age-checklist',
                options=[{'label': 'All', 'value': 'All'}] + 
                        [{'label': age, 'value': age} for age in treemap_df['age_group'].unique()],
                value=['All'],  # Select all by default, including All
                inline=True,
                style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'middle', 'fontSize': '14px', 'marginRight': '20px'}
            ),
        html.Label("Select gender:", style={'fontFamily': 'Arial', 'marginRight': '10px', 'fontSize': '16px', 'display': 'inline-block'}),
        dcc.Dropdown(
            id='sex-dropdown',
            options=[{'label': 'All', 'value': 'All'}] + 
                    [{'label': sex.capitalize(), 'value': sex} for sex in treemap_df['sex'].unique()],
            value='All',
            style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'middle', 'fontSize': '14px'}
        ),
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderBottom': '1px solid #dee2e6', 'display': 'flex', 'alignItems': 'center'}),
    html.Div([
        html.Label("Select a view:", style={'fontFamily': 'Arial', 'marginRight': '10px', 'fontSize': '16px'}),
        dcc.RadioItems(
            id='view-toggle',
            options=[
                {'label': 'Treemap', 'value': 'treemap'},
                {'label': 'Parallel Coordinates Plot', 'value': 'parcoords'},
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
        html.P(f"🥦 Best Diet: {best['diet_category']} (Score={best['total_impact_score']:.3f})", style={'color': '#28a745', 'fontSize': '16px'}),
        html.P(f"🥩 Worst diet: {worst['diet_category']} (Score={worst['total_impact_score']:.3f})", style={'color': '#dc3545', 'fontSize': '16px'}),
        html.P(f"disparity: {disparity_ratio:.2f}x", style={'color': '#007bff', 'fontSize': '16px'}),
        html.P(f"👶 Best age group: {best_age_group['age_group']} (Score={best_age_group['total_impact_score']:.3f})", style={'color': '#17a2b8', 'fontSize': '16px'}),
    ], style={'padding': '20px', 'backgroundColor': '#e9ecef', 'borderRadius': '5px', 'margin': '10px'}),
    html.Div(id='insight-text', style={'padding': '20px', 'fontSize': '16px', 'fontFamily': 'Arial', 'color': '#6c757d', 'backgroundColor': '#f1f3f5', 'borderRadius': '5px'})
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
    df_filtered = treemap_df.copy()  # Use a copy to avoid modifying the original data
    title = "Hierarchical Distribution of Environmental Impact Across Diet Groups"
    print("df_filtered total_impact_score 统计：")
    print(df_filtered['total_impact_score'].describe())


#Processing diet-checklist's All logic
    all_diets = list(treemap_df['diet_category'].unique())
    if 'All' in selected_diets:
        selected_diets = all_diets # Select All to include all diet categories
    elif not selected_diets:  # If All is cancelled and no other options are available
        selected_diets = []  # Display empty data (or set to all_diets, depending on your needs)
    
  
    all_ages = list(treemap_df['age_group'].unique())
    if 'All' in selected_age:
        selected_age = all_ages  # Select All to include all age groups
    elif not selected_age:  # If All is cancelled and no other options are available
        selected_age = []  # Display empty data

    # Define filter conditions
    filters = {
        'diet_category': selected_diets,
        'age_group': selected_age,
        'sex': selected_sex
    }
    
    # Apply filtering
    for column, value in filters.items():
        if value != 'All' and value:  
            if isinstance(value, list): 
                df_filtered = df_filtered[df_filtered[column].isin(value)]
               
            else: 
                df_filtered = df_filtered[df_filtered[column] == value]
              
    
    # Handling outliers
    if 'show' in show_outliers:
        all_outliers = get_all_outliers(df, indicators)
       # Ensure that there are no NaNs in the path column of the outlier data
        all_outliers = all_outliers.dropna(subset=['diet_category', 'description', 'age_group', 'sex'])
        # Only keep columns that are consistent with treemap_df
        required_cols = ['diet_category', 'description', 'age_group', 'sex', 'n_participants', 'total_impact_score',
                         'mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut']
        all_outliers = all_outliers[all_outliers.columns.intersection(required_cols)]
        df_filtered = pd.concat([df_filtered, all_outliers], ignore_index=True)
    
    # Ensure df_filtered has no NaNs
    df_filtered = df_filtered.dropna(subset=['diet_category', 'description', 'age_group', 'sex'])
    
    # # Check data 
    print("df_filtered NaN 检查：")
    print(df_filtered[['diet_category', 'description', 'age_group', 'sex']].isnull().sum())
 
    print("df_filtered total_impact_score 统计（显示异常值={}）：".format('show' in show_outliers))
    print(df_filtered['total_impact_score'].describe())
    # Create a tree diagram
    fig_treemap = px.treemap(
        df_filtered,
        path=['diet_category', 'description', 'age_group', 'sex'],
        values='n_participants',
        color='total_impact_score',
        color_continuous_scale=custom_colors,
        title=title
    )
    fig_treemap.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        coloraxis_colorbar=dict(title="Total Impact Rating")
    )
    
    # Parallel coordinate plot
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
    # Dynamic Insights (Obvious Observations)
    insight_map = {
        'plant_based': "Vegans typically have the lowest environmental impact (score ~0.35). Try different age groups to explore variations.",
        'vegetarian': "Vegetarians show low impacts, especially in greenhouse gas emissions. Compare with pescatarians.",
        'pescatarian': "Pescatarians have moderate impacts, but water use varies by age. Try selecting 60-79 years.",
        'low_meat': "Low-meat diets are better than high-meat but higher than plant-based. Check land use patterns.",
        'medium_meat': "Medium-meat diets show increased impacts, particularly in land use. Explore gender differences.",
        'high_meat': "High-meat diets have the highest impact (score ~0.85). See how age affects this."
    }
    
    
    if not selected_diets: # If no diet category is selected
        insight = "Select at least one diet category to see key environmental impact insights."
    elif len(selected_diets) == 1:  # If only one diet category is selected
        insight = insight_map.get(selected_diets[0], "Explore the data to uncover patterns.")
    else:  # If you select multiple diet categories
        insight = "Multiple diet categories selected. Compare their environmental impacts using the treemap or parallel coordinates plot."
    
    
    # Return the chart based on the view type
    if view_type == 'treemap':
        return dcc.Graph(figure=fig_treemap), insight
    elif view_type == 'parcoords':
        return dcc.Graph(figure=fig_parcoords), insight
    else: #Combined view
        return html.Div([
            dcc.Graph(figure=fig_treemap, style={'width': '70%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_parcoords, style={'width': '30%', 'display': 'inline-block'})
        ]), insight

# Run the application
if __name__ == '__main__':
    print("treemap_df NaN 检查：")
   

    print(treemap_df[['diet_category', 'description', 'age_group', 'sex']].isnull().sum())

    print("\n=== Analysis verification output (for reporting) ===")

    # 老年男性 pescatarian（60-69 & 70-79）
    elder_male_pescatarian = df[
        (df['diet_category'] == 'pescatarian') &
        (df['age_group'].isin(['60-69', '70-79'])) &
        (df['sex'] == 'male')
    ]
    elder_male_avg = elder_male_pescatarian['mean_watscar'].mean()
    print(f"Average water consumption among older male pescatarians: {elder_male_avg:.2f} L/day")

    # All men aged 60–79 years (regardless of diet)
    elder_male_all = df[
        (df['age_group'].isin(['60-69', '70-79'])) &
        (df['sex'] == 'male')
    ]
    elder_male_all_avg = elder_male_all['mean_watscar'].mean()
    print(f"Average water consumption for men aged 60-79: {elder_male_all_avg:.2f} L/day")

   # Difference percentage calculation
    diff_ratio = elder_male_avg / elder_male_all_avg
    diff_percent = (diff_ratio - 1) * 100
    print(f"Older male pescatarians outperformed males of the same age by: {diff_percent:.1f}%")

   # Young pescatarian (20-29 & 30-39) regardless of gender
    young_pescatarian = df[
        (df['diet_category'] == 'pescatarian') &
        (df['age_group'].isin(['20-29', '30-39']))
    ]
    young_avg = young_pescatarian['mean_watscar'].mean()
    print(f"年轻 pescatarian 平均水资源消耗: {young_avg:.2f} L/day")

  # t-test: old vs young pescatarian (regardless of sex)
    elder_pescatarian = df[
        (df['diet_category'] == 'pescatarian') &
        (df['age_group'].isin(['60-69', '70-79']))
    ]
    t_stat1, p_val1 = ttest_ind(
        elder_pescatarian['mean_watscar'],
        young_pescatarian['mean_watscar']
    )
    print(f"T-test（elder vs young pescatarian）: t = {t_stat1:.2f}, p = {p_val1:.4f}")

# t test: elderly male vs female pescatarian
    elder_female_pescatarian = df[
        (df['diet_category'] == 'pescatarian') &
        (df['age_group'].isin(['60-69', '70-79'])) &
        (df['sex'] == 'female')
    ]
    t_stat2, p_val2 = ttest_ind(
        elder_male_pescatarian['mean_watscar'],
        elder_female_pescatarian['mean_watscar']
    )
    print(f"T-test(elder male vs female pescatarian): t = {t_stat2:.2f}, p = {p_val2:.4f}")

    print("=== Analysis and verification completed ===")
 


    app.run(debug=True)
