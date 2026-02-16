"""
High School Football Subscription Predictor
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Subscription Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    """Load the trained model and feature names"""
    try:
        with open('models.pkl', 'rb') as f:
            models = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('processed_data.csv') as f:
            df_historical = pd.read_csv(f)
            df_historical['Start Time'] = pd.to_datetime(df_historical['Start Time'])
        
        return models['Gradient Boosting'], feature_names, df_historical
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def get_school_avg(school_name, school_type, df_historical, home_avg_dict, away_avg_dict):
    """Get historical average for a school"""
    if school_type == 'home':
        return home_avg_dict.get(school_name, away_avg_dict.get(school_name, 5.0))
    else:
        return away_avg_dict.get(school_name, home_avg_dict.get(school_name, 5.0))

def engineer_features(df, df_historical, home_avg_dict, away_avg_dict):
    """Engineer features for prediction"""
    df = df.copy()
    
    # Ensure Start Time is datetime
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    
    # Time features
    df['hour'] = df['Start Time'].dt.hour
    df['day_of_week'] = df['Start Time'].dt.dayofweek
    df['month'] = df['Start Time'].dt.month
    df['day_of_year'] = df['Start Time'].dt.dayofyear
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_evening'] = (df['hour'] >= 18).astype(int)
    df['is_primetime'] = (df['hour'].isin([19, 20])).astype(int)
    
    # Clean Game Type
    df['Game Type'] = df['Game Type'].str.lower().str.strip()
    
    # School historical averages
    df['home_school_hist_avg'] = df['Home School'].apply(
        lambda x: get_school_avg(x, 'home', df_historical, home_avg_dict, away_avg_dict)
    )
    df['away_school_hist_avg'] = df['Away School'].apply(
        lambda x: get_school_avg(x, 'away', df_historical, home_avg_dict, away_avg_dict)
    )
    
    # Matchup features
    df['matchup'] = df['Home School'] + '_vs_' + df['Away School']
    historical_matchups = set(df_historical['matchup'].values)
    df['is_repeat_matchup'] = df['matchup'].apply(
        lambda x: 1 if x in historical_matchups else 0
    ).astype(int)
    
    # State grouping
    top_states = ['CA', 'TX', 'MI', 'GA', 'AL', 'NC', 'IL', 'VA', 'WA', 
                  'FL', 'OH', 'TN', 'NM', 'MS', 'DE']
    df['State_Group'] = df['State Code'].apply(
        lambda x: x if x in top_states else 'Other'
    )
    
    # Fill missing distances
    df['Distance Between Schools'] = df['Distance Between Schools'].fillna(26.3)
    
    return df

def prepare_features(df, feature_names):
    """Prepare features for model prediction"""
    feature_cols = [
        'hour', 'day_of_week', 'month', 'Weekofsy', 'day_of_year',
        'is_friday', 'is_weekend', 'is_evening', 'is_primetime',
        'Distance Between Schools',
        'home_school_hist_avg', 'away_school_hist_avg',
        'is_repeat_matchup',
        'State_Group', 'Game Type'
    ]
    
    X = df[feature_cols].copy()
    X = pd.get_dummies(X, columns=['State_Group', 'Game Type'], drop_first=True)
    
    # Ensure all features from training are present
    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = 0
    
    return X[feature_names]

def make_predictions(df_input, model, feature_names, df_historical, home_avg_dict, away_avg_dict):
    """Make predictions on input data"""
    # Engineer features
    df_processed = engineer_features(df_input, df_historical, home_avg_dict, away_avg_dict)
    
    # Prepare features
    X = prepare_features(df_processed, feature_names)
    
    # Make predictions
    predictions = model.predict(X.values)
    predictions = np.maximum(predictions, 1)  # Minimum 1 subscription
    
    # Add predictions to dataframe
    df_processed['Predicted_Subscriptions'] = predictions.round(1)
    
    return df_processed

def create_summary_stats(df):
    """Create summary statistics"""
    stats = {
        'total_games': len(df),
        'total_subs': df['Predicted_Subscriptions'].sum(),
        'avg_subs': df['Predicted_Subscriptions'].mean(),
        'median_subs': df['Predicted_Subscriptions'].median(),
        'max_subs': df['Predicted_Subscriptions'].max(),
        'min_subs': df['Predicted_Subscriptions'].min()
    }
    return stats

# Main app
def main():
    st.markdown('<p class="main-header">🏈 High School Football Subscription Predictor</p>', unsafe_allow_html=True)
    
    # Load model
    model, feature_names, df_historical = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please ensure model files are in the same directory.")
        return
    
    # Build lookup dictionaries
    home_avg_dict = {}
    away_avg_dict = {}
    
    for school in df_historical['Home School'].unique():
        school_data = df_historical[df_historical['Home School'] == school]['home_school_hist_avg']
        school_data = school_data[school_data.notna()]
        if len(school_data) > 0:
            home_avg_dict[school] = school_data.iloc[-1]
    
    for school in df_historical['Away School'].unique():
        school_data = df_historical[df_historical['Away School'] == school]['away_school_hist_avg']
        school_data = school_data[school_data.notna()]
        if len(school_data) > 0:
            away_avg_dict[school] = school_data.iloc[-1]
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload** your game schedule (Excel or CSV)
        2. **Review** the predictions and insights
        3. **Download** the results
        
        ### Required Columns:
        - `Start Time` (datetime)
        - `Home School` (text)
        - `Away School` (text)
        - `State Code` (2-letter code)
        - `Weekofsy` (week number)
        - `Game Type` (text)
        - `Distance Between Schools` (optional)
        """)
        
        st.divider()
        
        st.header("ℹ️ About")
        st.markdown("""
        This model predicts subscription numbers based on:
        - School historical performance
        - Game timing and location
        - Distance between schools
        - Week of season
        
        **Model Accuracy**: ±3.8 subscriptions on average
        """)
    
    # Main content
    st.header("📤 Upload Game Schedule")
    
    uploaded_file = st.file_uploader(
        "Choose a file (Excel or CSV)",
        type=['xlsx', 'xls', 'csv'],
        help="Upload your game schedule with required columns"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Successfully loaded {len(df_input):,} games!")
            
            # Show preview
            with st.expander("📊 Preview uploaded data"):
                st.dataframe(df_input.head(10), use_container_width=True)
            
            # Make predictions
            with st.spinner('🔮 Generating predictions...'):
                df_predictions = make_predictions(
                    df_input, model, feature_names, df_historical, 
                    home_avg_dict, away_avg_dict
                )
                stats = create_summary_stats(df_predictions)
            
            st.success("✅ Predictions complete!")
            
            # Display summary metrics
            st.header("📊 Summary Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Games",
                    value=f"{stats['total_games']:,}"
                )
            
            with col2:
                st.metric(
                    label="Total Predicted Subs",
                    value=f"{stats['total_subs']:,.0f}"
                )
            
            with col3:
                st.metric(
                    label="Average per Game",
                    value=f"{stats['avg_subs']:.1f}"
                )
            
            with col4:
                st.metric(
                    label="Highest Game",
                    value=f"{stats['max_subs']:.0f}"
                )
            
            # Visualizations
            st.header("📈 Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Top Games", "By State", "By Month"])
            
            with tab1:
                # Distribution histogram
                fig = px.histogram(
                    df_predictions, 
                    x='Predicted_Subscriptions',
                    nbins=50,
                    title='Distribution of Predicted Subscriptions',
                    labels={'Predicted_Subscriptions': 'Predicted Subscriptions'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Range breakdown
                bins = [0, 3, 5, 8, 12, 20, 50, 2000]
                labels = ['1-3', '4-5', '6-8', '9-12', '13-20', '21-50', '50+']
                df_predictions['Range'] = pd.cut(df_predictions['Predicted_Subscriptions'], bins=bins, labels=labels)
                range_counts = df_predictions['Range'].value_counts().sort_index()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Games by Prediction Range:**")
                    for category, count in range_counts.items():
                        pct = count / len(df_predictions) * 100
                        st.write(f"- {category}: {count:,} games ({pct:.1f}%)")
            
            with tab2:
                # Top 20 games
                st.subheader("🏆 Top 20 Highest Predicted Games")
                top_20 = df_predictions.nlargest(20, 'Predicted_Subscriptions')[
                    ['Start Time', 'Home School', 'Away School', 'State Code', 
                     'Predicted_Subscriptions', 'home_school_hist_avg', 'away_school_hist_avg']
                ].copy()
                top_20['Start Time'] = top_20['Start Time'].dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(
                    top_20.style.format({
                        'Predicted_Subscriptions': '{:.1f}',
                        'home_school_hist_avg': '{:.1f}',
                        'away_school_hist_avg': '{:.1f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            
            with tab3:
                # By state
                state_stats = df_predictions.groupby('State Code').agg({
                    'Predicted_Subscriptions': ['count', 'sum', 'mean']
                }).round(1)
                state_stats.columns = ['Games', 'Total Subs', 'Avg per Game']
                state_stats = state_stats.sort_values('Total Subs', ascending=False).head(15)
                
                fig = px.bar(
                    state_stats.reset_index(),
                    x='State Code',
                    y='Total Subs',
                    title='Top 15 States by Total Predicted Subscriptions',
                    labels={'Total Subs': 'Total Subscriptions'},
                    color='Avg per Game',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(state_stats, use_container_width=True)
            
            with tab4:
                # By month
                df_predictions['Month'] = pd.to_datetime(df_predictions['Start Time']).dt.to_period('M')
                monthly_stats = df_predictions.groupby('Month').agg({
                    'Predicted_Subscriptions': ['count', 'sum', 'mean']
                }).round(1)
                monthly_stats.columns = ['Games', 'Total Subs', 'Avg per Game']
                monthly_stats.index = monthly_stats.index.astype(str)
                
                fig = px.bar(
                    monthly_stats.reset_index(),
                    x='Month',
                    y='Total Subs',
                    title='Monthly Subscription Forecast',
                    labels={'Total Subs': 'Total Subscriptions'},
                    color='Avg per Game',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(monthly_stats, use_container_width=True)
            
            # Download section
            st.header("💾 Download Results")
            
            # Prepare output dataframe
            output_cols = [
                'Start Time', 'Sport', 'Level', 'Game Type', 'State Code',
                'Home School', 'Away School', 'Distance Between Schools',
                'Weekofsy', 'Predicted_Subscriptions',
                'home_school_hist_avg', 'away_school_hist_avg'
            ]
            
            # Only include columns that exist
            available_cols = [col for col in output_cols if col in df_predictions.columns]
            df_output = df_predictions[available_cols].copy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Excel download
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_output.to_excel(writer, index=False, sheet_name='Predictions')
                output.seek(0)
                
                st.download_button(
                    label="📥 Download Excel",
                    data=output,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # CSV download
                csv = df_output.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
    
    else:
        # Show example format
        st.info("👆 Upload a file to get started")
        
        with st.expander("📋 Example file format"):
            example_data = {
                'Sport': ['Football', 'Football'],
                'Start Time': ['2026-09-04 19:00:00', '2026-09-05 20:00:00'],
                'Weekofsy': [1, 1],
                'Level': ['Varsity', 'Varsity'],
                'Game Type': ['Regular Season', 'Regular Season'],
                'State Code': ['TX', 'CA'],
                'Home School': ['school-1', 'school-2'],
                'Away School': ['school-3', 'school-4'],
                'Distance Between Schools': [25.5, 45.0]
            }
            st.dataframe(pd.DataFrame(example_data), use_container_width=True)

if __name__ == "__main__":
    main()
