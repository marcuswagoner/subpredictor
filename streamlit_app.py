"""
High School Football Subscription Predictor - v2
Updated model with State Rankings, Win Percentages, and Followers
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(
    page_title="Subscription Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.4rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1.5rem; }
    .version-tag { text-align: center; color: #888; font-size: 0.85rem; margin-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ── Load model assets ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        models       = pickle.load(open('models_v2.pkl',       'rb'))
        feature_names = pickle.load(open('feature_names_v2.pkl', 'rb'))
        df_hist       = pd.read_csv('processed_data_v2.csv')
        df_hist['Start Time Et'] = pd.to_datetime(df_hist['Start Time Et'])

        school_avgs = json.load(open('school_avgs_v2.json'))
        home_avgs   = school_avgs['home']
        away_avgs   = school_avgs['away']

        return models['Gradient Boosting'], feature_names, df_hist, home_avgs, away_avgs
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None, None


def get_school_avg(slug, which, home_avgs, away_avgs, default=5.0):
    if which == 'home':
        return home_avgs.get(slug, {}).get('mean', away_avgs.get(slug, {}).get('mean', default))
    return away_avgs.get(slug, {}).get('mean', home_avgs.get(slug, {}).get('mean', default))


def engineer_and_predict(df_input, model, feature_names, df_hist, home_avgs, away_avgs):
    df = df_input.copy()

    # Detect time column
    time_col = 'Start Time Et' if 'Start Time Et' in df.columns else 'Start Time'
    df['_dt'] = pd.to_datetime(df[time_col])

    # Time features
    df['hour']        = df['_dt'].dt.hour
    df['day_of_week'] = df['_dt'].dt.dayofweek
    df['month']       = df['_dt'].dt.month
    df['day_of_year'] = df['_dt'].dt.dayofyear
    df['Weekofsy']    = df.get('Weekofsy', ((df['_dt'] - pd.Timestamp('2022-08-20')).dt.days // 7) % 52 + 1)
    df['is_friday']   = (df['day_of_week'] == 4).astype(int)
    df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_evening']  = (df['hour'] >= 18).astype(int)
    df['is_primetime']= df['hour'].isin([19, 20]).astype(int)

    # Pixellot
    if 'Pixellot' in df.columns:
        df['is_pixellot'] = (df['Pixellot'] == 'Yes').astype(int)
    else:
        df['is_pixellot'] = 0  # Unknown — default to 0

    # Game Type
    df['Game Type'] = df['Game Type'].str.lower().str.strip() if 'Game Type' in df.columns else 'regular season'

    # School slugs — fall back to Home/Away School if slugs not provided
    home_slug_col = 'Home Slug' if 'Home Slug' in df.columns else 'Home School'
    away_slug_col = 'Away Slug' if 'Away Slug' in df.columns else 'Away School'

    df['home_school_hist_avg'] = df[home_slug_col].apply(lambda x: get_school_avg(x, 'home', home_avgs, away_avgs))
    df['away_school_hist_avg'] = df[away_slug_col].apply(lambda x: get_school_avg(x, 'away', home_avgs, away_avgs))

    # Matchup
    df['matchup']         = df[home_slug_col] + '_vs_' + df[away_slug_col]
    hist_matchups         = set(df_hist['matchup'].values) if 'matchup' in df_hist.columns else set()
    df['is_repeat_matchup']= df['matchup'].isin(hist_matchups).astype(int)

    # Rankings
    df['Staterankhome'] = df.get('Staterankhome', pd.Series([300]*len(df))).clip(upper=500).fillna(300)
    df['Staterankaway'] = df.get('Staterankaway', pd.Series([300]*len(df))).clip(upper=500).fillna(300)
    df['combined_rank']     = df['Staterankhome'] + df['Staterankaway']
    df['rank_differential'] = (df['Staterankhome'] - df['Staterankaway']).abs()
    df['both_top50']        = ((df['Staterankhome'] <= 50) & (df['Staterankaway'] <= 50)).astype(int)

    # Win percentages
    df['Hometeamwinpercentage'] = df.get('Hometeamwinpercentage', pd.Series([0.5]*len(df))).fillna(0.5)
    df['Awayteamwinpercentage'] = df.get('Awayteamwinpercentage', pd.Series([0.5]*len(df))).fillna(0.5)
    df['combined_win_pct'] = df['Hometeamwinpercentage'] + df['Awayteamwinpercentage']
    df['win_pct_diff']     = (df['Hometeamwinpercentage'] - df['Awayteamwinpercentage']).abs()

    # Followers
    med_followers = 481
    df['Followers Home'] = df.get('Followers Home', pd.Series([med_followers]*len(df))).fillna(med_followers)
    df['Followers Away'] = df.get('Followers Away', pd.Series([med_followers]*len(df))).fillna(med_followers)
    df['log_followers_home']  = np.log1p(df['Followers Home'])
    df['log_followers_away']  = np.log1p(df['Followers Away'])
    df['total_followers']     = df['Followers Home'] + df['Followers Away']
    df['log_total_followers'] = np.log1p(df['total_followers'])

    # Distance
    df['Distance Between Schools'] = df.get('Distance Between Schools', pd.Series([26.3]*len(df))).fillna(26.3)

    # State group
    top_states = ['CA', 'TX', 'MI', 'GA', 'AL', 'NC', 'IL', 'VA', 'WA',
                  'FL', 'OH', 'TN', 'NM', 'MS', 'DE']
    df['State_Group'] = df['State Code'].apply(lambda x: x if x in top_states else 'Other')

    # Encode & predict
    base_cols = [
        'hour', 'day_of_week', 'month', 'Weekofsy', 'day_of_year',
        'is_friday', 'is_weekend', 'is_evening', 'is_primetime',
        'home_school_hist_avg', 'away_school_hist_avg',
        'Distance Between Schools', 'is_repeat_matchup',
        'Staterankhome', 'Staterankaway', 'combined_rank',
        'rank_differential', 'both_top50',
        'Hometeamwinpercentage', 'Awayteamwinpercentage',
        'combined_win_pct', 'win_pct_diff',
        'log_followers_home', 'log_followers_away', 'log_total_followers',
        'is_pixellot', 'State_Group', 'Game Type'
    ]

    X = df[base_cols].copy()
    X = pd.get_dummies(X, columns=['State_Group', 'Game Type'], drop_first=True)
    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = 0
    X = X[feature_names]

    preds = np.maximum(np.expm1(model.predict(X.values)), 1)
    df['Predicted_Subscriptions'] = preds.round(1)

    return df


# ── App layout ───────────────────────────────────────────────────────
def main():
    st.markdown('<p class="main-header">🏈 Subscription Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="version-tag">v2.0 — with State Rankings, Win %, & Followers</p>', unsafe_allow_html=True)

    model, feature_names, df_hist, home_avgs, away_avgs = load_model()
    if model is None:
        return

    # Sidebar
    with st.sidebar:
        st.header("📋 How to Use")
        st.markdown("""
        1. **Upload** your game schedule (Excel or CSV)
        2. **View** predictions and insights
        3. **Download** results

        ---

        ### Required Columns
        | Column | Example |
        |--------|---------|
        | Start Time Et | 2026-09-04 19:00 |
        | State Code | TX |
        | Home Slug | school-name-city-tx |
        | Away Slug | school-name-city-al |
        | Game Type | regular season |
        | Weekofsy | 3 |

        ### Optional (improve accuracy)
        - `Staterankhome` / `Staterankaway`
        - `Hometeamwinpercentage` / `Awayteamwinpercentage`
        - `Followers Home` / `Followers Away`
        - `Pixellot` (Yes/No)
        - `Distance Between Schools`

        ---
        """)

        st.subheader("📊 Model Info")
        st.markdown("""
        **Algorithm**: Gradient Boosting  
        **Training games**: 56,923  
        **Regular season MAE**: ±3.1 subs  
        **R² score**: 0.557  

        **Top Predictors:**
        1. Day of year / week of season  
        2. Total followers (home + away)  
        3. School historical avg  
        4. Pixellot camera  
        5. Win percentages  
        6. State rankings  
        """)

    # File upload
    st.header("📤 Upload Game Schedule")
    uploaded = st.file_uploader("Choose a file", type=['xlsx', 'xls', 'csv'])

    if uploaded:
        try:
            df_input = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
            st.success(f"✅ Loaded **{len(df_input):,}** games")

            with st.expander("👀 Preview uploaded data"):
                st.dataframe(df_input.head(10), use_container_width=True)

            # Check for optional columns and warn
            optional_cols = {
                'Staterankhome': 'State Rankings',
                'Hometeamwinpercentage': 'Win Percentages',
                'Followers Home': 'Followers'
            }
            missing_optional = [name for col, name in optional_cols.items() if col not in df_input.columns]
            if missing_optional:
                st.info(f"ℹ️ **Optional columns not found** — {', '.join(missing_optional)}. Predictions will use defaults. Add these columns for better accuracy.")

            with st.spinner('🔮 Generating predictions...'):
                df_out = engineer_and_predict(df_input, model, feature_names, df_hist, home_avgs, away_avgs)

            st.success("✅ Predictions complete!")

            # ── Summary metrics ──────────────────────────────────────
            st.header("📊 Summary")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Games",         f"{len(df_out):,}")
            c2.metric("Total Predicted Subs", f"{df_out['Predicted_Subscriptions'].sum():,.0f}")
            c3.metric("Average / Game",       f"{df_out['Predicted_Subscriptions'].mean():.1f}")
            c4.metric("Median / Game",        f"{df_out['Predicted_Subscriptions'].median():.1f}")
            c5.metric("Highest Game",         f"{df_out['Predicted_Subscriptions'].max():.0f}")

            # ── Tabs ─────────────────────────────────────────────────
            st.header("📈 Analysis")
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Distribution", "Top Games", "By Game Type", "By State", "By Month"])

            with tab1:
                fig = px.histogram(df_out, x='Predicted_Subscriptions', nbins=60,
                                   title='Distribution of Predicted Subscriptions',
                                   color_discrete_sequence=['#1f77b4'])
                fig.update_layout(height=380, xaxis_range=[0, df_out['Predicted_Subscriptions'].quantile(0.97)])
                st.plotly_chart(fig, use_container_width=True)

                bins   = [0, 3, 5, 8, 12, 20, 50, 2000]
                labels = ['1-3', '4-5', '6-8', '9-12', '13-20', '21-50', '50+']
                df_out['Range'] = pd.cut(df_out['Predicted_Subscriptions'], bins=bins, labels=labels)
                range_df = df_out['Range'].value_counts().sort_index().reset_index()
                range_df.columns = ['Range', 'Games']
                range_df['%'] = (range_df['Games'] / len(df_out) * 100).round(1)
                st.dataframe(range_df, use_container_width=True, hide_index=True)

            with tab2:
                home_col = 'Home Slug' if 'Home Slug' in df_out.columns else 'Home School'
                away_col = 'Away Slug' if 'Away Slug' in df_out.columns else 'Away School'
                time_col = 'Start Time Et' if 'Start Time Et' in df_out.columns else 'Start Time'
                top20 = df_out.nlargest(20, 'Predicted_Subscriptions')[
                    [time_col, home_col, away_col, 'State Code', 'Game Type',
                     'Predicted_Subscriptions', 'home_school_hist_avg', 'away_school_hist_avg']
                ].copy()
                top20[time_col] = pd.to_datetime(top20[time_col]).dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(top20.style.format({'Predicted_Subscriptions': '{:.0f}',
                                                  'home_school_hist_avg': '{:.1f}',
                                                  'away_school_hist_avg': '{:.1f}'}),
                             use_container_width=True, hide_index=True)

            with tab3:
                if 'Game Type' in df_out.columns:
                    gt_stats = df_out.groupby('Game Type').agg(
                        Games=('Predicted_Subscriptions', 'count'),
                        Total_Subs=('Predicted_Subscriptions', 'sum'),
                        Avg_Subs=('Predicted_Subscriptions', 'mean')
                    ).round(1).sort_values('Avg_Subs', ascending=False)
                    fig = px.bar(gt_stats.reset_index(), x='Game Type', y='Avg_Subs',
                                 title='Average Predicted Subs by Game Type',
                                 color='Avg_Subs', color_continuous_scale='RdYlGn')
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(gt_stats, use_container_width=True)

            with tab4:
                state_stats = df_out.groupby('State Code').agg(
                    Games=('Predicted_Subscriptions', 'count'),
                    Total_Subs=('Predicted_Subscriptions', 'sum'),
                    Avg_Subs=('Predicted_Subscriptions', 'mean')
                ).round(1).sort_values('Total_Subs', ascending=False).head(15)
                fig = px.bar(state_stats.reset_index(), x='State Code', y='Total_Subs',
                             title='Top 15 States — Total Predicted Subscriptions',
                             color='Avg_Subs', color_continuous_scale='Viridis')
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(state_stats, use_container_width=True)

            with tab5:
                df_out['Month'] = pd.to_datetime(df_out.get('Start Time Et', df_out.get('Start Time'))).dt.to_period('M')
                monthly = df_out.groupby('Month').agg(
                    Games=('Predicted_Subscriptions', 'count'),
                    Total_Subs=('Predicted_Subscriptions', 'sum'),
                    Avg_Subs=('Predicted_Subscriptions', 'mean')
                ).round(1)
                monthly.index = monthly.index.astype(str)
                fig = px.bar(monthly.reset_index(), x='Month', y='Total_Subs',
                             title='Monthly Subscription Forecast',
                             color='Avg_Subs', color_continuous_scale='RdYlGn')
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(monthly, use_container_width=True)

            # ── Downloads ────────────────────────────────────────────
            st.header("💾 Download Results")

            output_cols = [c for c in [
                'Key', time_col if time_col in df_out.columns else 'Start Time',
                'Sport', 'Level', 'Game Type', 'State Code',
                home_col, away_col,
                'Staterankhome', 'Staterankaway',
                'Hometeamwinpercentage', 'Awayteamwinpercentage',
                'Followers Home', 'Followers Away',
                'Distance Between Schools', 'Weekofsy',
                'Predicted_Subscriptions',
                'home_school_hist_avg', 'away_school_hist_avg'
            ] if c in df_out.columns]

            df_dl = df_out[output_cols].copy()

            col1, col2 = st.columns(2)
            with col1:
                buf = BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    df_dl.to_excel(writer, index=False, sheet_name='Predictions')
                st.download_button("📥 Download Excel", buf.getvalue(),
                                   file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with col2:
                st.download_button("📥 Download CSV", df_dl.to_csv(index=False),
                                   file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

    else:
        st.info("👆 Upload a game schedule file to get started.")
        with st.expander("📋 See example file format"):
            st.dataframe(pd.DataFrame({
                'Start Time Et': ['2026-09-04 19:00', '2026-10-09 20:00'],
                'State Code': ['AL', 'TX'],
                'Home Slug': ['school-a-city-al', 'school-b-city-tx'],
                'Away Slug': ['school-c-city-al', 'school-d-city-tx'],
                'Game Type': ['regular season', 'playoffs'],
                'Weekofsy': [3, 8],
                'Staterankhome': [12, 45],
                'Staterankaway': [28, 67],
                'Hometeamwinpercentage': [0.80, 0.65],
                'Awayteamwinpercentage': [0.70, 0.55],
                'Followers Home': [850, 620],
                'Followers Away': [720, 490],
                'Pixellot': ['Yes', 'No'],
                'Distance Between Schools': [18.5, 35.0]
            }), use_container_width=True)


if __name__ == "__main__":
    main()
