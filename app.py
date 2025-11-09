import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load pre-trained objects ---
scaler = joblib.load("scaler.pkl")
with open("selected_features.pkl", "rb") as f:
    FEATURE_ORDER = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

TARGET = "stress_flag"

# --- Feature preparation ---
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    if "hours_on_TikTok" in df.columns and "screen_time_hours" in df.columns:
        df["tiktok_ratio"] = df["hours_on_TikTok"] / df["screen_time_hours"].replace(0, np.nan)
        df["tiktok_ratio"] = df["tiktok_ratio"].fillna(0)
    
    if "sleep_hours" in df.columns:
        df["sleep_deficit"] = np.maximum(0, 8 - df["sleep_hours"])
    
    missing_cols = [col for col in FEATURE_ORDER if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for prediction: {missing_cols}")
    
    df_ordered = df[FEATURE_ORDER]
    df_scaled = pd.DataFrame(scaler.transform(df_ordered), columns=FEATURE_ORDER)
    return df_scaled

# --- Advanced Recommendations with icons ---
def advanced_recommendation(df_scaled: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    df_result = pd.DataFrame()
    preds = model.predict(df_scaled)
    pred_probs = model.predict_proba(df_scaled)[:,1]
    
    df_result[TARGET] = np.where(preds == 1, "⚠️ Stressed", "✅ Not Stressed")
    df_result["probability"] = pred_probs
    
    recommendations = []
    for idx, row in original_df.iterrows():
        recs = []
        if preds[idx] == 1:
            recs.append("Reduce overall screen time.")
            if row.get("hours_on_TikTok", 0) / max(row.get("screen_time_hours", 1), 1) > 0.5:
                recs.append("Limit TikTok usage; it dominates screen time.")
            if row.get("sleep_hours", 0) < 7:
                recs.append("Increase sleep hours to meet healthy recommendations.")
            if row.get("mood_score", 5) < 5:
                recs.append("Engage in mood-lifting activities (exercise, hobbies, social time).")
            recs.append("Practice mindfulness or meditation for stress relief.")
            recs.append("Parents and schools can monitor digital behavior for early intervention.")
        else:
            recs.append("Maintain healthy habits: balanced sleep, moderate screen time, positive mood.")
        recommendations.append(" ".join(recs))
    
    df_result["recommendation"] = recommendations
    return df_result

# --- Streamlit App ---
st.set_page_config(page_title="Adolescent Stress Dashboard", layout="wide")
st.title("Adolescent Stress Prediction Dashboard")

menu = ["Single User", "Batch CSV Upload"]
choice = st.sidebar.selectbox("Mode", menu)

# --- Color function ---
def color_text(status):
    if "Stressed" in status:
        return f"<p style='color:red; font-size:20px;'><b>{status}</b></p>"
    else:
        return f"<p style='color:orange; font-size:20px;'><b>{status}</b></p>"

# --- Session history ---
if "history" not in st.session_state:
    st.session_state.history = []

if choice == "Single User":
    st.header("Single User Input")
    
    # Input fields
    screen_time_hours = st.number_input("Screen Time (hours)", min_value=0.0, step=0.1)
    hours_on_TikTok = st.number_input("Hours on TikTok", min_value=0.0, step=0.1)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, step=0.1)
    stress_level = st.number_input("Stress Level (1-10)", min_value=0, max_value=10, step=1)
    mood_score = st.number_input("Mood Score (1-10)", min_value=1, max_value=10, step=1)
    
    user_df = pd.DataFrame([{
        "screen_time_hours": screen_time_hours,
        "hours_on_TikTok": hours_on_TikTok,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "mood_score": mood_score
    }])
    
    if st.button("Predict Stress & Recommend"):
        try:
            X_user = prepare_features(user_df)
            result = advanced_recommendation(X_user, user_df)
            
            # Display prediction and probability
            st.subheader("Prediction Result")
            st.markdown(color_text(result.loc[0, TARGET]), unsafe_allow_html=True)
            st.write(f"**Probability of Being Stressed:** {result.loc[0, 'probability']*100:.1f}%")
            st.write(f"**Recommendation:** {result.loc[0, 'recommendation']}")
            
            # Session history
            st.session_state.history.append(result.loc[0].to_dict())
            st.subheader("Session History")
            st.table(pd.DataFrame(st.session_state.history))
            
            # Feature visualization
            st.subheader("Input Metrics Overview")
            features = ["screen_time_hours", "hours_on_TikTok", "sleep_hours", "mood_score"]
            values = [screen_time_hours, hours_on_TikTok, sleep_hours, mood_score]
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x=features, y=values, palette="viridis", ax=ax)
            ax.set_ylim(0, max(values)+2)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(str(e))

else:
    st.header("Batch CSV Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        try:
            df_batch = pd.read_csv(uploaded_file)
            X_batch = prepare_features(df_batch)
            result_batch = advanced_recommendation(X_batch, df_batch)
            
            st.subheader("Batch Predictions")
            for i in range(len(result_batch)):
                st.markdown(f"**Entry {i+1}:**")
                st.markdown(color_text(result_batch.loc[i, TARGET]), unsafe_allow_html=True)
                st.write(f"**Probability:** {result_batch.loc[i, 'probability']*100:.1f}%")
                st.write(f"**Recommendation:** {result_batch.loc[i, 'recommendation']}")
                st.write("---")
            
            # Batch summary chart
            st.subheader("Batch Summary")
            summary_counts = result_batch[TARGET].value_counts()
            st.bar_chart(summary_counts)
            
            # Download button
            csv = result_batch.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", data=csv, file_name="stress_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(str(e))
