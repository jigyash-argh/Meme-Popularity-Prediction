import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
from textblob import TextBlob

# 1. Load Resources
try:
    model = joblib.load('meme_model_final.pkl')
    tfidf = joblib.load('tfidf_final.pkl')
except FileNotFoundError:
    st.error("🚨 Files missing! Run the notebook first to generate .pkl files.")
    st.stop()

# 2. Define "Cool" Feedback Messages
feedback_low = [
    "😬 Yikes. My grandma gets more likes.",
    "😐 Tough crowd. Maybe delete this later.",
    "📉 Cringe levels are critical.",
    "🐜 Is this a meme for ants?",
    "🛑 Stop. Get some help."
]

feedback_mid = [
    "🤔 Not bad. I exhaled through my nose.",
    "🙂 A respectable effort. Solid.",
    "🤷‍♂️ It's honest work.",
    "📈 Trending... slowly.",
    "👍 Approved by the council."
]

feedback_high = [
    "🔥 TO THE MOON! 🚀",
    "💎 DIAMOND HANDS! This is strictly viral.",
    "🤯 Internet Breaker detected.",
    "👑 You dropped this, King/Queen.",
    "🐐 GOATED meme status achieved."
]

# 3. Prediction Logic with Dynamic Inputs
def get_prediction(caption, subreddit, category, hour, day, virality_factor):
    # A. Numerical Features (Now Dynamic!)
    input_data = pd.DataFrame({
        'caption_length': [len(caption)],
        'word_count': [len(caption.split())],
        'sentiment': [TextBlob(caption).sentiment.polarity],
        'hour_posted': [hour],       # User selected
        'day_of_week': [day],        # User selected
        'brightness': [120],         # Kept static (hard to guess)
        'contrast': [50],            # Kept static
        'num_comments': [0]
    })
    
    # B. Categorical
    cat_df = pd.DataFrame({'category': [category], 'subreddit': [subreddit]})
    input_encoded = pd.get_dummies(cat_df)
    
    # C. Text Features
    tfidf_vector = tfidf.transform([caption])
    tfidf_df_input = pd.DataFrame(tfidf_vector.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_df_input.columns = ['word_' + col for col in tfidf_df_input.columns]
    
    # D. Combine
    final_input = pd.concat([input_data, input_encoded, tfidf_df_input], axis=1)
    
    # E. Align
    model_features = model.feature_names_in_
    final_input = final_input.reindex(columns=model_features, fill_value=0)
    
    # F. Predict & Add "Chaos"
    # We add a random variation +/- 15% based on the "Virality Factor" slider
    # This simulates how luck plays a role in memes
    base_prediction = model.predict(final_input)[0]
    noise = base_prediction * (virality_factor / 100.0) * np.random.choice([-1, 1])
    final_prediction = base_prediction + noise
    
    return max(0, round(final_prediction)) # No negative upvotes

# 4. Streamlit UI
st.set_page_config(page_title="Meme God Predictor", page_icon="🐸", layout="centered")

st.title("🐸 Meme God Predictor 3000")
st.markdown("Will you go **viral** or get **cancelled**? Let the AI decide.")
st.markdown("---")

# --- Inputs Section ---
col1, col2 = st.columns(2)
with col1:
    subreddit = st.selectbox("📍 Subreddit", ['r/funny', 'r/memes', 'r/dankmemes', 'r/ProgrammerHumor', 'r/wholesomememes', 'r/teenagers'])
    day = st.selectbox("📅 Day to Post", options=[0,1,2,3,4,5,6], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])

with col2:
    category = st.selectbox("🏷️ Category", ['relatable', 'coding', 'animal', 'school', 'sports', 'gaming', 'dark_humor'])
    hour = st.slider("⏰ Time Posted (24h)", 0, 23, 14)

st.markdown("###")
caption = st.text_area("✍️ Your Caption", "Me explaining to my teacher why the code broke")

# Hidden "Luck" Factor (Optional for user to play with)
with st.expander("⚙️ Advanced Settings (Tweak the Algorithm)"):
    virality = st.slider("Luck / Chaos Factor (%)", 0, 30, 10, help="The internet is random. How much luck do you have?")

# --- Results Section ---
if st.button("🚀 RATE MY MEME", use_container_width=True):
    if not caption:
        st.warning("Write a caption first, bro.")
    else:
        with st.spinner("Calculating internet clout..."):
            import time
            time.sleep(1) # Fake loading for suspense
            
            score = get_prediction(caption, subreddit, category, hour, day, virality)
            
            # Dynamic Feedback
            if score > 1800:
                msg = random.choice(feedback_high)
                color = "green"
            elif score > 1200:
                msg = random.choice(feedback_mid)
                color = "orange"
            else:
                msg = random.choice(feedback_low)
                color = "red"
            
            st.markdown(f"## Predicted Upvotes: :{color}[{score}]")
            st.info(f"🤖 AI says: **{msg}**")
            
            if score > 2000:
                st.balloons()

st.markdown("---")
st.caption("Powered by Scikit-Learn & Broken Dreams")