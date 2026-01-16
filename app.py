"""

*YouTube Sentiment Analyzer*

A Streamlit web application that fetches YouTube comments and performs sentiment analysis
using a pre-trained model called "VADER". 

VADER = Valence Aware Dictionary and sEntiment Reasoner

Features:
- Analyze sentiment of custom text input
- Fetch and analyze YouTube video comments via 'YouTube Data API'
- Visualize sentiment distribution with interactive charts
- Produce word clouds from aggregated comments
- Export results to CSV

"""


import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title = "YouTube Sentiment Analyzer",
    page_icon = "📺",
    layout = "wide"
)


# --------------------------------------------------
# Initialize sentiment analyzer
# --------------------------------------------------

analyzer = SentimentIntensityAnalyzer()


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>
    /* Sentiment category cards with color-coded backgrounds */
    .positive { 
        background-color: #d4edda; 
        padding: 20px; 
        border-radius: 10px; 
        color: #155724; 
    }
    
    .negative { 
        background-color: #f8d7da; 
        padding: 20px; 
        border-radius: 10px; 
        color: #721c24; 
    }
    
    .neutral { 
        background-color: #fff3cd; 
        padding: 20px; 
        border-radius: 10px; 
        color: #856404; 
    }
    
    /* Ensure text within sentiment cards is readable */
    .positive h2, .positive p, 
    .negative h2, .negative p, 
    .neutral h2, .neutral p {
        color: #000000;
    }
</style>
""", unsafe_allow_html = True)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def analyze_sentiment(text: str) -> tuple:
    """
    Analyze sentiment using VADER. Returns a categorical label and compound score, 
    which is treated as a sentiment strength indicator (not a probability).
    """
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    # Classify sentiment based on compound score thresholds
    if compound > 0.05:
        sentiment = "Positive"
    elif compound < -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, compound, scores


def get_youtube_client():
    """
    Initialize the 'YouTube Data API' client using an environment variable.
    Credentials are intentionally not hardcoded to avoid leaking secrets.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    
    if not api_key:
        return None

    return build("youtube", "v3", developerKey = api_key)


def extract_video_id(video_url: str):
    """
    Extract the video ID from a standard YouTube URL.
    Falls back to the raw input if parsing fails.
    """
    if "v=" in video_url:
        # Extract ID from URL parameter and remove any additional query parameters
        return video_url.split("v=")[-1].split("&")[0]
        
    # If input doesn't contain 'v=', assume it's already a video ID
    return video_url.strip()


def fetch_youtube_comments(video_id: str, limit: int = 100):
    """
    Fetch comments from a YouTube video and apply sentiment analysis.
    """
    youtube = get_youtube_client()
    
    if youtube is None:
        st.error("YouTube API key not found in environment variables.")
        return None

    comments = []

    # Initial API request for comment threads
    request = youtube.commentThreads().list(
        part = "snippet",
        videoId = video_id,
        maxResults = 100,  # Maximum allowed by API per request
        textFormat = "plainText"
    )

    # Paginate through results until we reach the desired limit
    while request and len(comments) < limit:
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_text": snippet["textDisplay"],
                "like_count": snippet["likeCount"],
                "published_at": snippet["publishedAt"]
            })

            if len(comments) >= limit:
                break

        # Get next page of results, if available
        request = youtube.commentThreads().list_next(request, response)

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(comments)

    if df.empty:
        return df

    # Apply sentiment analysis to each comment
    # This creates a series of tuples (sentiment, compound, scores)
    sentiments = df["comment_text"].apply(analyze_sentiment)

    # Extract sentiment label and compound score from tuples
    df["sentiment"] = sentiments.apply(lambda x: x[0])
    df["compound_score"] = sentiments.apply(lambda x: x[1])

    return df


def load_data():
    """
    Load previously fetched YouTube comments from CSV.
    Returns 'None' if no dataset exists.
    """
    try:
        return pd.read_csv("data/youtube_comments.csv")
    except FileNotFoundError:
        return None


def save_data(df: pd.DataFrame):
    """
    Save comment data to CSV.
    Overwrites existing data for reproducibility.
    """
    os.makedirs("data", exist_ok = True)
    df.to_csv("data/youtube_comments.csv", index = False)


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------

st.sidebar.title("📺 YouTube Sentiment Analyzer")

# Radio buttons provide clear navigation between app sections
page = st.sidebar.radio("Navigation", ["Analyze Text", "Dataset View", "Fetch New Comments"])


# ==================================================
# PAGE 1 — ANALYZE TEXT
# ==================================================

if page == "Analyze Text":
    st.title("🎯 Analyze Text Sentiment")

    user_text = st.text_area(
        "What text should I analyze?",
        height = 150,
        placeholder = "Paste a comment, review, or any short text..."
    )

    if st.button("Analyze", type = "primary"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            # Perform sentiment analysis
            sentiment, compound, scores = analyze_sentiment(user_text)

            # Map sentiment to its corresponding styling and icons
            css_class = sentiment.lower()
            icon = "✅" if sentiment == "Positive" else "❌" if sentiment == "Negative" else "➖"

            # Display sentiment result with color-coded styling
            st.markdown(
                f'<div class="{css_class}">'
                f'<h2>{icon} {sentiment}</h2>'
                f'<p>Compound Score: {compound:.3f}</p>'
                f'</div>',
                unsafe_allow_html = True
            )

            # Create a bar chart showing the breakdown of pos/neu/neg components
            score_df = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Score": [scores["pos"], scores["neu"], scores["neg"]]
            })
            fig = px.bar(score_df, x = "Sentiment", y = "Score")
            fig.update_layout(showlegend = False, height = 400)
            st.plotly_chart(fig, use_container_width = True)


# ==================================================
# PAGE 2 — DATASET VIEW
# ==================================================

elif page == "Dataset View":
    st.title("📊 Dataset Analysis")

    df = load_data()

    if df is None or df.empty:
        st.info("No data available. Fetch some comments first.")
    else:
        total = len(df)
        
        # Calculate percentage distribution for each sentiment category
        counts = df["sentiment"].value_counts(normalize = True) * 100

        # Display key metrics, in columns, for quick overview
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Comments", total)
        col2.metric("% Positive", f"{counts.get('Positive', 0):.1f}%")
        col3.metric("% Negative", f"{counts.get('Negative', 0):.1f}%")
        col4.metric("% Neutral", f"{counts.get('Neutral', 0):.1f}%")

        st.divider()

        # Create a two-column layout for visualizations
        col_left, col_right = st.columns(2)

        with col_left:
            # Bar chart shows the frequency of each sentiment
            fig_bar = px.bar(
                df["sentiment"].value_counts(),
                labels = {"index": "Sentiment", "value": "Count"}
            )
            st.plotly_chart(fig_bar, use_container_width = True)

        with col_right:
            # # Pie chart illustrates proportional distribution
            fig_pie = px.pie(
                names = df["sentiment"].value_counts().index,
                values = df["sentiment"].value_counts().values
            )
            st.plotly_chart(fig_pie, use_container_width = True)

        # Word cloud visualization
        st.subheader("Word Cloud")

        # Combine all comments into a single text blob
        text_blob = " ".join(df["comment_text"].dropna().astype(str))
        
        if len(text_blob.strip()) < 50:
            st.info("Not enough text to generate a word cloud.")
        else:
            try:
                # Generate word cloud with custom parameters
                wc = WordCloud(
                    width = 1200,
                    height = 400,
                    background_color = "white",
                    min_font_size = 10,
                    collocations = False  # Avoid repeating word pairs
                ).generate(text_blob)

                if not wc.words_:
                    st.info("Not enough meaningful words to generate a word cloud.")
                else:
                    # Display word cloud using Matplotlib
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.imshow(wc.to_image())
                    ax.axis("off")
                    st.pyplot(fig)
            except (ValueError, Exception) as e:
                st.info(f"Unable to generate word cloud: {str(e)}")

        # Display sample of comments in a table
        st.subheader("Sample Comments")
        
        st.dataframe(
            df[["comment_text", "like_count", "sentiment", "compound_score"]].head(10),
            use_container_width = True
        )


# ==================================================
# PAGE 3 — SCRAPE NEW DATA
# ==================================================

else:
    st.title("💬 Fetch YouTube Comments")

    col1, col2 = st.columns(2)
    
    with col1:
        video_input = st.text_input("YouTube Video (link or ID):")
    with col2:
        num_comments = st.number_input(
            "Number of comments:", 
            min_value = 10, 
            max_value = 1000, 
            value = 100, 
            step = 10
        )

    if st.button("🚀 Fetch Comments", type = "primary"):
        video_id = extract_video_id(video_input)

        # Show a loading spinner while fetching comments
        with st.spinner("Fetching comments from YouTube..."):
            df = fetch_youtube_comments(video_id, num_comments)

            if df is None or df.empty:
                st.error("Unable to load comments.")
            else:
                # Save fetched data to CSV
                save_data(df)
                st.success(f"Fetched {len(df)} comments.")

                # Display fetched comments
                st.dataframe(
                    df[["comment_text", "like_count", "sentiment", "compound_score"]],
                    use_container_width = True
                )

                # Provide download button for CSV export
                st.download_button(
                    "📥 Download CSV",
                    df.to_csv(index = False),
                    file_name = "youtube_comments.csv",
                    mime = "text/csv"
                )


# ==================================================
# SIDEBAR FOOTER
# ==================================================

st.sidebar.divider()
st.sidebar.caption("Built with Streamlit, YouTube Data API, and VADER")

