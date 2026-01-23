# 🎥 YouTube Sentiment Analysis

An interactive web application that examines **sentiment trends** in YouTube comments using "natural language processing" (NLP).

Users can paste a YouTube video URL (or ID), fetch comments via the YouTube Data API, and visualize sentiment distribution and keywords in real time.

Live App Demo: https://youtube-sentiment-analysis-92ve9tquli8hnjougnwq3f.streamlit.app/


## 🚀 Features

- 🔍 Fetches YouTube comments using the **YouTube Data API v3**
- 🧠 Performs sentiment analysis using **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
- 📊 Visualizes sentiment distribution (Positive / Neutral / Negative)
- ☁️ Generates a **word cloud** from comment text
- 🖥️ Interactive **Streamlit** web interface
- 🔐 Secure API key handling using **environment variables**


## 🛠️ Tech Stack

- **Python**
- **Streamlit** — Interactive web dashboard
- **YouTube Data API v3** — Comment retrieval
- **VADER Sentiment Analyzer** — NLP sentiment scoring
- **Pandas** — Data processing
- **Plotly** — Interactive charts
- **Matplotlib + WordCloud** — Text visualization


## 📂 Project Structure

```
youtube-sentiment/
├── app.py                      # Main Streamlit application
├── data/
│   └── youtube_comments.csv    # Generated dataset
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```


## ⚙️ Installation & Steup

#### 1️⃣ Clone the repository
```
git clone https://github.com/sgnesh123/youtube-sentiment-analysis.git
cd youtube-sentiment-analysis
```

#### 2️⃣ Install dependencies
```
pip install -r requirements.txt
```

#### 3️⃣ Set up your YouTube API key

Create an environment variable named `YOUTUBE_API_KEY`.

- macOS / Linux:
  ```
  export YOUTUBE_API_KEY = "your_api_key_here"
  ```

- Windows (PowerShell):
  ```
  setx YOUTUBE_API_KEY "your_api_key_here"
  ```

⚠️ Never commit API keys to GitHub.


## ▶️ Running the App

From the project directory:
```
streamlit run app.py
```

Then open your browser at:
```
http://localhost:8501
```


## 📊 Example Outputs

- Sentiment classification (Positive / Neutral / Negative)
- Sentiment distribution bar charts
- Word cloud generated from YouTube comments
- Downloadable CSV of analyzed comments


## 🔒 API Usage Notes

- Uses **YouTube Data API v3**, which is **free** within quota limits
- Each request consumes quota units set by Google Cloud
- API key restrictions are recommended for security


## 📈 Why This Project Matters

This project demonstrates:

- Real-world API integration
- NLP and sentiment analysis
- Data cleaning and transformation
- Interactive data visualization
- Full-stack Python app development
- Secure credential management
- Cloud deployment (Streamlit Cloud)


## 📌 Future Improvements:

- Topic modeling
- Time-based sentiment trends
- Comment filtering (by likes or replies)
- Multi-language sentiment support
