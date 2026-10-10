import re
from io import BytesIO
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="FeedbackIQ | Customer Intelligence", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(120,130,160,.10); padding: 14px; border-radius: 12px;}
.hero {padding: 1.2rem 1.4rem; border-radius: 16px; background: linear-gradient(120deg,#18233f,#334b7d); color:white; margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2rem;} .hero p {margin:.35rem 0 0 0; opacity:.9;}
.small-note {font-size:.85rem; opacity:.8;}
</style>
""", unsafe_allow_html=True)

TOPIC_KEYWORDS = {
    "Delivery": ["delivery", "deliver", "late", "shipping", "courier", "arrived", "tracking"],
    "Product quality": ["quality", "broken", "damaged", "defect", "faulty", "poor quality", "durable"],
    "Customer support": ["support", "agent", "staff", "representative", "service team", "rude", "response"],
    "Pricing & value": ["price", "pricing", "expensive", "cost", "value", "overpriced", "discount"],
    "Refunds & returns": ["refund", "return", "replacement", "money back", "cancel", "cancellation"],
    "App & website": ["app", "website", "login", "crash", "bug", "loading", "checkout", "interface"],
    "Product features": ["feature", "battery", "performance", "design", "size", "function", "works"],
}

POSITIVE_WORDS = {
    "good","great","excellent","amazing","love","loved","fast","helpful","friendly","easy",
    "perfect","happy","satisfied","wonderful","reliable","recommend","best","smooth","quick",
    "impressive","comfortable","beautiful","resolved","thank","awesome","fantastic"
}
NEGATIVE_WORDS = {
    "bad","terrible","awful","hate","hated","slow","late","broken","damaged","poor","worst",
    "angry","disappointed","disappointing","refund","unhelpful","rude","defect","faulty",
    "expensive","issue","problem","crash","failed","failure","unacceptable","missing","never",
    "waste","unhappy","difficult","frustrating","delay","delayed","wrong"
}
NEGATIONS = {"not","never","no","isn't","wasn't","don't","didn't","hardly"}

def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()

def sentiment_rule(text):
    """Transparent local baseline. Replace/augment with a pretrained model for stronger NLP."""
    words = re.findall(r"[a-z']+", text.lower())
    pos = neg = 0
    for i, word in enumerate(words):
        flip = i > 0 and words[i-1] in NEGATIONS
        if word in POSITIVE_WORDS:
            if flip: neg += 1
            else: pos += 1
        if word in NEGATIVE_WORDS:
            if flip: pos += 1
            else: neg += 1
    if pos == neg:
        label = "Neutral"
    elif pos > neg:
        label = "Positive"
    else:
        label = "Negative"
    total = pos + neg
    confidence = 0.50 if total == 0 else min(0.97, 0.52 + abs(pos-neg) / (2*total + 1))
    return label, round(confidence, 2)

@st.cache_resource(show_spinner=False)
def load_transformer():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text, use_transformer=False, model=None):
    if use_transformer and model is not None and text:
        try:
            result = model(text[:1500])[0]
            label = "Positive" if result["label"].upper() == "POSITIVE" else "Negative"
            return label, round(float(result["score"]), 2)
        except Exception:
            pass
    return sentiment_rule(text)

def get_topic(text):
    low = text.lower()
    scores = {topic: sum(1 for keyword in kws if keyword in low) for topic, kws in TOPIC_KEYWORDS.items()}
    top = max(scores, key=scores.get)
    return top if scores[top] else "General feedback"

def get_priority(text, sentiment, topic):
    low = text.lower()
    urgent_terms = ["fraud", "unsafe", "safety", "injury", "dangerous", "charged twice", "stolen", "data leak", "security"]
    high_terms = ["refund", "broken", "damaged", "never arrived", "unacceptable", "failed", "angry", "cancel"]
    if any(term in low for term in urgent_terms):
        return "Urgent"
    if sentiment == "Negative" and any(term in low for term in high_terms):
        return "High"
    if sentiment == "Negative":
        return "Medium"
    return "Low"

def make_recommendations(df):
    recs = []
    negatives = df[df["Sentiment"] == "Negative"]
    topic_counts = negatives["Topic"].value_counts()
    suggestions = {
        "Delivery": "Review courier performance, delivery promises, and proactive delay notifications.",
        "Product quality": "Audit the highest-complaint product batches and strengthen quality checks.",
        "Customer support": "Review support response times and coach agents on resolution quality.",
        "Pricing & value": "Compare pricing with alternatives and explain product value more clearly.",
        "Refunds & returns": "Simplify refund steps and publish clear processing timelines.",
        "App & website": "Reproduce reported bugs, prioritize stability, and test checkout/login flows.",
        "Product features": "Group feature requests and validate the most frequent needs with customers.",
        "General feedback": "Review representative comments and tag them into more specific themes."
    }
    for topic, count in topic_counts.items():
        recs.append({"Priority": "High" if count >= max(3, len(negatives)*0.25) else "Medium",
                     "Issue": topic, "Negative mentions": int(count),
                     "Suggested action": suggestions.get(topic, suggestions["General feedback"])})
    if not recs:
        recs.append({"Priority":"Monitor","Issue":"No negative feedback detected",
                     "Negative mentions":0,"Suggested action":"Continue monitoring new feedback and validate the sentiment predictions."})
    return pd.DataFrame(recs).sort_values("Negative mentions", ascending=False)

def process_frame(raw, use_transformer=False, model=None):
    text_col = next((c for c in raw.columns if c.lower().strip() in ["feedback","review","comment","text","customer_feedback","message"]), None)
    if text_col is None:
        text_col = raw.columns[0]
    df = raw.copy()
    df["Feedback"] = df[text_col].map(clean_text)
    df = df[df["Feedback"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["Feedback"]).reset_index(drop=True)
    if df.empty:
        return df
    sentiments = [analyze_sentiment(t, use_transformer, model) for t in df["Feedback"]]
    df["Sentiment"] = [s[0] for s in sentiments]
    df["Confidence"] = [s[1] for s in sentiments]
    df["Topic"] = df["Feedback"].map(get_topic)
    df["Priority"] = [get_priority(t, s, topic) for t,s,topic in zip(df["Feedback"],df["Sentiment"],df["Topic"])]
    return df

def sample_data():
    return pd.DataFrame({"feedback": [
        "The delivery arrived two days late and tracking never updated.",
        "Excellent product quality and very easy to use. I love it!",
        "Customer support was rude and did not resolve my issue.",
        "The price is too expensive for the features offered.",
        "My item arrived broken and I need a refund immediately.",
        "The mobile app keeps crashing during checkout.",
        "Fast delivery and helpful support team. Great experience.",
        "I was charged twice and need this fixed today.",
        "The product design is beautiful and the battery lasts long.",
        "The return process is confusing and my refund is delayed.",
        "The website is easy to navigate.",
        "This product caused a safety concern; please contact me urgently.",
        "Average experience, nothing special to report.",
        "The replacement arrived quickly and the team was helpful.",
        "Poor quality, the item stopped working after one day.",
        "The price is fair and the product works as expected.",
        "I cannot log in to the app after the latest update.",
        "My order never arrived and nobody answered my messages.",
        "The new feature is useful, but setup was difficult.",
        "Thank you, the support agent resolved my problem quickly."
    ]})

st.markdown('<div class="hero"><h1>🧠 FeedbackIQ</h1><p>AI Customer Feedback Intelligence Platform · Turn customer voice into clear action</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Workspace")
    st.caption("Upload feedback or use the included demo dataset.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_demo = st.button("Load demo feedback", use_container_width=True)
    st.divider()
    st.subheader("AI settings")
    use_transformer = st.checkbox("Use pretrained Transformer model", value=False,
        help="Optional: downloads model files on first use and requires internet access and transformers/torch installed.")
    st.caption("If disabled or unavailable, a transparent local sentiment baseline is used.")
    st.divider()
    st.caption("Privacy: uploaded data is processed by this app session. Avoid uploading sensitive personal information.")

if "raw_data" not in st.session_state:
    st.session_state.raw_data = sample_data()
if use_demo:
    st.session_state.raw_data = sample_data()
if uploaded is not None:
    try:
        st.session_state.raw_data = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

raw = st.session_state.raw_data
model = None
if use_transformer:
    try:
        with st.spinner("Loading pretrained sentiment model (first run may take a while)..."):
            model = load_transformer()
        st.success("Pretrained model loaded.")
    except Exception as e:
        st.warning(f"Transformer unavailable; using local baseline instead. Details: {e}")

if raw is None or raw.empty:
    st.info("Upload a CSV or load the demo feedback to begin.")
    st.stop()

with st.spinner("Analysing customer feedback..."):
    result = process_frame(raw, use_transformer, model)

if result.empty:
    st.warning("No usable feedback rows found. Ensure your CSV has a text column such as feedback, review, comment, or text.")
    st.stop()

st.caption(f"Analysed {len(result):,} unique feedback items · Sentiment engine: {'Pretrained Transformer' if use_transformer and model else 'Local baseline'}")

# Filters
with st.expander("🔎 Filter results", expanded=False):
    c1,c2,c3 = st.columns(3)
    sentiments_available = ["All"] + sorted(result["Sentiment"].unique().tolist())
    topics_available = ["All"] + sorted(result["Topic"].unique().tolist())
    priorities_available = ["All","Urgent","High","Medium","Low"]
    sentiment_filter = c1.selectbox("Sentiment", sentiments_available)
    topic_filter = c2.selectbox("Topic", topics_available)
    priority_filter = c3.selectbox("Priority", priorities_available)
filtered = result.copy()
if sentiment_filter != "All": filtered = filtered[filtered["Sentiment"] == sentiment_filter]
if topic_filter != "All": filtered = filtered[filtered["Topic"] == topic_filter]
if priority_filter != "All": filtered = filtered[filtered["Priority"] == priority_filter]

total = len(result)
negative_pct = (result["Sentiment"].eq("Negative").mean()*100) if total else 0
positive_pct = (result["Sentiment"].eq("Positive").mean()*100) if total else 0
urgent_count = int(result["Priority"].eq("Urgent").sum())
avg_conf = float(result["Confidence"].mean()) if total else 0
m1,m2,m3,m4 = st.columns(4)
m1.metric("Total feedback", f"{total:,}")
m2.metric("Positive sentiment", f"{positive_pct:.0f}%")
m3.metric("Negative sentiment", f"{negative_pct:.0f}%")
m4.metric("Urgent flags", f"{urgent_count:,}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview","🚨 Issues & actions","💬 Feedback explorer","⬇️ Export"])

with tab1:
    left,right = st.columns([1,1])
    with left:
        sentiment_counts = result["Sentiment"].value_counts().reindex(["Positive","Neutral","Negative"], fill_value=0).reset_index()
        sentiment_counts.columns = ["Sentiment","Count"]
        fig = px.pie(sentiment_counts, names="Sentiment", values="Count", hole=.58,
                     title="Sentiment distribution", color="Sentiment",
                     color_discrete_map={"Positive":"#2CA58D","Neutral":"#F4B942","Negative":"#E76F51"})
        fig.update_layout(margin=dict(t=55,b=10,l=10,r=10), legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        topic_counts = result["Topic"].value_counts().reset_index()
        topic_counts.columns = ["Topic","Feedback count"]
        fig2 = px.bar(topic_counts, x="Feedback count", y="Topic", orientation="h",
                      title="Most discussed topics", color="Feedback count", color_continuous_scale="Blues")
        fig2.update_layout(yaxis={"categoryorder":"total ascending"}, margin=dict(t=55,b=10,l=10,r=10), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Priority distribution")
    pcounts = result["Priority"].value_counts().reindex(["Urgent","High","Medium","Low"], fill_value=0).reset_index()
    pcounts.columns = ["Priority","Count"]
    st.plotly_chart(px.bar(pcounts, x="Priority", y="Count", color="Priority",
        color_discrete_map={"Urgent":"#8E2A2A","High":"#E76F51","Medium":"#F4B942","Low":"#2CA58D"}), use_container_width=True)
    st.caption(f"Average model/baseline confidence score: {avg_conf:.0%}. Confidence is not a guarantee of correctness.")

with tab2:
    st.subheader("Recommended actions from negative feedback")
    recommendations = make_recommendations(result)
    st.dataframe(recommendations, use_container_width=True, hide_index=True)
    st.subheader("Urgent feedback for human review")
    urgent = result[result["Priority"] == "Urgent"][["Feedback","Topic","Sentiment","Priority"]]
    if urgent.empty:
        st.success("No feedback matched the urgent keyword rules in this dataset.")
    else:
        st.dataframe(urgent, use_container_width=True, hide_index=True)
    st.caption("Priority is a heuristic triage aid, not a safety or legal determination. A human should review urgent cases.")

with tab3:
    st.subheader("Explore individual feedback")
    st.write(f"Showing {len(filtered):,} of {len(result):,} records after filters.")
    display_cols = ["Feedback","Sentiment","Confidence","Topic","Priority"]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
    query = st.text_input("Search feedback text", placeholder="Try: refund, delivery, app...")
    if query:
        matches = filtered[filtered["Feedback"].str.contains(query, case=False, na=False)]
        st.write(f"{len(matches)} matching item(s)")
        st.dataframe(matches[display_cols], use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Download analysis")
    export_df = result.copy()
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download analysed feedback (CSV)", data=csv_bytes,
        file_name="feedbackiq_analysis.csv", mime="text/csv", use_container_width=True)
    rec_csv = make_recommendations(result).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download recommended actions (CSV)", data=rec_csv,
        file_name="feedbackiq_recommendations.csv", mime="text/csv", use_container_width=True)
    st.info("The CSV contains the original feedback plus sentiment, confidence, topic, and priority fields.")

st.divider()
st.markdown("**FeedbackIQ prototype** · Built with Python, Streamlit, Pandas and Plotly")
st.markdown('<p class="small-note">Prototype note: the local sentiment baseline is a transparent keyword heuristic, not a trained ML model. For production, validate predictions on labelled domain data, use a suitable multilingual model, add access controls, and review sensitive cases manually.</p>', unsafe_allow_html=True)
