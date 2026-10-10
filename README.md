# FeedbackIQ — AI Customer Feedback Intelligence Platform

A solo-developer Streamlit prototype for analysing customer feedback and turning recurring issues into action.

## Features
- Upload a CSV or load a built-in demo dataset
- Sentiment labels: Positive, Neutral, Negative
- Confidence score (interpret as indicative, not guaranteed correctness)
- Topic grouping: Delivery, Product quality, Customer support, Pricing, Refunds, App/website, Product features
- Heuristic priority flags, including urgent keyword flags for human review
- Interactive Plotly charts and filters
- Feedback search and downloadable CSV reports
- Optional pretrained Transformer sentiment model

## Run locally (Windows)
1. Install Python 3.10+.
2. Extract this ZIP to a folder.
3. Open PowerShell in that folder.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
5. Install core packages:
   ```powershell
   python -m pip install -r requirements.txt
   ```
6. Start the app:
   ```powershell
   python -m streamlit run app.py
   ```
7. Browser opens a local address (usually `http://localhost:8501`). Click **Load demo feedback** to reset to sample data.

## Optional pretrained Transformer
Install optional packages:
```powershell
python -m pip install -r requirements-ai.txt
```
Then enable **Use pretrained Transformer model** in the sidebar. On first run, the model must download, so internet access is required. If unavailable, the app falls back to the local baseline.

## CSV format
Recommended column names: `feedback`, `review`, `comment`, `text`, `customer_feedback`, or `message`.
If none is found, the first column is treated as feedback text.
Example:
```csv
feedback
"The delivery was late and tracking did not update."
"Excellent quality; very easy to use."
"The app crashes during checkout."
```

## Architecture
CSV / demo input -> validation and deduplication -> sentiment analysis -> topic extraction -> priority rules -> dashboard and exports.

## Important limitations
- The default sentiment engine is a transparent keyword heuristic; it is not a trained machine-learning model.
- Topic extraction and urgency flags use keyword rules and can miss context, sarcasm, mixed-language feedback, and new topics.
- The optional Transformer model is English-focused and classifies sentiment as positive/negative; neutral is only available in the local baseline.
- Validate on labelled data before using for business decisions. Urgent flags are triage aids and require human review.
- This demo does not include authentication, database persistence, or external review-platform integrations.
