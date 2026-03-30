# ============================================================
#  EduSense Backend — app.py
#  Supports:
#    POST /analyze          → single feedback
#    POST /analyze-csv      → upload CSV file from Google Forms
#    POST /analyze-sheets   → Google Sheets public URL
# ============================================================

import re, io, warnings, os, json, sqlite3
from datetime import datetime
from collections import Counter
warnings.filterwarnings("ignore")

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
if not os.path.isdir(STATIC_DIR):
    STATIC_DIR = BASE_DIR
DB_PATH = os.path.join(BASE_DIR, "edusense.db")
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")

if os.path.isdir(NLTK_DATA_DIR) and NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir=NLTK_DATA_DIR, quiet=True)
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)
    stop_words = set(stopwords.words('english'))

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_type TEXT NOT NULL,
                source_ref TEXT,
                analyzed_column TEXT,
                total INTEGER DEFAULT 0,
                positive_count INTEGER DEFAULT 0,
                negative_count INTEGER DEFAULT 0,
                positive_pct REAL DEFAULT 0,
                negative_pct REAL DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()

init_db()

# ── Training Data ─────────────────────────────────────────────
TRAIN_FEEDBACK = [
    "The teacher explains concepts very clearly",
    "The classes are boring and not interactive",
    "I really enjoyed the course content",
    "The syllabus is too difficult",
    "Excellent teaching and good support",
    "The lectures are confusing and rushed",
    "Very helpful and friendly staff",
    "Poor explanation of topics",
    "The instructor is knowledgeable and engaging",
    "I am struggling to understand the material",
    "Great examples and practical exercises",
    "The assignments are unclear and confusing",
    "I love the way topics are structured",
    "The course moves too fast to follow",
    "Very organized and well-prepared lessons",
    "Feedback from teacher is never helpful",
    "Amazing learning experience overall",
    "The grading is unfair and inconsistent",
    "Professor is patient and always available",
    "Too much content crammed into short time",
    "The course material is well structured and easy to follow",
    "I cannot understand anything being taught",
    "The professor gives great real world examples",
    "Extremely dull and unengaging lectures",
    "I feel very supported and encouraged in this class",
    "No one helps when I have questions",
    "The teaching style is creative and interesting",
    "Assignments are too vague and poorly explained",
    "Loved every session it was always interesting",
    "The content is outdated and irrelevant",
]
TRAIN_LABELS = [
    1,0,1,0,1,0,1,0,
    1,0,1,0,1,0,1,0,
    1,0,1,0,1,0,1,0,
    1,0,1,0,1,0,
]

# ── Preprocessing ─────────────────────────────────────────────
stemmer    = PorterStemmer()

def preprocess(text):
    text  = str(text).lower()
    text  = re.sub(r'http\S+', '', text)
    text  = re.sub(r'[^a-z\s]', ' ', text)
    text  = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

# ── Train Model ───────────────────────────────────────────────
cleaned_train = [preprocess(t) for t in TRAIN_FEEDBACK]
vectorizer    = TfidfVectorizer(max_features=500, ngram_range=(1,2), min_df=1)
X_train       = vectorizer.fit_transform(cleaned_train)
model         = LogisticRegression(max_iter=1000)
model.fit(X_train, TRAIN_LABELS)
print("  Model trained and ready!")

# ── Helper: analyze one text ──────────────────────────────────
def analyze_text(text):
    cleaned    = preprocess(text)
    vector     = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    proba      = model.predict_proba(vector)[0]
    pos_prob   = float(proba[1])
    neg_prob   = float(proba[0])

    # Convert binary probability into 3 tiers:
    # Bad (negative), Good (moderately positive), Excellent (highly positive).
    bad_score       = round(neg_prob * 100, 1)
    excellent_score = round(max(0.0, (pos_prob - 0.55) / 0.45) * 100, 1)
    good_score      = round(max(0.0, (pos_prob * 100) - excellent_score), 1)
    confidence = round(float(max(proba)) * 100, 1)

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores  = vector.toarray()[0]
    top_indices   = tfidf_scores.argsort()[::-1][:5]
    keywords      = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]

    tier_scores = {
        "Bad": bad_score,
        "Good": good_score,
        "Excellent": excellent_score,
    }
    sentiment = max(tier_scores, key=tier_scores.get)

    return {
        "sentiment"  : sentiment,
        "confidence" : confidence,
        "bad_score"  : bad_score,
        "good_score" : good_score,
        "excellent_score": excellent_score,
        # Backward-compatible aliases used by older frontend blocks.
        "pos_score"  : round(good_score + excellent_score, 1),
        "neg_score"  : bad_score,
        "keywords"   : keywords,
        "feedback"   : text,
    }

# ── Helper: summary stats ─────────────────────────────────────
def build_summary(results):
    total    = len(results)
    bad      = sum(1 for r in results if r['sentiment'] == 'Bad')
    good     = sum(1 for r in results if r['sentiment'] == 'Good')
    excellent= sum(1 for r in results if r['sentiment'] == 'Excellent')
    pos      = good + excellent
    neg      = bad
    avg_conf = round(sum(r['confidence'] for r in results) / total, 1) if total else 0
    return {
        "total"          : total,
        "bad_count"      : bad,
        "good_count"     : good,
        "excellent_count": excellent,
        "bad_pct"        : round(bad / total * 100, 1) if total else 0,
        "good_pct"       : round(good / total * 100, 1) if total else 0,
        "excellent_pct"  : round(excellent / total * 100, 1) if total else 0,
        # Compatibility fields.
        "positive_count" : pos,
        "negative_count" : neg,
        "positive_pct"   : round(pos / total * 100, 1) if total else 0,
        "negative_pct"   : round(neg / total * 100, 1) if total else 0,
        "avg_confidence" : avg_conf,
    }

def build_suggestions(summary):
    total         = summary.get("total", 0)
    confidence    = summary.get("avg_confidence", 0)
    positive_pct  = summary.get("positive_pct", 0)
    negative_pct  = summary.get("negative_pct", 0)
    sentiment_gap = abs(positive_pct - negative_pct)

    volume_text = "Response count is still low. Collect more feedback for stronger trend confidence."
    if 30 <= total < 100:
        volume_text = "Response volume is moderate and generally usable for directional decisions."
    elif total >= 100:
        volume_text = "Response volume is strong. You can treat this trend as broadly representative."

    confidence_text = "Model confidence is low. Review ambiguous comments manually before final conclusions."
    if 70 <= confidence < 85:
        confidence_text = "Model confidence is solid for high-level reporting, with selective manual checks."
    elif confidence >= 85:
        confidence_text = "Model confidence is high. This output is reliable for executive-style summaries."

    action_text = "Sentiment is mixed. Review negative comments and identify repeated friction themes."
    if sentiment_gap >= 30 and positive_pct > negative_pct:
        action_text = "Sentiment is clearly positive. Preserve what works and address isolated negative themes."
    elif sentiment_gap >= 30 and negative_pct > positive_pct:
        action_text = "Sentiment is clearly negative. Prioritize immediate interventions and follow-up communication."

    return [
        {
            "title": f"Response volume ({total})",
            "description": volume_text,
        },
        {
            "title": f"Confidence ({confidence}%)",
            "description": confidence_text,
        },
        {
            "title": "Recommended action",
            "description": action_text,
        },
    ]

def build_teacher_tips(results, summary):
    bad_rows = [r for r in results if r.get("sentiment") == "Bad"]
    if not bad_rows:
        return [
            "Great momentum. Keep collecting quick pulse feedback and reinforce the strategies students already value.",
            "Share one successful teaching practice with your team so positive outcomes scale across classes.",
            "Continue checking pacing and clarity regularly to maintain strong student confidence."
        ]

    keyword_counter = Counter()
    for row in bad_rows:
        for kw in row.get("keywords", []):
            keyword_counter[kw] += 1

    # Map recurring issue signals to concrete coaching tips.
    tip_map = {
        "confus": "Clarify one concept at a time: start each topic with a simple example, then move to advanced detail.",
        "unclear": "Make instructions explicit: provide assignment rubrics and one worked sample before students start.",
        "fast": "Slow pacing in high-friction lessons: add recap pauses every 10–15 minutes and quick comprehension checks.",
        "bor": "Increase engagement: use short interactive activities, polls, or pair discussions during lectures.",
        "dull": "Vary delivery format with stories, real-world cases, and student-led mini explanations.",
        "difficult": "Break difficult units into smaller milestones and publish a weekly 'what to focus on' guide.",
        "struggl": "Offer targeted support hours for struggling students and follow up with short action plans.",
        "outdat": "Refresh examples and references so students can connect learning with current real-world contexts.",
        "vagu": "Reduce vagueness with checklists: what good work includes, common mistakes, and grading expectations.",
        "rush": "Avoid rushed endings by reserving the last 5 minutes for recap, questions, and next-step guidance.",
    }

    tips = []
    for keyword, _count in keyword_counter.most_common(8):
        for signal, tip in tip_map.items():
            if signal in keyword and tip not in tips:
                tips.append(tip)
                break
        if len(tips) >= 3:
            break

    if len(tips) < 3:
        tips.append("Review top negative comments together and prioritize the top two recurring issues for next week.")
    if len(tips) < 3:
        tips.append("Ask students one follow-up question: 'What one change would most improve your learning experience?'")
    if len(tips) < 3:
        tips.append("Close the feedback loop by telling students what will change and when.")

    if summary.get("bad_pct", 0) >= 50:
        tips.insert(0, "Urgent focus: run a short teaching reset plan for the next 2 weeks and monitor trend movement weekly.")

    return tips[:4]

def save_analysis_run(run_type, source_ref, analyzed_column, payload):
    summary = payload.get("summary", {})
    conn = get_db_conn()
    try:
        conn.execute(
            """
            INSERT INTO analysis_runs (
                run_type, source_ref, analyzed_column,
                total, positive_count, negative_count, positive_pct, negative_pct, avg_confidence,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_type,
                source_ref,
                analyzed_column,
                int(summary.get("total", 0)),
                int(summary.get("positive_count", 0)),
                int(summary.get("negative_count", 0)),
                float(summary.get("positive_pct", 0)),
                float(summary.get("negative_pct", 0)),
                float(summary.get("avg_confidence", 0)),
                json.dumps(payload),
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()

def fetch_history(limit=20):
    conn = get_db_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, run_type, source_ref, analyzed_column, total,
                   positive_count, negative_count, positive_pct, negative_pct,
                   avg_confidence, created_at
            FROM analysis_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def count_runs():
    conn = get_db_conn()
    try:
        row = conn.execute("SELECT COUNT(*) AS count FROM analysis_runs").fetchone()
        return int(row["count"]) if row else 0
    finally:
        conn.close()

# ── Helper: parse CSV dataframe ───────────────────────────────
def process_df(df):
    # Auto-detect best feedback column
    feedback_col = None
    priority_kws = ['feedback','comment','response','answer','opinion','review','suggestion','thought']
    for col in df.columns:
        if any(k in col.lower() for k in priority_kws):
            feedback_col = col
            break
    if feedback_col is None:
        text_cols    = [c for c in df.columns if df[c].dtype == object]
        feedback_col = text_cols[-1] if text_cols else None
    if feedback_col is None:
        return None, None, "No text column found in the file."

    rows    = df[feedback_col].dropna().astype(str).tolist()
    rows    = [r.strip() for r in rows if len(r.strip()) > 3][:200]
    results = [analyze_text(r) for r in rows]
    summary = build_summary(results)
    suggestions = build_suggestions(summary)
    teacher_tips = build_teacher_tips(results, summary)
    return feedback_col, {"column": feedback_col, "columns": list(df.columns),
                          "summary": summary, "suggestions": suggestions, "teacher_tips": teacher_tips, "results": results}, None

# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/analyze', methods=['POST'])
def analyze():
    body = request.get_json()
    text = body.get('feedback', '').strip()
    if not text:
        return jsonify({"error": "No feedback provided"}), 400
    result = analyze_text(text)
    payload = {
        "summary": {
            "total": 1,
            "bad_count": 1 if result["sentiment"] == "Bad" else 0,
            "good_count": 1 if result["sentiment"] == "Good" else 0,
            "excellent_count": 1 if result["sentiment"] == "Excellent" else 0,
            "bad_pct": 100.0 if result["sentiment"] == "Bad" else 0.0,
            "good_pct": 100.0 if result["sentiment"] == "Good" else 0.0,
            "excellent_pct": 100.0 if result["sentiment"] == "Excellent" else 0.0,
            "positive_count": 0 if result["sentiment"] == "Bad" else 1,
            "negative_count": 1 if result["sentiment"] == "Bad" else 0,
            "positive_pct": 0.0 if result["sentiment"] == "Bad" else 100.0,
            "negative_pct": 100.0 if result["sentiment"] == "Bad" else 0.0,
            "avg_confidence": result["confidence"],
        },
        "results": [result],
    }
    payload["suggestions"] = build_suggestions(payload["summary"])
    save_analysis_run("single", "manual-text", "feedback", payload)
    return jsonify(result)


@app.route('/analyze-csv', methods=['POST'])
def analyze_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "Please upload a .csv file"}), 400
    try:
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8-sig')))  # utf-8-sig handles BOM
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {str(e)}"}), 400

    analyzed_col, payload, err = process_df(df)
    if err:
        return jsonify({"error": err}), 400
    save_analysis_run("csv", file.filename, analyzed_col, payload)
    return jsonify(payload)


@app.route('/analyze-sheets', methods=['POST'])
def analyze_sheets():
    body = request.get_json()
    url  = body.get('url', '').strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        if '/d/' in url:
            sheet_id = url.split('/d/')[1].split('/')[0]
            csv_url  = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        else:
            return jsonify({"error": "Invalid Google Sheets URL. Copy the link from File → Share → Copy link."}), 400

        df = pd.read_csv(csv_url)
    except Exception as e:
        return jsonify({"error": f"Could not fetch sheet. Make sure sharing is set to 'Anyone with the link'. Error: {str(e)}"}), 400

    analyzed_col, payload, err = process_df(df)
    if err:
        return jsonify({"error": err}), 400
    save_analysis_run("sheets", url, analyzed_col, payload)
    return jsonify(payload)

@app.route('/db-status', methods=['GET'])
def db_status():
    return jsonify({
        "database_path": DB_PATH,
        "analysis_runs_count": count_runs(),
    })

@app.route('/analysis-history', methods=['GET'])
def analysis_history():
    try:
        limit = int(request.args.get("limit", 20))
    except ValueError:
        limit = 20
    limit = max(1, min(limit, 200))
    return jsonify({
        "count": count_runs(),
        "items": fetch_history(limit=limit),
    })


@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/dashboard')
def dashboard():
    return send_from_directory(STATIC_DIR, 'dashboard.html')


if __name__ == '__main__':
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
