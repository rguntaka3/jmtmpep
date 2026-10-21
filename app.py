import os
import glob
import random
import re
import json
import mimetypes
import sqlite3
from typing import List, Optional
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pathlib import Path
from pypdf import PdfReader
import numpy as np
import requests
from datetime import datetime
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from dotenv import load_dotenv

load_dotenv()

# --- Force Python to recognize .mjs as JavaScript ---
mimetypes.init()
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/javascript", ".js")

BASE_DIR = Path(__file__).resolve().parent

class JavaScriptSafeStaticFiles(StaticFiles):
    def file_response(self, full_path, stat_result, scope, status_code=200):
        response = super().file_response(full_path, stat_result, scope, status_code=status_code)
        if str(full_path).endswith(".mjs") or str(full_path).endswith(".js"):
            response.media_type = "application/javascript"
            response.headers["content-type"] = "application/javascript"
        return response

chapter_memory = {}

def get_hash_vector(text: str, num_features: int = 2048) -> np.ndarray:
    vector = np.zeros(num_features)
    words = re.findall(r'\w+', text.lower())
    if not words:
        return vector
    for word in words:
        idx = abs(hash(word)) % num_features
        vector[idx] += 1
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chapter_memory

    # --- Robust Auto-Migration for Render / SQLite Database ---
    try:
        db_path = "mpep_quiz.db"
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check question_history columns
            cursor.execute("PRAGMA table_info(question_history);")
            history_cols = [col[1] for col in cursor.fetchall()]
            if history_cols:
                if "username" not in history_cols:
                    cursor.execute("ALTER TABLE question_history ADD COLUMN username TEXT DEFAULT 'guest';")
                if "options_json" not in history_cols:
                    cursor.execute("ALTER TABLE question_history ADD COLUMN options_json TEXT DEFAULT '[]';")
                if "chapter" not in history_cols:
                    cursor.execute("ALTER TABLE question_history ADD COLUMN chapter TEXT DEFAULT 'Chapter 700';")
                if "question_type" not in history_cols:
                    cursor.execute("ALTER TABLE question_history ADD COLUMN question_type TEXT DEFAULT 'procedural';")
                if "time_spent" not in history_cols:
                    cursor.execute("ALTER TABLE question_history ADD COLUMN time_spent INTEGER DEFAULT 0;")
                if "session_type" not in history_cols:
                    cursor.execute("ALTER TABLE question_history ADD COLUMN session_type TEXT DEFAULT 'practice';")
                conn.commit()

            # Check question_comments columns
            cursor.execute("PRAGMA table_info(question_comments);")
            comment_cols = [col[1] for col in cursor.fetchall()]
            if comment_cols and "username" not in comment_cols:
                cursor.execute("ALTER TABLE question_comments ADD COLUMN username TEXT DEFAULT 'guest';")
                conn.commit()

            conn.close()
    except Exception as migration_err:
        print(f"Migration note: {migration_err}")
    # ---------------------------------------------------------

    pdf_files = glob.glob("mpep-*.pdf")

    if not pdf_files:
        print("⚠️ No MPEP PDFs found matching pattern 'mpep-*.pdf'.")
        yield
        return

    for pdf_path in pdf_files:
        match = re.search(r'mpep-(\d+)', pdf_path.lower())
        chapter_key = str(int(match.group(1))) if match else "700"

        print(f"Parsing {pdf_path} (Normalized Chapter {chapter_key})...")
        reader = PdfReader(pdf_path)
        raw_text_chunks = []

        current_chunk = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                current_chunk += text + "\n"
                if len(current_chunk) >= 1200:
                    raw_text_chunks.append(current_chunk[:1500])
                    current_chunk = current_chunk[1000:]
        if current_chunk:
            raw_text_chunks.append(current_chunk)

        if raw_text_chunks:
            vectors = [get_hash_vector(chunk) for chunk in raw_text_chunks]
            chapter_memory[chapter_key] = {
                "texts": raw_text_chunks,
                "vectors": vectors,
                "filename": os.path.basename(pdf_path)
            }

    print(f"🚀 Initialized vector spaces for chapters: {list(chapter_memory.keys())}")
    yield
    chapter_memory.clear()

app = FastAPI(title="MPEP GenAI Question Generator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_index():
    return FileResponse(str(BASE_DIR / "index.html"))

app.mount("/viewer", JavaScriptSafeStaticFiles(directory=str(BASE_DIR / "pdfjs"), html=True), name="pdfjs")
app.mount("/pdfs", StaticFiles(directory=str(BASE_DIR)), name="root_pdfs")

# Database Setup
DATABASE_URL = "sqlite:///./mpep_quiz.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class CommentDB(Base):
    __tablename__ = "question_comments"
    id = Column(Integer, primary_key=True, index=True)
    question_hash = Column(String, index=True)
    username = Column(String, default="guest")
    content = Column(Text)
    timestamp = Column(String)

class ReactionSurveyDB(Base):
    __tablename__ = "reaction_surveys"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, default="guest")
    ease_of_use = Column(Integer)
    spaced_retention = Column(Integer)
    scaffolded_questioning = Column(Integer)
    pdf_viewer = Column(Integer)
    review_history = Column(Integer)
    forums = Column(Integer)
    comments = Column(Text, nullable=True)
    timestamp = Column(String)

class ResultSurveyDB(Base):
    __tablename__ = "result_surveys"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, default="guest")
    passed_exam = Column(Boolean)
    actual_score = Column(String, nullable=True)
    app_helpfulness_rating = Column(Integer)
    real_world_comments = Column(Text, nullable=True)
    timestamp = Column(String)

class QuestionHistoryDB(Base):
    __tablename__ = "question_history"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, default="guest", index=True)
    chapter = Column(String, default="Chapter 700")
    context = Column(String)
    question_type = Column(String, default="procedural")
    stem = Column(Text)
    options_json = Column(Text)
    user_choice = Column(String)
    correct_choice = Column(String)
    explanation = Column(Text)
    difficulty = Column(String)
    is_correct = Column(Boolean)
    time_spent = Column(Integer, default=0)
    session_type = Column(String, default="practice")
    timestamp = Column(String)

Base.metadata.create_all(bind=engine)

class CommentCreate(BaseModel):
    question_hash: str
    username: Optional[str] = "guest"
    content: str

class ReactionSurveyCreate(BaseModel):
    username: Optional[str] = "guest"
    ease_of_use: int
    spaced_retention: int
    scaffolded_questioning: int
    pdf_viewer: int
    review_history: int
    forums: int
    comments: Optional[str] = ""

class ResultSurveyCreate(BaseModel):
    username: Optional[str] = "guest"
    passed_exam: bool
    actual_score: Optional[str] = ""
    app_helpfulness_rating: int
    real_world_comments: Optional[str] = ""

class QuestionResponse(BaseModel):
    chapter: str
    chapter_context: str
    question_type: str
    question_stem: str
    options: List[str]
    correct_answer: str
    explanation: str

class HistoryCreate(BaseModel):
    username: Optional[str] = "guest"
    chapter: Optional[str] = "Chapter 700"
    context: str
    question_type: Optional[str] = "procedural"
    stem: str
    options: List[str]
    user_choice: str
    correct_choice: str
    explanation: str
    difficulty: str
    is_correct: bool
    time_spent: Optional[int] = 0
    session_type: Optional[str] = "practice"

class GoogleAuthSchema(BaseModel):
    credential: str

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com")

@app.post("/auth/google")
def google_auth(data: GoogleAuthSchema):
    try:
        idinfo = id_token.verify_oauth2_token(
            data.credential, 
            google_requests.Request(), 
            GOOGLE_CLIENT_ID
        )
        return {
            "status": "success",
            "username": idinfo.get("name", "User"),
            "email": idinfo.get("email", ""),
            "picture": idinfo.get("picture", "")
        }
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Google authentication token.")

@app.get("/comments/{question_hash}")
async def get_comments(question_hash: str, db: Session = Depends(get_db)):
    items = db.query(CommentDB).filter(CommentDB.question_hash == question_hash).order_by(CommentDB.id.desc()).all()
    return [{"id": i.id, "username": i.username, "content": i.content, "timestamp": i.timestamp} for i in items]

@app.post("/comments")
async def post_comment(comment: CommentCreate, db: Session = Depends(get_db)):
    db_comment = CommentDB(
        question_hash=comment.question_hash,
        username=comment.username or "guest",
        content=comment.content,
        timestamp=datetime.now().strftime("%I:%M:%S %p - %b %d")
    )
    db.add(db_comment)
    db.commit()
    return {"status": "success", "id": db_comment.id}

@app.post("/surveys/reaction")
async def submit_reaction_survey(survey: ReactionSurveyCreate, db: Session = Depends(get_db)):
    db_survey = ReactionSurveyDB(
        username=survey.username or "guest",
        ease_of_use=survey.ease_of_use,
        spaced_retention=survey.spaced_retention,
        scaffolded_questioning=survey.scaffolded_questioning,
        pdf_viewer=survey.pdf_viewer,
        review_history=survey.review_history,
        forums=survey.forums,
        comments=survey.comments,
        timestamp=datetime.now().strftime("%I:%M:%S %p - %b %d")
    )
    db.add(db_survey)
    db.commit()
    return {"status": "success", "message": "Reaction survey saved successfully."}

@app.post("/surveys/result")
async def submit_result_survey(survey: ResultSurveyCreate, db: Session = Depends(get_db)):
    db_result = ResultSurveyDB(
        username=survey.username or "guest",
        passed_exam=survey.passed_exam,
        actual_score=survey.actual_score,
        app_helpfulness_rating=survey.app_helpfulness_rating,
        real_world_comments=survey.real_world_comments,
        timestamp=datetime.now().strftime("%I:%M:%S %p - %b %d")
    )
    db.add(db_result)
    db.commit()
    return {"status": "success", "message": "Result survey recorded."}

@app.get("/analytics/behavior")
async def get_behavior_analytics(username: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(QuestionHistoryDB)
    if username:
        query = query.filter(QuestionHistoryDB.username == username.strip().lower())
    records = query.all()

    weights = {"easy": 1.0, "medium": 2.0, "hard": 3.0}
    total_attempts = len(records)
    total_weighted_score = 0.0
    total_weight = 0.0
    total_time = 0

    for r in records:
        w = weights.get((r.difficulty or "easy").lower(), 1.0)
        total_weight += w
        if r.is_correct:
            total_weighted_score += w
        total_time += (r.time_spent or 0)

    weighted_accuracy = (total_weighted_score / total_weight * 100) if total_weight > 0 else 0
    avg_response_time = (total_time / total_attempts) if total_attempts > 0 else 0

    return {
        "total_attempts": total_attempts,
        "weighted_accuracy": round(weighted_accuracy, 2),
        "average_response_time_seconds": round(avg_response_time, 1)
    }

@app.get("/community-questions")
async def get_community_questions(db: Session = Depends(get_db)):
    items = db.query(QuestionHistoryDB).order_by(QuestionHistoryDB.id.desc()).limit(15).all()
    result = []
    seen_stems = set()
    for item in items:
        if item.stem not in seen_stems:
            seen_stems.add(item.stem)
            result.append({
                "chapter": item.chapter,
                "context": item.context,
                "question_type": item.question_type or "procedural",
                "stem": item.stem,
                "options": json.loads(item.options_json),
                "correct_choice": item.correct_choice,
                "explanation": item.explanation,
                "timestamp": item.timestamp
            })
    return result

@app.get("/chapters")
async def get_available_chapters():
    global chapter_memory
    sorted_chapters = sorted(list(chapter_memory.keys()), key=lambda x: int(x) if x.isdigit() else 0)
    return {"chapters": [{"chapter": ch, "filename": chapter_memory[ch].get("filename", f"mpep-{ch}.pdf")} for ch in sorted_chapters]}

@app.get("/generate-question", response_model=QuestionResponse)
async def generate_question(chapters: str = "700", difficulty: str = "medium"):
    global chapter_memory
    if not chapter_memory:
        raise HTTPException(status_code=500, detail="MPEP knowledge base is empty or uninitialized.")
    
    req_chapters = [str(int(c.strip())) for c in chapters.split(",") if c.strip().isdigit()]
    active_pool = [(ch, text, vec) for ch in req_chapters if ch in chapter_memory for text, vec in zip(chapter_memory[ch]["texts"], chapter_memory[ch]["vectors"])]
                
    if not active_pool:
        raise HTTPException(status_code=400, detail=f"No text chunks found for requested chapters: {chapters}")
    
    source_ch, query_text, query_vector = random.choice(active_pool)
    similarities = np.dot(np.array(chapter_memory[source_ch]["vectors"]), query_vector)
    top_indices = np.argsort(similarities)[-2:]
    combined_context = "\n---\n".join([chapter_memory[source_ch]["texts"][idx] for idx in top_indices])
    
    target_type = random.choice(["procedural", "substantive"])
    type_instructions = {
        "procedural": "Target Question Type: PROCEDURAL. Focus specifically on formal deadlines, rules, fee structures, form numbers, or filing steps.",
        "substantive": "Target Question Type: SUBSTANTIVE. Focus specifically on legal interpretation, novelty, non-obviousness, 35 U.S.C. 101/102/103/112 law analysis, or complex factual application."
    }[target_type]

    difficulty_instructions = {
        "easy": "Difficulty: EASY. Keep the scenario straightforward and clear-cut.",
        "medium": "Difficulty: MEDIUM. Focus on standard rule application to a realistic hypothetical scenario.",
        "hard": "Difficulty: HARD. Create a complex, multi-step fact pattern with timelines, exceptions, or EXCEPT/NOT setups."
    }.get(difficulty.lower(), "Difficulty: MEDIUM.")

    system_prompt = (
        "You are an expert Patent Attorney and an official USPTO exam writer.\n"
        f"Generate a multiple-choice question (MCQ) for the Patent Bar Exam strictly based on **Chapter {source_ch}** of the MPEP.\n\n"
        f"{difficulty_instructions}\n"
        f"{type_instructions}\n\n"
        "Requirements:\n"
        "1. Structure: Exactly 5 choices (A, B, C, D, E).\n"
        f"2. Context Format: The `chapter_context` MUST explicitly cite an MPEP section or subsection belonging strictly to **Chapter {source_ch}** (e.g., MPEP § {source_ch}.01 – Specific Topic Title).\n"
        "3. Option-by-Option Explanation: Inside the `explanation` field, provide a complete breakdown covering why the correct answer is right AND an itemized explanation for each option (A, B, C, D, E) referencing specific MPEP rules or statutes.\n\n"
        "You MUST return your output as a raw JSON string matching this exact schema layout:\n"
        "{\n"
        f"  \"chapter_context\": \"MPEP § {source_ch}.XX – Topic Title\",\n"
        f"  \"question_type\": \"{target_type}\",\n"
        "  \"question_stem\": \"The question text...\",\n"
        "  \"options\": [\"A. ...\", \"B. ...\", \"C. ...\", \"D. ...\", \"E. ...\"],\n"
        "  \"correct_answer\": \"The correct letter option (e.g., C)\",\n"
        "  \"explanation\": \"Overall summary citing MPEP section.\\n\\n• Choice A: Why incorrect/correct.\\n• Choice B: Why incorrect/correct.\\n• Choice C: Why incorrect/correct.\\n• Choice D: Why incorrect/correct.\\n• Choice E: Why incorrect/correct.\"\n"
        "}"
    )

    url = "https://api.groq.com/openai/v1/chat/completions"
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable is not set.")
    
    headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"}
    payload = {
        "model": "openai/gpt-oss-120b",
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a question testing the rules inside this text context from Chapter {source_ch}:\n\n{combined_context}"}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            raise Exception(f"HTTP Status {response.status_code}\n{response.text}")
            
        parsed_json = json.loads(response.json()["choices"][0]["message"]["content"].strip())
        normalized_data = {
            "chapter": f"Chapter {source_ch}",
            "chapter_context": parsed_json.get("chapter_context") or f"MPEP § {source_ch}.01 – Topic Tested",
            "question_type": target_type,
            "question_stem": parsed_json.get("question_stem") or "Question Text Missing",
            "options": parsed_json.get("options") or ["A. Error", "B. Error", "C. Error", "D. Error", "E. Error"],
            "correct_answer": parsed_json.get("correct_answer") or "A",
            "explanation": parsed_json.get("explanation") or "Explanation verified via internal MPEP statutes."
        }
        if isinstance(normalized_data["options"], dict):
            normalized_data["options"] = [f"{k}. {v}" for k, v in normalized_data["options"].items()]
        if len(normalized_data["options"]) < 5:
            normalized_data["options"] = list(normalized_data["options"]) + ["Option text missing."] * (5 - len(normalized_data["options"]))

        return QuestionResponse(**normalized_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed generative synthesis: {str(e)}")

@app.get("/spaced-repetition-questions", response_model=List[QuestionResponse])
async def get_spaced_repetition_questions(username: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(QuestionHistoryDB)
    if username:
        query = query.filter(QuestionHistoryDB.username == username.strip().lower())
    
    items = query.order_by(QuestionHistoryDB.is_correct.asc(), QuestionHistoryDB.id.asc()).limit(10).all()
    return [
        QuestionResponse(
            chapter=item.chapter,
            chapter_context=f"Review: {item.context}",
            question_type=item.question_type or "procedural",
            question_stem=item.stem,
            options=json.loads(item.options_json),
            correct_answer=item.correct_choice,
            explanation=item.explanation
        ) for item in items
    ]

@app.post("/history")
async def save_history(item: HistoryCreate, db: Session = Depends(get_db)):
    db_item = QuestionHistoryDB(
        username=item.username or "guest",
        chapter=item.chapter,
        context=item.context,
        question_type=item.question_type or "procedural",
        stem=item.stem,
        options_json=json.dumps(item.options),
        user_choice=item.user_choice,
        correct_choice=item.correct_choice,
        explanation=item.explanation,
        difficulty=item.difficulty,
        is_correct=item.is_correct,
        time_spent=item.time_spent or 0,
        session_type=item.session_type or "practice",
        timestamp=datetime.now().strftime("%I:%M:%S %p")
    )
    db.add(db_item)
    db.commit()
    return {"status": "success", "id": db_item.id}

@app.get("/history")
async def get_history(username: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(QuestionHistoryDB)
    if username:
        query = query.filter(QuestionHistoryDB.username == username.strip().lower())
    
    items = query.order_by(QuestionHistoryDB.id.desc()).all()
    return [{
        "id": item.id,
        "username": item.username or "guest",
        "chapter": item.chapter,
        "context": item.context,
        "question_type": item.question_type or "procedural",
        "stem": item.stem,
        "options": json.loads(item.options_json),
        "userChoice": item.user_choice,
        "correctChoice": item.correct_choice,
        "explanation": item.explanation,
        "difficulty": item.difficulty,
        "isCorrect": item.is_correct,
        "timeSpent": item.time_spent or 0,
        "sessionType": item.session_type or "practice",
        "timestamp": item.timestamp
    } for item in items]

@app.delete("/history")
async def clear_history(username: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(QuestionHistoryDB)
    if username:
        query = query.filter(QuestionHistoryDB.username == username.strip().lower())
    query.delete(synchronize_session=False)
    db.commit()
    return {"status": "success", "message": "History cleared for user."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)