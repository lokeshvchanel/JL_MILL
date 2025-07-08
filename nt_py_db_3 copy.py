import cv2
import numpy as np
import asyncio
import sqlite3
import threading
from datetime import datetime
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import re
from fastapi.middleware.cors import CORSMiddleware
import config
import logging
import os
import socket
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from twilio.rest import Client

# Initialize FastAPI app
app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging Configuration
LOG_DIR = "log"
CONF_LOG_DIR = os.path.join(LOG_DIR, "conf_log")
os.makedirs(CONF_LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")
network_error_log = os.path.join(CONF_LOG_DIR, "network_error.log")
application_error_log = os.path.join(CONF_LOG_DIR, "application_error.log")

LOG_LEVEL = config.LOG_LEVEL.upper()

logger = logging.getLogger("app_logger")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

network_error_handler = logging.FileHandler(network_error_log)
network_error_handler.setLevel(logging.ERROR)

application_error_handler = logging.FileHandler(application_error_log)
application_error_handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
network_error_handler.setFormatter(formatter)
application_error_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.addHandler(network_error_handler)
logger.addHandler(application_error_handler)

# SQLite database setup
DB_NAME = "sessions.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                video_url TEXT UNIQUE,
                created_time TEXT,
                status TEXT,
                detected_count INTEGER
            )
        """)
        conn.commit()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        log_application_error(f"SQLite Error initializing database: {e}")
    finally:
        conn.close()

init_db()

# YOLO model setup
model = YOLO('two_model.pt')
ZONE_COORDINATES = np.array([config.ZONE_COORDINATES], np.int32)
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD

# In-memory active sessions tracker
active_sessions = {}

# Helper functions for database operations
def create_session(session_id, video_url):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (session_id, video_url, created_time, status, detected_count)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, video_url, datetime.now().isoformat(), "running", 0))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="This video URL is already being processed.")
    finally:
        conn.close()

def update_session(session_id, **kwargs):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        for key, value in kwargs.items():
            cursor.execute(f"UPDATE sessions SET {key} = ? WHERE session_id = ?", (value, session_id))
        conn.commit()
    finally:
        conn.close()

def stop_session(session_id):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE sessions
            SET status = 'stopped'
            WHERE session_id = ?
        """, (session_id,))
        conn.commit()
    finally:
        conn.close()

def get_session(session_id):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        return cursor.fetchone()
    finally:
        conn.close()

def get_daily_count(target_date):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT SUM(detected_count)
            FROM sessions
            WHERE DATE(created_time) = ?
        """, (target_date,))
        result = cursor.fetchone()
        return result[0] or 0
    finally:
        conn.close()

# Frame processing logic
async def process_frame(session_id, frame):
    results = model(frame)
    detection_count = 0

    for result in results:
        for box in result.boxes:
            if box.conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2
                in_zone = cv2.pointPolygonTest(ZONE_COORDINATES, (centroid_x, centroid_y), False) >= 0

                if in_zone:
                    detection_count += 1

    if detection_count > 0:
        update_session(session_id, detected_count=detection_count)

# Video capture loop
async def video_capture_loop(session_id, video_url):
    try:
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            stop_session(session_id)
            log_network_error(f"Failed to open video for session {session_id}. URL: {video_url}")
            return

        while cap.isOpened():
            session = get_session(session_id)
            if not session or session[3] != "running":
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 640))
            await process_frame(session_id, frame)
            await asyncio.sleep(0)
    finally:
        cap.release()
        stop_session(session_id)

# API Endpoints
class DetectionRequest(BaseModel):
    video_url: str
    session_id: str

@app.post("/start_detection/")
async def start_detection(request: DetectionRequest):
    session_id = request.session_id
    video_url = request.video_url

    if video_url in active_sessions:
        raise HTTPException(status_code=400, detail="This video URL is already being processed.")

    create_session(session_id, video_url)
    active_sessions[video_url] = "running"
    asyncio.create_task(video_capture_loop(session_id, video_url))
    return {"status": "Detection started", "session_id": session_id}

@app.post("/stop_detection/{session_id}")
async def stop_detection(session_id: str):
    session = get_session(session_id)
    if not session or session[3] != "running":
        raise HTTPException(status_code=400, detail="No active detection session found.")

    stop_session(session_id)
    active_sessions.pop(session[1], None)
    return {"status": "Detection stopped", "session_id": session_id}

@app.get("/get_detected_count/{session_id}")
async def get_detected_count(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    return {"session_id": session_id, "detected_count": session[4]}

@app.get("/get_daily_count/")
async def get_daily_count_endpoint():
    today_date = datetime.today().strftime('%Y-%m-%d')
    daily_count = get_daily_count(today_date)
    return {"date": today_date, "daily_count": daily_count}

# Initialize the database
init_db()
