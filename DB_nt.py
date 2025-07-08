import cv2
import numpy as np
import asyncio
import gc
import sqlite3
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Initialize FastAPI and Middleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize YOLO model
model = YOLO('bindhu_ram.pt')
ZONE_COORDINATES = np.array([[(343, 16), (451, 16), (451, 530), (343, 530)]], np.int32)

# SQLite Database Configuration
DB_NAME = "check.db"

# Pydantic Models
class DetectionRequest(BaseModel):
    video_url: str
    session_id: str  # Unique session identifier

# Database Initialization
def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                video_url TEXT,
                created_time TEXT,
                status TEXT,
                detected_count INTEGER
            )
        """)
        conn.commit()
    finally:
        conn.close()


init_db()

# Helper Functions for Database Operations
def create_session(session_id: str, video_url: str):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (session_id, video_url, created_time, status, detected_count)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, video_url, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'running', 0))
        conn.commit()
    finally:
        conn.close()


def update_session_detection_count(session_id: str, count: int):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE sessions
            SET detected_count = detected_count + ?
            WHERE session_id = ?
        """, (count, session_id))
        conn.commit()
    finally:
        conn.close()


def stop_session(session_id: str):
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


def get_session(session_id: str):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        return cursor.fetchone()
    finally:
        conn.close()


def get_daily_count(target_date: str):
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


# Process Frame
async def process_frame(session_id, frame):
    results = model(frame)
    detection_count = 0

    for result in results:
        for box in result.boxes:
            if box.conf >= 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = model.names[class_id]
                centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2

                if cv2.pointPolygonTest(ZONE_COORDINATES, (centroid_x, centroid_y), False) >= 0:
                    detection_count += 1
                    color = (0, 255, 0) if class_name == 'blue_box' else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (centroid_x, centroid_y), 5, color, -1)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if detection_count > 0:
        update_session_detection_count(session_id, detection_count)

    cv2.polylines(frame, [ZONE_COORDINATES], isClosed=True, color=(0, 255, 0), thickness=2)


# Video Capture Loop
async def video_capture_loop(session_id, video_url):
    cap = cv2.VideoCapture(video_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame_index = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    frames_to_skip = 10

    try:
        while cap.isOpened():
            session = get_session(session_id)
            if not session or session[3] != 'running':
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 640))
            if frame_index % (fps // frames_to_skip) != 0:
                frame_index += 1
                continue

            await process_frame(session_id, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_index += 1
            if frame_index % 50 == 0:
                gc.collect()

            await asyncio.sleep(0)
    finally:
        cap.release()
        cv2.destroyAllWindows()


# Active sessions dictionary
active_sessions = {}

@app.post("/start_detection/")
async def start_detection(request: DetectionRequest):
    session_id = request.session_id
    video_url = request.video_url

    if video_url in active_sessions and active_sessions[video_url] == 'running':
        raise HTTPException(status_code=400, detail="A session is already running for this video URL.")

    if get_session(session_id):
        raise HTTPException(status_code=400, detail="Session already exists.")

    create_session(session_id, video_url)
    active_sessions[video_url] = 'running'
    asyncio.create_task(video_capture_loop(session_id, video_url))
    return {"status": "Detection started", "session_id": session_id}


@app.post("/stop_detection/{session_id}")
async def stop_detection(session_id: str):
    session = get_session(session_id)
    if not session or session[3] != 'running':
        raise HTTPException(status_code=400, detail="No active detection session found.")

    video_url = session[1]
    stop_session(session_id)
    if video_url in active_sessions:
        del active_sessions[video_url]

    return {"status": "Detection stopped", "session_id": session_id, "updated_status": "stopped"}


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
