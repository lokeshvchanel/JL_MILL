import cv2
import numpy as np
import asyncio
import sqlite3
import threading
from datetime import datetime
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import config

# Initialize FastAPI app
app = FastAPI()

# Add CORS Middleware (optional for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model
model = YOLO('two_model.pt')

# Detection zone and confidence from the config file
ZONE_COORDINATES = np.array([config.ZONE_COORDINATES], np.int32)
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD

# SQLite database setup
DB_NAME = "sessions.db"

def init_db():
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
    conn.close()

init_db()

# In-memory event store for session control
active_sessions = {}

# Pydantic model for detection request
class DetectionRequest(BaseModel):
    video_url: str
    session_id: str  # Unique session identifier

# Helper functions for database operations
def update_session(session_id, **kwargs):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    for key, value in kwargs.items():
        cursor.execute(f"UPDATE sessions SET {key} = ? WHERE session_id = ?", (value, session_id))
    conn.commit()
    conn.close()

def get_session(session_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    session = cursor.fetchone()
    conn.close()
    return session

# Create or clear session event
def create_session_event(session_id):
    active_sessions[session_id] = threading.Event()

def clear_session_event(session_id):
    if session_id in active_sessions:
        active_sessions[session_id].set()  # Signal the event to stop
        del active_sessions[session_id]

# Video processing logic
async def process_frame(session_id, frame, frame_count, recent_frame_data):
    try:
        results = model(frame)
        for result in results:
            for box in result.boxes:
                if box.conf >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2

                    if cv2.pointPolygonTest(ZONE_COORDINATES, (centroid_x, centroid_y), False) >= 0:
                        if class_name in ['blue_box', 'brown_box'] and frame_count not in recent_frame_data["recent_frames"]:
                            recent_frame_data["recent_frames"].append(frame_count)
                            session = get_session(session_id)
                            detected_count = session[4] + 1  # Increment detected_count
                            update_session(session_id, detected_count=detected_count)
    except Exception as e:
        print(f"Error processing frame: {e}")

# Video capture loop
async def video_capture_loop(session_id, video_url):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        update_session(session_id, status="error")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)
    frame_index = 0
    skip_frames = config.SKIP_FRAMES
    recent_frame_data = {"recent_frames": []}

    # Use the event to control the session's active state
    session_event = active_sessions.get(session_id)

    try:
        update_session(session_id, status="running")
        while cap.isOpened():
            # Check if the event has been signaled to stop
            if session_event.is_set():
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 640))
            if frame_index % skip_frames != 0:
                frame_index += 1
                continue

            await process_frame(session_id, frame, frame_index, recent_frame_data)
            frame_index += 1
            await asyncio.sleep(0)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        update_session(session_id, status="stopped")

# API Endpoints
@app.post("/start_detection/")
async def start_detection(request: DetectionRequest):
    session_id = request.session_id
    video_url = request.video_url
    created_time = datetime.now().isoformat()

    session = get_session(session_id)
    if session:
        if session[3] == "running":
            return {"status": "Detection already running", "detected_count": session[4]}

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO sessions (session_id, video_url, created_time, status, detected_count)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, video_url, created_time, "pending", 0))
    conn.commit()
    conn.close()

    create_session_event(session_id)  # Create an event for this session
    asyncio.create_task(video_capture_loop(session_id, video_url))
    return {"status": "Detection started", "session_id": session_id, "detected_count": 0}

@app.post("/stop_detection/{session_id}")
async def stop_detection(session_id: str):
    session = get_session(session_id)
    if not session or session[3] != "running":
        raise HTTPException(status_code=400, detail="No active detection session found")

    clear_session_event(session_id)  # Signal the event to stop the loop
    return {"status": "Detection stopping"}

@app.get("/get_detected_count/{session_id}")
async def get_detected_count(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=400, detail="Session not found")
    return {"session_id": session_id, "detected_count": session[4], "status": session[3]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
