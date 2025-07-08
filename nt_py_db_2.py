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
import logging
import os

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

# Logging Configuration
# Logging Configuration
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")

# Configure logger
logger = logging.getLogger("app_logger")  # Hierarchical logger
logger.setLevel(logging.DEBUG)  # Set global logging level to DEBUG

# File Handler (write all logs to file)
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)  # Log all levels, including DEBUG, to the file

# Stream Handler (console output)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)  # Log everything to the console

# Custom Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

CONF_LOG_DIR = "conf_log"
os.makedirs(CONF_LOG_DIR, exist_ok=True)

# Log file paths for network and application errors
NETWORK_LOG_FILE = os.path.join(CONF_LOG_DIR, f"network_error_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")
APPLICATION_LOG_FILE = os.path.join(CONF_LOG_DIR, f"application_error_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")

# Custom formatter to include timestamp in the logs
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Logger for network-related errors
network_logger = logging.getLogger("network_error_logger")
network_logger.setLevel(logging.ERROR)  # Only log errors
network_file_handler = logging.FileHandler(NETWORK_LOG_FILE)
network_file_handler.setFormatter(formatter)
network_logger.addHandler(network_file_handler)

# Logger for application-related errors
app_logger = logging.getLogger("application_error_logger")
app_logger.setLevel(logging.ERROR)  # Only log errors
app_file_handler = logging.FileHandler(APPLICATION_LOG_FILE)
app_file_handler.setFormatter(formatter)
app_logger.addHandler(app_file_handler)

# Example of network error logging
def log_network_error(error_message):
    network_logger.error(error_message)

# Example of application error logging
def log_application_error(error_message):
    app_logger.error(error_message)

# Load the YOLO model
model = YOLO('bindhu_ram.pt')

# Detection zone and confidence from the config file
ZONE_COORDINATES = np.array([config.ZONE_COORDINATES], np.int32)
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD

# SQLite database setup
DB_NAME = "sessions.db"

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
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
    finally:
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
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        for key, value in kwargs.items():
            cursor.execute(f"UPDATE sessions SET {key} = ? WHERE session_id = ?", (value, session_id))
        conn.commit()
        logger.info(f"Session {session_id} updated with {kwargs}.")
    except Exception as e:
        logger.error(f"Error updating session {session_id}: {e}")
    finally:
        conn.close()

def get_session(session_id):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        session = cursor.fetchone()
        logger.debug(f"Retrieved session {session_id}: {session}.")
        return session
    except Exception as e:
        logger.error(f"Error retrieving session {session_id}: {e}")
    finally:
        conn.close()

# Create or clear session event
def create_session_event(session_id):
    active_sessions[session_id] = threading.Event()
    logger.info(f"Session event created for {session_id}.")

def clear_session_event(session_id):
    if session_id in active_sessions:
        active_sessions[session_id].set()  # Signal the event to stop
        del active_sessions[session_id]
        logger.info(f"Session event cleared for {session_id}.")

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
        logger.error(f"Error processing frame for session {session_id}, frame {frame_count}: {e}")

# Video capture loop
async def video_capture_loop(session_id, video_url):
    try:
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            update_session(session_id, status="error")
            logger.error(f"Failed to open video for session {session_id}.")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)
        frame_index = 0
        skip_frames = config.SKIP_FRAMES
        recent_frame_data = {"recent_frames": []}

        session_event = active_sessions.get(session_id)
        update_session(session_id, status="running")

        while cap.isOpened():
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
    except Exception as e:
        logger.error(f"Error in video capture loop for session {session_id}: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        update_session(session_id, status="stopped")

# API Endpoints
@app.post("/start_detection/")
async def start_detection(request: DetectionRequest):
    try:
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

        create_session_event(session_id)
        asyncio.create_task(video_capture_loop(session_id, video_url))
        logger.info(f"Detection started for session {session_id}.")
        return {"status": "Detection started", "session_id": session_id, "detected_count": 0}
    except Exception as e:
        logger.error(f"Error starting detection for session {session_id}: {e}")
        log_network_error(f"Network Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting detection")

@app.post("/stop_detection/{session_id}")
async def stop_detection(session_id: str):
    try:
        session = get_session(session_id)
        if not session or session[3] != "running":
            raise HTTPException(status_code=400, detail="No active detection session found")

        clear_session_event(session_id)
        logger.info(f"Detection stopping for session {session_id}.")
        return {"status": "Detection stopping"}
    except Exception as e:
        logger.error(f"Error stopping detection for session {session_id}: {e}")
        log_application_error(f"Application Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error stopping detection")

@app.get("/get_detected_count/{session_id}")
async def get_detected_count(session_id: str):
    try:
        session = get_session(session_id)
        if not session:
            raise HTTPException(status_code=400, detail="Session not found")
        return {"session_id": session_id, "detected_count": session[4], "status": session[3]}
    except Exception as e:
        logger.error(f"Error getting detected count for session {session_id}: {e}")
        log_application_error(f"Application Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving detected count")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
