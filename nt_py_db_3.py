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
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from twilio.rest import Client
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

# Logging Configuration
LOG_DIR = "log"
CONF_LOG_DIR = os.path.join(LOG_DIR, "conf_log")
os.makedirs(CONF_LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")
network_error_log = os.path.join(CONF_LOG_DIR, "network_error.log")
application_error_log = os.path.join(CONF_LOG_DIR, "application_error.log")

# Logging Level Based on Config
LOG_LEVEL = config.LOG_LEVEL.upper()

logger = logging.getLogger("app_logger")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# File Handlers
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

network_error_handler = logging.FileHandler(network_error_log)
network_error_handler.setLevel(logging.ERROR)

application_error_handler = logging.FileHandler(application_error_log)
application_error_handler.setLevel(logging.ERROR)

# Stream Handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Custom Formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
network_error_handler.setFormatter(formatter)
application_error_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.addHandler(network_error_handler)
logger.addHandler(application_error_handler)

# Enhanced Error Logging
def log_network_error(message):
    if LOG_LEVEL in ["ERROR", "DEBUG"]:
        network_error_logger = logging.getLogger("network_error_logger")
        network_error_logger.setLevel(logging.ERROR)
        network_error_logger.addHandler(network_error_handler)
        network_error_logger.error(message)

def log_application_error(message):
    if LOG_LEVEL in ["ERROR", "DEBUG"]:
        application_error_logger = logging.getLogger("application_error_logger")
        application_error_logger.setLevel(logging.ERROR)
        application_error_logger.addHandler(application_error_handler)
        application_error_logger.error(message)

# Internet connection check
def check_internet_connection():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False

# Load the YOLO model
model = YOLO('two_model.pt')

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
    except sqlite3.Error as e:
        log_application_error(f"SQLite Error initializing database: {e}")
        logger.error(f"Error initializing database: {e}")
    except Exception as e:
        log_application_error(f"General Error initializing database: {e}")
        logger.error(f"Error initializing database: {e}")
    finally:
        conn.close()

init_db()

# In-memory event store for session control
active_sessions = {}  # For session metadata
session_events = {}

# Pydantic model for detection request
class DetectionRequest(BaseModel):
    video_url: str
    session_id: str
class StopDetectionRequest(BaseModel):
    supervisor_name: str
    vehicle_number: str
    detected_count: int

    @validator("vehicle_number")
    def validate_vehicle_number(cls, value):
        pattern = re.compile(r"^[A-Z]{2} \d{2}(?: [A-Z]{2})? \d{1,4}$", re.IGNORECASE)
        if not pattern.match(value):
            raise ValueError("Invalid vehicle number format")
        return value
# Helper functions for database operations
def update_session(session_id, **kwargs):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        for key, value in kwargs.items():
            cursor.execute(f"UPDATE sessions SET {key} = ? WHERE session_id = ?", (value, session_id))
        conn.commit()
        if LOG_LEVEL in ["INFO", "DEBUG"]:
            logger.info(f"Session {session_id} updated with {kwargs}.")
    except sqlite3.Error as e:
        log_application_error(f"SQLite Error updating session {session_id}: {e}")
        logger.error(f"Error updating session {session_id}: {e}")
    except Exception as e:
        log_application_error(f"General Error updating session {session_id}: {e}")
        logger.error(f"Error updating session {session_id}: {e}")
    finally:
        conn.close()
def is_frame_recent(frame_index, recent_frames):
    """Check if the frame index is recent (to avoid processing the same frame multiple times)."""
    return any(frame_index - i < 30 for i in recent_frames)

# Update zone counts and log detections
async def update_zone_counts(session_id, class_name, recent_frame_data, frame_index):
    """
    Update detection count and log when objects are detected in the zone.
    """
    if session_id not in active_sessions:
        logger.error(f"Session {session_id} not found in active_sessions.")
        return

    if class_name =="box":
        if not is_frame_recent(frame_index, recent_frame_data["recent_frames"]):
            active_sessions[session_id]["detected_count"] += 1
            recent_frame_data["recent_frames"].append(frame_index)
            logger.info(f"Detected {class_name} in zone. Detected count: {active_sessions[session_id]['detected_count']}")


def get_session(session_id):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        session = cursor.fetchone()
        if LOG_LEVEL == "DEBUG":
            logger.debug(f"Retrieved session {session_id}: {session}.")
        return session
    except sqlite3.Error as e:
        log_application_error(f"SQLite Error retrieving session {session_id}: {e}")
        logger.error(f"Error retrieving session {session_id}: {e}")
    except Exception as e:
        log_application_error(f"General Error retrieving session {session_id}: {e}")
        logger.error(f"Error retrieving session {session_id}: {e}")
    finally:
        conn.close()

# Create or clear session event
def create_session_event(session_id):
    session_events[session_id] = threading.Event()  # Only manage events here
    active_sessions[session_id] = {"detected_count": 0, "active": True}  # Session metadata
    if LOG_LEVEL in ["INFO", "DEBUG"]:
        logger.info(f"Session event created for {session_id}.")

def clear_session_event(session_id):
    if session_id in session_events:
        session_events[session_id].set()  # Signal to stop the session
        del session_events[session_id]
    if session_id in active_sessions:
        del active_sessions[session_id]  # Remove session metadata
    if LOG_LEVEL in ["INFO", "DEBUG"]:
        logger.info(f"Session event cleared for {session_id}.")

# Video processing logic
async def process_frame(session_id, frame, frame_index, recent_frame_data):
    """
    Processes each frame for object detection and updates detection counts.
    Converts the frame to grayscale, adds detection count and frame index to the display.
    """
    try:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert grayscale back to BGR for drawing and displaying purposes
        processed_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # Get the YOLO model's detection results
        results = model(processed_frame)

        # Draw the detection zone
        cv2.polylines(processed_frame, [ZONE_COORDINATES], isClosed=True, color=(255, 0, 0), thickness=2)

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)  # Convert tensor to float
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Extract coordinates and class information
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert tensor to list of ints
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2

                    # Check if the centroid is inside the detection zone
                    in_zone = cv2.pointPolygonTest(ZONE_COORDINATES, (centroid_x, centroid_y), False) >= 0

                    if in_zone:
                        # Update detection count
                        await update_zone_counts(session_id, class_name, recent_frame_data, frame_index)

                        # Draw bounding box and label for objects in the zone
                        color = (0, 255, 0)  # Green for objects in the zone
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            processed_frame,
                            f"{class_name} {confidence:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            color,
                            2
                        )
                        # Draw the centroid
                        cv2.circle(processed_frame, (centroid_x, centroid_y), 5, color, -1)
                    else:
                        # Draw bounding box for objects outside the zone
                        color = (0, 0, 255)  # Red for objects outside the zone
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            processed_frame,
                            f"{class_name} {confidence:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            color,
                            2
                        )
                        # Draw the centroid
                        cv2.circle(processed_frame, (centroid_x, centroid_y), 5, color, -1)

        # Display frame index and detection count on the frame
        detection_count = active_sessions[session_id]["detected_count"]
        cv2.putText(
            processed_frame,
            f"Frame Index: {frame_index}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )
        cv2.putText(
            processed_frame,
            f"Detected Count: {detection_count}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        # Show the processed frame
        cv2.imshow("Detection Frame", processed_frame)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    except Exception as e:
        log_application_error(f"Error processing frame for session {session_id}, frame {frame_index}: {e}")
        logger.error(f"Error processing frame for session {session_id}, frame {frame_index}: {e}")


# Remaining parts of the code stay unchanged...


# Video capture loop
async def video_capture_loop(session_id, video_url):
    try:
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            update_session(session_id, status="error")
            log_network_error(f"Failed to open video for session {session_id}. URL: {video_url}")
            logger.error(f"Failed to open video for session {session_id}.")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)
        frame_index = 0
        skip_frames = config.SKIP_FRAMES
        recent_frame_data = {"recent_frames": []}
        session_event = session_events.get(session_id)  # Use session_events here
        update_session(session_id, status="running")

        while cap.isOpened():
            if session_event and session_event.is_set():  # Check event signal
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 640))
            if frame_index % skip_frames != 0:
                frame_index += 1
                continue

            # Process the frame
            await process_frame(session_id, frame, frame_index, recent_frame_data)
            frame_index += 1
            await asyncio.sleep(0)

    except cv2.error as e:
        log_application_error(f"OpenCV Error in video capture loop for session {session_id}: {e}")
        logger.error(f"Error in video capture loop for session {session_id}: {e}")
    except Exception as e:
        log_application_error(f"General Error in video capture loop for session {session_id}: {e}")
        logger.error(f"Error in video capture loop for session {session_id}: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        update_session(session_id, status="stopped")
def send_email(subject: str, body: str, recipient_email: str):
    """
    Sends an email using the SMTP configuration from the config file.

    :param subject: Subject of the email
    :param body: Body content of the email
    :param recipient_email: Recipient email address
    """
    try:
        # SMTP configuration
        smtp_server = config.SMTP_SERVER
        smtp_port = config.SMTP_PORT
        sender_email = config.SENDER_EMAIL
        sender_password = config.SENDER_PASSWORD

        # Email setup
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Sending the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")


def send_whatsapp_message(to: str, body: str):
    """
    Sends a WhatsApp message using Twilio.

    :param to: Recipient WhatsApp number in international format (e.g., '+919876543210')
    :param body: Message content
    """
    try:
        # Twilio credentials
        account_sid = config.TWILIO_ACCOUNT_SID
        auth_token = config.TWILIO_AUTH_TOKEN
        from_whatsapp_number = config.TWILIO_WHATSAPP_NUMBER

        # Initialize Twilio client
        client = Client(account_sid, auth_token)

        # Send the WhatsApp message
        message = client.messages.create(
            body=body,
            from_=f'whatsapp:{from_whatsapp_number}',
            to=f'whatsapp:{to}'
        )

        print(f"WhatsApp message sent successfully! Message SID: {message.sid}")
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")

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
        log_application_error(f"Error starting detection for session {session_id}: {e}")
        logger.error(f"Error starting detection for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error starting detection")

@app.post("/stop_detection/{session_id}")
async def stop_detection(session_id: str, request: StopDetectionRequest):
    try:
        session = get_session(session_id)
        if not session or session[3] != "running":
            raise HTTPException(status_code=400, detail="No active detection session found")

        clear_session_event(session_id)
        logger.info(f"Detection stopping for session {session_id}.")

        # Extract details from the request
        supervisor_name = request.supervisor_name
        vehicle_number = request.vehicle_number
        detected_count = request.detected_count

        # Prepare email and WhatsApp notifications
        subject = "Detection Session Ended"
        body = (
            f"Session ID: {session_id}\n"
            f"Supervisor Name: {supervisor_name}\n"
            f"Vehicle Number: {vehicle_number}\n"
            f"Detected Count: {detected_count}"
        )
        recipient_email = config.RECIPIENT_EMAIL
        send_email(subject, body, recipient_email)

        whatsapp_body = (
            f"Detection Session Ended\n"
            f"Session ID: {session_id}\n"
            f"Supervisor: {supervisor_name}\n"
            f"Vehicle: {vehicle_number}\n"
            f"Count: {detected_count}"
        )
        recipient_whatsapp = config.whatsapp_phone_number
        send_whatsapp_message(recipient_whatsapp, whatsapp_body)

        return {"status": "Detection stopping"}
    except Exception as e:
        log_application_error(f"Error stopping detection for session {session_id}: {e}")
        logger.error(f"Error stopping detection for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error stopping detection")

@app.get("/get_detected_count/{session_id}")
async def get_detected_count(session_id: str):
    try:
        session = get_session(session_id)
        if not session:
            raise HTTPException(status_code=400, detail="Session not found")
        return {"session_id": session_id, "detected_count": session[4], "status": session[3]}
    except Exception as e:
        log_application_error(f"Error getting detected count for session {session_id}: {e}")
        logger.error(f"Error getting detected count for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving detected count")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
