import cv2
import numpy as np
import asyncio
import gc
import sqlite3
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import re
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import socket
import logging
import os
import config
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from twilio.rest import Client
# Initialize FastAPI and Middleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
def send_error_notification(subject: str, message: str):
    """
    Sends an error notification via email and WhatsApp.

    :param subject: Subject for the email notification
    :param message: Message content for both email and WhatsApp
    """
    try:
        # Email recipients
        recipient_emails = [
            config.RECIPIENT_EMAIL,
            config.RECIPIENT_EMAIL_VCHANEL,



        ]

        # Send email to all configured recipients
        for recipient in recipient_emails:
            send_email(subject, message, recipient)

        # WhatsApp notification
        recipient_whatsapp = config.whatsapp_phone_number
        send_whatsapp_message(recipient_whatsapp, message)
    except Exception as e:
        logger.error(f"Failed to send error notification: {e}")
# Enhanced Error Logging
def log_network_error(message):
    """
    Logs and sends notifications for network-related errors.
    """
    try:
        if LOG_LEVEL in ["ERROR", "DEBUG"]:
            network_error_logger = logging.getLogger("network_error_logger")
            network_error_logger.setLevel(logging.ERROR)
            network_error_logger.addHandler(network_error_handler)
            network_error_logger.error(message)

        # Send email and WhatsApp notifications for network errors
        subject = "Network Error Notification"
        send_error_notification(subject, message)
    except Exception as e:
        logger.error(f"Failed to send network error notification: {e}")

def log_application_error(message):
    """
    Logs and sends notifications for application-related errors.
    """
    try:
        if LOG_LEVEL in ["ERROR", "DEBUG"]:
            application_error_logger = logging.getLogger("application_error_logger")
            application_error_logger.setLevel(logging.ERROR)
            application_error_logger.addHandler(application_error_handler)
            application_error_logger.error(message)

        # Send email and WhatsApp notifications for application errors
        subject = "Application Error Notification"
        send_error_notification(subject, message)
    except Exception as e:
        logger.error(f"Failed to send application error notification: {e}")


# Internet connection check
def check_internet_connection():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False
async def ping_internet_connection():
    """
    Periodically checks for internet connectivity and sends notifications if the connection is lost.
    """
    while True:
        if not check_internet_connection():
            print("No internet connection. Sending notifications.")
            send_error_notification("Internet Connection Error", "No internet connection detected. Please check the network.")
        else:
            print("Internet connection is active.")

        await asyncio.sleep(60)
# Initialize YOLO model
model = YOLO('two_model.pt')
ZONE_COORDINATES = np.array([[(201,166), (344,161), (408,468), (201,487)]], np.int32)

# SQLite Database Configuration
DB_NAME = "sessions.db"
active_sessions = {}  # For session metadata
session_events = {}
# Pydantic Models
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
        return value # Unique session identifier

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
        if LOG_LEVEL in ["INFO", "DEBUG"]:
            logger.info(f"Session event created for {session_id}.")
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
        if LOG_LEVEL in ["INFO", "DEBUG"]:
            logger.info(f"Session {session_id} updated.")
    except sqlite3.Error as e:
        log_application_error(f"SQLite Error updating session {session_id}: {e}")
        logger.error(f"Error updating session {session_id}: {e}")
    except Exception as e:
        log_application_error(f"General Error updating session {session_id}: {e}")
        logger.error(f"Error updating session {session_id}: {e}")
    finally:
        conn.close()
recent_frames = []
def is_frame_recent(frame_index, recent_frames):
    for i in recent_frames:
        if frame_index - i < 30:
            return True
    return False

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
        if LOG_LEVEL in ["INFO", "DEBUG"]:
            logger.info(f"Session event stopped for {session_id}.")
    finally:
        conn.close()


def get_session(session_id: str):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        session=cursor.fetchone()
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
    except sqlite3.Error as e:
        log_application_error(f"SQLite Error get daily count : {e}")
        logger.error(f"Error get daily count : {e}")
    except Exception as e:
        log_application_error(f"General Error get daily count: {e}")
        logger.error(f"Error get daily count: {e}")
    finally:
        conn.close()

video_save = config.VIDEO_SAVE  # Set this in the config file (True/False)

# Video writer dictionary to handle multiple sessions
video_writers = {}
# Function to detect objects and process frames

frame_save_folder = "saved_frames"
if not os.path.exists(frame_save_folder):
    os.makedirs(frame_save_folder)

# Function to save the frame when 's' key is pressed
def save_frame(frame, frame_index):
    # Generate a unique filename for the frame
    filename = f"{frame_save_folder}/frame_{frame_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Frame saved as: {filename}")
# Process Frame
# Process Frame
async def process_frame(session_id, frame, frame_index, recent_frame_data):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_colored = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if config.VIDEO_SAVE:
        if session_id not in video_writers:
            video_output_folder = "saved_videos"
            if not os.path.exists(video_output_folder):
                os.makedirs(video_output_folder)
            video_output_path = os.path.join(video_output_folder, f"{session_id}.avi")
            fps = 30  # Set the desired frame rate
            frame_height, frame_width = frame_colored.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_writers[session_id] = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

        if session_id in video_writers:
            video_writers[session_id].write(frame_colored)
    try:
        results = model(frame_colored)
        detection_count = 0

        for result in results:
            for box in result.boxes:
                if box.conf >= 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2

                    if cv2.pointPolygonTest(ZONE_COORDINATES, (centroid_x, centroid_y), False) >= 0:
                        if not is_frame_recent(frame_index, recent_frames):
                            detection_count+=1
                            recent_frames.append(frame_index)  # Mark this frame as counted
                                # Mark the object as counted by drawing a different color
                            cv2.rectangle(frame_colored, (x1, y1), (x2, y2), (255, 255, 0), 5)
                            # cv2.putText(frame, f"detected count {detection_count}", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)



                        color = (0, 255, 0) if class_name == 'box' else (0, 0, 255)
                        cv2.rectangle(frame_colored, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(frame_colored, (centroid_x, centroid_y), 5, color, -1)
                        cv2.putText(frame_colored, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        if detection_count > 0:
            update_session_detection_count(session_id, detection_count)
        cv2.polylines(frame_colored, [ZONE_COORDINATES], isClosed=True, color=(0, 255, 0), thickness=2)
        total_detected = get_session(session_id)[4]

        cv2.putText(frame_colored, f"Detected Count: {total_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame_colored, f"Frame Count: {frame_index}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Detection Frame", frame_colored)
        if session_id in video_writers:
            video_writers[session_id].write(frame_colored)

            # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
    except MemoryError as mem_err:
        log_application_error("System Error ,Memory Error {mem_err}")
    except cv2.error as cv_err:
        log_application_error("Application Error ,OpenCV Error {cv_err}")
    except Exception as e:
        log_application_error("Application Error ,Unhandled Exception {e}")
    finally:

        del frame
# Video Capture Loop
# Video Capture Loop
async def video_capture_loop(session_id, video_url):
    # cap = cv2.VideoCapture(video_url)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # frame_index = 0
    # fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    # frames_to_skip = 10
    # recent_frame_data = {"recent_frames": []}
    try:

        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():

            log_network_error(f"Failed to open video for session {session_id}. URL: {video_url}")
            logger.error(f"Failed to open video for session {session_id}.")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)
        frame_index = 0
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        skip_frames = config.SKIP_FRAMES
        recent_frame_data = {"recent_frames": []}
        # session_event = session_events.get(session_id)
        while cap.isOpened():
            session = get_session(session_id)
            if not session or session[3] != 'running':
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 640))
            if frame_index % skip_frames != 0:
                frame_index += 1
                continue


            await process_frame(session_id, frame, frame_index, recent_frame_data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_index += 1
            if frame_index % 50 == 0:
                gc.collect()

            await asyncio.sleep(0)
    except ConnectionError as net_err:
       log_network_error ("Network Error, Network Error{net_err}")
    except Exception as e:
        log_network_error("Application Error ,Unhandled Exception {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()



# Active sessions dictionary
active_sessions = {}
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



# Active sessions dictionary
active_sessions = {}

@app.post("/start_detection/")
async def start_detection(request: DetectionRequest):
    try:
        session_id = request.session_id
        # video_url= request.video_url

        video_identifier= request.video_url
        rtmpurl="rtmp://localhost/live/stream"
        video_url=f"{rtmpurl}{video_identifier}"


        if video_url in active_sessions and active_sessions[video_url] == 'running':
            raise HTTPException(status_code=400, detail="A session is already running for this video URL.")

        if get_session(session_id):
            raise HTTPException(status_code=400, detail="Session already exists.")

        create_session(session_id, video_url)
        active_sessions[video_url] = 'running'
        asyncio.create_task(video_capture_loop(session_id, video_url))
        logger.info(f"Detection started for session {session_id}.")
        return {"status": "Detection started", "session_id": session_id}
    except Exception as e:
        log_application_error(f"Error starting detection for session {session_id}: {e}")
        logger.error(f"Error starting detection for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error starting detection")
async def startup_event():
    """
    Starts the background task for checking internet connectivity.
    """
    asyncio.create_task(ping_internet_connection())
@app.post("/stop_detection/{session_id}")
async def stop_detection(session_id: str, request: StopDetectionRequest):
    try:
        session = get_session(session_id)
        if not session or session[3] != 'running':
            raise HTTPException(status_code=400, detail="No active detection session found.")

        video_url = session[1]
        stop_session(session_id)
        if video_url in active_sessions:
            del active_sessions[video_url]
        supervisor_name = request.supervisor_name
        vehicle_number = request.vehicle_number
        detected_count = session[4]
        subject = "Detection Session Ended"
        body = (
            f"Session ID: {session_id}\n"
            f"Supervisor Name: {supervisor_name}\n"
            f"Vehicle Number: {vehicle_number}\n"
            f"Detected Count: {detected_count}"
        )
        recipient_email = config.RECIPIENT_EMAIL
        recipient_email_vchanle=config.RECIPIENT_EMAIL_VCHANEL
        recipient_email_bindhu=config.RECIPIENT_EMAIL_BINDHU


        send_email(subject, body, recipient_email)
        send_email(subject, body, recipient_email_vchanle)
        send_email(subject, body, recipient_email_bindhu)



        whatsapp_body = (
            f"Detection Session Ended\n"
            f"Session ID: {session_id}\n"
            f"Supervisor: {supervisor_name}\n"
            f"Vehicle: {vehicle_number}\n"
            f"Count: {detected_count}"
        )
        recipient_whatsapp = config.whatsapp_phone_number
        send_whatsapp_message(recipient_whatsapp, whatsapp_body)
        return {"status": "Detection stopped", "session_id": session_id, "updated_status": "stopped"}
    except Exception as e:
        log_application_error(f"Error stopping detection for session {session_id}: {e}")
        logger.error(f"Error stopping detection for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error stopping detection")

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
