import cv2
import numpy as np
import asyncio
import gc
import os
import csv
from datetime import datetime
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import config
from twilio.rest import Client
import config
from datetime import datetime
import psutil
# Initialize the FastAPI app
app = FastAPI()
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
ZONE_COORDINATES_4 = np.array([[(201, 166), (344, 161), (408, 468), (201, 487)]], np.int32)
ZONE_COORDINATES_11 = np.array([[(185, 140), (397, 135), (397, 468), (189, 468)]], np.int32)
def get_zone_coordinates(video_url: str):
    # Normalize the input for reliable matching
    video_url = video_url.strip().lower()

    # Match directly based on the normalized input
    if video_url == "4":
        return ZONE_COORDINATES_4
    elif video_url == "11":
        return ZONE_COORDINATES_11
    elif "4" in video_url:
        return ZONE_COORDINATES_4
    elif "11" in video_url:
        return ZONE_COORDINATES_11
    else:
        raise ValueError("Invalid video URL")
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD

# Session storage to keep track of detection data
sessions = {}

# Pydantic model for detection request
class DetectionRequest(BaseModel):
    video_url: str
    session_id: str  # Unique session identifier

# Create or get a unified detection log file path
session_folder = None
log_file_path = None
ERROR_LOG_FOLDER = "error_logs"
if not os.path.exists(ERROR_LOG_FOLDER):
    os.makedirs(ERROR_LOG_FOLDER)

SYSTEM_ERROR_LOG = os.path.join(ERROR_LOG_FOLDER, "system_errors.csv")
APPLICATION_ERROR_LOG = os.path.join(ERROR_LOG_FOLDER, "application_errors.csv")
NETWORK_ERROR_LOG = os.path.join(ERROR_LOG_FOLDER, "network_errors.csv")

# Ensure CSV headers for error logs
for log_file in [SYSTEM_ERROR_LOG, APPLICATION_ERROR_LOG, NETWORK_ERROR_LOG]:
    if not os.path.exists(log_file):
        with open(log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Error Type", "Details"])

# Functions to log different error types
def log_error(error_type, description, details):
    log_file = {
        "System Error": SYSTEM_ERROR_LOG,
        "Application Error": APPLICATION_ERROR_LOG,
        "Network Error": NETWORK_ERROR_LOG,
    }.get(error_type, SYSTEM_ERROR_LOG)

    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), description, details])

    send_error_notification(error_type, description, details)  # Corrected call


def send_error_notification(error_type, description, details):
    subject = f"Critical {error_type} Alert"
    body = f"{description}\n\nDetails: {details}\n\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    recipient_email = config.RECIPIENT_EMAIL
    recipient_phone_number = config.whatsapp_phone_number

    # Send email
    send_email(subject, body, recipient_email)
    # Send WhatsApp message
    send_whatsapp_message(recipient_phone_number, body)


def get_detection_log_file_path():
    global session_folder, log_file_path

    # Define the detection logs main folder
    detection_folder = "detection_logs"
    if not os.path.exists(detection_folder):
        os.makedirs(detection_folder)

    # Create a session folder only if it hasn't been initialized
    if session_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = os.path.join(detection_folder, f"detection_{timestamp}")
        os.makedirs(session_folder, exist_ok=True)

    # Define a unified CSV file name based on the current date within the session folder
    if log_file_path is None:
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_file_path = os.path.join(session_folder, f"detected_count_{current_date}.csv")

        # Create the CSV file and write the header if it doesn't already exist
        if not os.path.exists(log_file_path):
            with open(log_file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Session ID", "Detection Count"])  # CSV header

    return log_file_path

async def log_detection_count(session_id, detection_count):
    # Fetch the file path
    log_file_path = get_detection_log_file_path()

    # Append new detection count to the CSV file
    with open(log_file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session_id, detection_count])

# Create log file for each session (not used for detection counts but for session-specific logs)
def get_log_file_path(session_id):
    log_folder = "logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = os.path.join(log_folder, f"log_file_{timestamp}")
    os.makedirs(session_folder, exist_ok=True)

    log_file_path = os.path.join(session_folder, f"log_file_{session_id}_log.csv")
    return log_file_path
# def monitor_system_health():
#     try:
#         memory = psutil.virtual_memory()
#         cpu_usage = psutil.cpu_percent(interval=1)

#         if memory.percent > 90:
#             log_error("System Error", "High memory usage", f"Memory usage: {memory.percent}%")
#         if cpu_usage > 90:
#             log_error("System Error", "High CPU usage", f"CPU usage: {cpu_usage}%")
#     except Exception as e:
#         log_error("System Monitoring Error", "System health monitoring failure", str(e))
# Log events to CSV file based on log level
async def log_event(log_file_path, event, frame_index=None, error_message=None):
    log_level = config.LOG_LEVEL
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if log_level == 'debug':
        log_data = [current_time, event, frame_index, error_message or ""]
    elif log_level == 'info' and event in ["Start Stream", "End Stream", "Frame Processed"]:
        log_data = [current_time, event, frame_index]
    elif log_level == 'error' and error_message:
        log_data = [current_time, event, frame_index, error_message]
    else:
        return

    with open(log_file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(log_data)

# Helper function to check recent frames
def is_frame_recent(frame_index, recent_frames):
    return any(frame_index - i < 30 for i in recent_frames)

# Update zone counts and log detections
async def update_zone_counts(session_id, class_name, recent_frame_data, log_file_path, frame_index):
    if session_id not in sessions:
        sessions[session_id] = {"active": False, "detected_count": 0, "recent_frames": []}

    if class_name == "box":
        if not is_frame_recent(frame_index, recent_frame_data["recent_frames"]):
            sessions[session_id]["detected_count"] += 1
            detection_count = sessions[session_id]["detected_count"]
            await log_detection_count(session_id, detection_count)

            await log_event(log_file_path, f"Detected {class_name}, Current Count: {detection_count}", frame_index)

            # Log to the unified CSV file
            recent_frame_data["recent_frames"].append(frame_index)

            return detection_count

    return None
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
async def process_frame(session_id, frame, frame_count, log_file_path, recent_frame_data):
    # Convert frame to grayscale
    global ZONE_COORDINATES
    # Fetch the correct zone coordinates
    video_url = sessions[session_id].get("video_url", "")
    try:
        ZONE_COORDINATES = get_zone_coordinates(video_url)
    except ValueError as e:
        log_error("Application Error", "Zone Coordinate Error", str(e))
        return
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_colored = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Keep a colored version for annotations

    # Initialize video writer if video_save is True
    if video_save:
        if session_id not in video_writers:
            video_output_folder = "saved_videos"
            if not os.path.exists(video_output_folder):
                os.makedirs(video_output_folder)
            video_output_path = os.path.join(video_output_folder, f"{session_id}.avi")
            fps = 30  # Set the desired frame rate
            frame_height, frame_width = frame_colored.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_writers[session_id] = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

    await log_event(log_file_path, "Entered process_frame", frame_index=frame_count)
    try:
        results = model(frame_colored)  # Pass the colored version for YOLO detection
        for result in results:
            for box in result.boxes:
                if box.conf >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2

                    if cv2.pointPolygonTest(ZONE_COORDINATES[0], (centroid_x, centroid_y), False) >= 0:
                        detection_count = await update_zone_counts(session_id, class_name, recent_frame_data, log_file_path, frame_count)

                        if detection_count is not None and class_name == 'box':
                            color = (0, 255, 0)
                            cv2.rectangle(frame_colored, (x1, y1), (x2, y2), color, 2)
                            cv2.circle(frame_colored, (centroid_x, centroid_y), 5, color, -1)
                            cv2.putText(frame_colored, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        await log_event(log_file_path, "Frame Processed", frame_index=frame_count)
        cv2.polylines(frame_colored, ZONE_COORDINATES, isClosed=True, color=(0, 255, 0), thickness=2)
        detected_count = sessions[session_id]["detected_count"]
        cv2.putText(frame_colored, f"Detected Count: {detected_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame_colored, f"Frame Count: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        key = cv2.waitKey(1)  # Check for key press
        if key == ord('s'):
            save_frame(frame_colored, frame_count)
        cv2.imshow(f"Detection - Session {session_id}", frame_colored)

        # Wait for keypress


        # Write the frame to video if video_save is True
        if video_save and session_id in video_writers:
            video_writers[session_id].write(frame_colored)

    except MemoryError as mem_err:
        log_error("System Error", "Memory Error", str(mem_err))
    except cv2.error as cv_err:
        log_error("Application Error", "OpenCV Error", str(cv_err))
    except Exception as e:
        log_error("Application Error", "Unhandled Exception", str(e))
    finally:
        await log_event(log_file_path, "Exited process_frame", frame_index=frame_count)
        del frame
# Video capture loop
async def video_capture_loop(session_id, video_url, log_file_path):
    await log_event(log_file_path, "Start Stream")
    try:

        cap = cv2.VideoCapture(video_url)


        if not cap.isOpened():
            raise ConnectionError("Unable to open video stream")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)
        frame_index = 0
        skip_frames = config.SKIP_FRAMES
        recent_frame_data = {"recent_frames": []}
        while cap.isOpened() and sessions.get(session_id, {}).get("active", False):
            # Monitor system health within the loop
            # monitor_system_health()

            ret, frame = cap.read()
            if not ret:
                raise ConnectionError("Failed to read frame from video stream")

            frame = cv2.resize(frame, (640, 640))
            if frame_index % skip_frames != 0:
                frame_index += 1
                continue

            await process_frame(session_id, frame, frame_index, log_file_path, recent_frame_data)
            frame_index += 1

            if frame_index % 50 == 0:
                gc.collect()
            await asyncio.sleep(0)

    except ConnectionError as net_err:
        log_error("Network Error", "Network Error", str(net_err))
    except Exception as e:
        log_error("Application Error", "Unhandled Exception", str(e))

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # await log_event(log_file_path, "End Stream")

        # await log_event(log_file_path, "End Stream")

        # Check if the session is still active before logging End Stream
        if sessions.get(session_id, {}).get("active", False) is False:
            await log_event(log_file_path, "End Stream")
        else:
            await log_event(log_file_path, "Error: Stream ended unexpectedly")

        print(f"Stream for session {session_id} ended")


@app.get("/get_detected_count/{session_id}")
async def get_detected_count(session_id: str):
    if session_id not in sessions or not sessions[session_id]["active"]:
        raise HTTPException(status_code=400, detail="Detection is not active for this session")
    return {"detected_count": sessions[session_id]["detected_count"]}
# Run detection log_file_path = get_log_file_path(session_id)  await log_event(log_file_path, f"Session {session_id} started detection")   await log_event(log_file_path, "Error in detection", error_message=str(e))
@app.post("/start_detection/")
async def start_detection(request: DetectionRequest):
    session_id = request.session_id
    video_url = request.video_url.strip()

    # Debugging to check what `video_url` is received
    print(f"Received video_url: {video_url}")

    # Add RTMP prefix if necessary
    rtmpurl = "rtmp://localhost/live/stream"
    video_url_full = f"{rtmpurl}{video_url}"
    print(f"Constructed video URL: {video_url_full}")

    if session_id in sessions and sessions[session_id]["active"]:
        return {"status": "Detection already running", "detected_count": sessions[session_id]["detected_count"]}

    sessions[session_id] = {"active": True, "detected_count": 0, "video_url": video_url}
    log_file_path = get_log_file_path(session_id)

    # Start video capture loop
    asyncio.create_task(video_capture_loop(session_id, video_url_full, log_file_path))
    try:
        await log_event(log_file_path, f"Session {session_id} started detection")
    except Exception as e:
        await log_event(log_file_path, "Error in detection", error_message=str(e))

    return {"status": "Detection started", "session_id": session_id, "detected_count": sessions[session_id]["detected_count"]}

def send_email(subject, body, recipient_email):
    try:
        # SMTP configuration (replace with your credentials)
        smtp_server = config.SMTP_SERVER
        smtp_port = config.SMTP_PORT
        sender_email = config.SENDER_EMAIL
        sender_password = config.SENDER_PASSWORD

        # Set up the email message
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Connect to the SMTP server and send the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Enable encryption
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")
def send_whatsapp_message(to: str, body: str):
    try:
        # Twilio credentials from config or environment variables
        account_sid = config.TWILIO_ACCOUNT_SID
        auth_token = config.TWILIO_AUTH_TOKEN
        from_whatsapp_number = config.TWILIO_WHATSAPP_NUMBER  # Your Twilio WhatsApp number

        # Initialize Twilio client
        client = Client(account_sid, auth_token)

        # Send WhatsApp message
        message = client.messages.create(
            body=body,
            from_=f'whatsapp:{from_whatsapp_number}',
            to=f'whatsapp:{to}'
        )

        print(f"WhatsApp message sent: {message.sid}")
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
@app.post("/stop_detection/{session_id}")
async def stop_detection(session_id: str):
    if session_id not in sessions or not sessions[session_id]["active"]:
        raise HTTPException(status_code=400, detail="No active detection session found")

    # Stop the session
    sessions[session_id]["active"] = False
    detected_count = sessions[session_id]["detected_count"]

    # Close the video writer if video_save is True
    if video_save and session_id in video_writers:
        video_writers[session_id].release()
        del video_writers[session_id]

    # Prepare email content
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"Detection Session {session_id} - Stopped"
    body = f"Session ID: {session_id}\nDate & Time: {current_time}\nDetected Count: {detected_count}"
    message = f"Session ID: {session_id}\nDate & Time: {current_time}\nDetected Count: {detected_count}"
    recipient_phone_number = config.whatsapp_phone_number
    # Send the email (replace with recipient email address)
    recipient_email = config.RECIPIENT_EMAIL
    recipient_email_bindhu =config.RECIPIENT_EMAIL_BINDHU
    send_email(subject, body, recipient_email_bindhu)
    send_email(subject, body, recipient_email)


    send_whatsapp_message(recipient_phone_number, message)

    return {"status": "Detection stopped", "detected_count": detected_count}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)