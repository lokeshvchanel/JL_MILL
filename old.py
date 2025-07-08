from datetime import datetime
import os
import logging
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,field_validator
from typing import List

app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
sessions = {}
output_writers = {}
CONFIDENCE_THRESHOLD = 0.5
SAVE_VIDEO_DIR = "saved_videos"
os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)

class DetectionRequest(BaseModel):
    video_url: str
    zone_coordinates: List[List[int]]

    @field_validator("video_url")
    def validate_video_url(cls, value):
        if value not in ["4", "11"]:
            raise ValueError("Invalid video_url. Use '4' or '11'.")
        return value

    @field_validator("zone_coordinates")
    def validate_zone_coordinates(cls, value):
        if not all(len(coords) == 4 for coords in value):
            raise ValueError("Each zone must have 4 coordinates [x1, y1, x2, y2].")
        return value

def get_zone_coordinates(zone_coordinates):
    """Parse and validate zone coordinates."""
    try:
        zones = []
        for coords in zone_coordinates:
            x1, y1, x2, y2 = coords
            zones.append(((x1, y1), (x2, y2)))
        return zones
    except Exception as e:
        logger.error("Error parsing zone coordinates:%s", e)
        raise ValueError("Invalid zone coordinates format.") from e

def process_frame(frame, zones, model):
    """Process a video frame, perform detections, and annotate zones."""
    results = model(frame)
    detections = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), conf in zip(detections, confidences):
        if conf >= CONFIDENCE_THRESHOLD:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Draw zones
    for (start, end) in zones:
        cv2.rectangle(frame, start, end, (255, 0, 0), 2)
    return frame

def save_frame(frame, session_id):
    """Save a frame to the corresponding video file."""
    if session_id not in output_writers:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(SAVE_VIDEO_DIR, f"{session_id}_{now}.avi")
        height, width, _ = frame.shape
        output_writers[session_id] = cv2.VideoWriter(
            file_path, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (width, height)
        )
        logger.info(f"output writer initialized for session {session_id}: {file_path}")
    output_writers[session_id].write(frame)

@app.post("/detect")
async def detect_objects(request: DetectionRequest):
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Starting detection session: {session_id}")

    try:
        zones = get_zone_coordinates(request.zone_coordinates)
        cap = cv2.VideoCapture(0 if request.video_url == "4" else "test_video.mp4")

        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Unable to open video stream.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame, zones, model=None)  # Replace with YOLO model instance
            save_frame(frame, session_id)

        cap.release()
        if session_id in output_writers:
            output_writers[session_id].release()
            del output_writers[session_id]

        logger.info(f"Detection session {session_id} completed.")
        return {"status": "success", "session_id": session_id}

    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/sessions")
async def get_sessions():
    """Retrieve active sessions."""
    return {"active_sessions": list(sessions.keys())}

@app.delete("/sessions/{session_id}")
async def terminate_session(session_id: str):
    """Terminate a session and release resources."""
    if session_id in output_writers:
        output_writers[session_id].release()
        del output_writers[session_id]
    sessions.pop(session_id, None)
    logger.info(f"Session {session_id} terminated.")
    return {"status": "terminated", "session_id": session_id}