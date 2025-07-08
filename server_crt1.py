"""
This module implements a FastAPI-based object detection service using OpenCV.
It supports detection in video streams, saving annotated frames, and managing detection sessions.

Key functionalities:
- Validate zone coordinates and video sources.
- Process video frames with object detection.
- Save annotated frames to video files.
- Manage active detection sessions via REST endpoints.
"""
from datetime import datetime
import os
import logging
from typing import List
import cv2 as cv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,field_validator


app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
sessions = {}
output_writers = {}
CONFIDENCE_THRESHOLD = 0.5
SAVE_VIDEO_DIR = "saved_videos"
os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)

class DetectionRequest(BaseModel):
    """Request model for object detection service."""
    video_url: str
    zone_coordinates: List[List[int]]
#VIDEO URL
    @field_validator("video_url")
    @classmethod
    def validate_video_url(cls, value: str) -> str:
        """Validate video URL and return the URL if valid, otherwise raise an exception."""
        if value not in ["4", "11"]:
            raise ValueError("Invalid video_url. Use '4' or '11'.")
        return value
    #zone validator
    @field_validator("zone_coordinates")
    @classmethod
    def validate_zone_coordinates(cls, value: List[List[int]]) -> List[List[int]]:
        """Validate zone coordinates and return the coordinates if valid"""
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
        logger.error("Error parsing zone coordinates:%s",e)
        raise ValueError("Invalid zone coordinates format.") from e

def process_frame(frame, zones, model):
    """Process a video frame, perform detections, and annotate zones."""
    results = model(frame)
    detections = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), conf in zip(detections, confidences):
        if conf >= CONFIDENCE_THRESHOLD:
            # pylint: disable=no-member
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Draw zones
    for (start, end) in zones:
        # pylint: disable=no-member
        cv.rectangle(frame, start, end, (255, 0, 0), 2)
    return frame

def save_frame(frame, session_id):
    """Save a frame to the corresponding video file."""
    if session_id not in output_writers:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(SAVE_VIDEO_DIR, f"{session_id}_{now}.avi")
        height, width, _ = frame.shape
        # pylint: disable=no-member
        fourcc = cv.VideoWriter_fourcc('X', 'V', 'I', 'D')
        # pylint: disable=no-member
        output_writers[session_id] = cv.VideoWriter(
        file_path, fourcc, 20.0, (width, height)
        )

        logger.info("output writer initialized for session %s:%s", session_id ,file_path)
    output_writers[session_id].write(frame)

@app.post("/detect")
async def detect_objects(request: DetectionRequest):
    """ this is detect object"""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info("Starting detection session:%s", session_id)

    try:
        zones = get_zone_coordinates(request.zone_coordinates)
        # pylint: disable=no-member
        cap = cv.VideoCapture(0 if request.video_url == "4" else "test_video.mp4")

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

        logger.info("Detection session %s  completed.",session_id)
        return {"status": "success", "session_id": session_id}

    except Exception as e:
        logger.error("Error during detection:%s",e)
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
    logger.info("Session %s terminated.",session_id)
    return {"status": "terminated", "session_id": session_id}
