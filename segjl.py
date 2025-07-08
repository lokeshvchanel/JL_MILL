from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import cv2
import numpy as np
import os
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
import gc
from ultralytics.utils.plotting import Annotator, colors

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('js.pt')
class_list = model.names
countable_classes = ["bag5", "bale1", "tbale2", "tbox5"]

sessions = {}
video_writers = {}
video_save = True
CONFIDENCE_THRESHOLD = 0.5
line_x = 500  

class_counts_by_session = defaultdict(lambda: defaultdict(int))
cumulative_count = 0
detection_running = False
crossed_ids = set()  

class_count_mapping = {
    'tbox5': 5, 'bag5': 5,
    'tbale2': 2, 'bale1': 1
}

# Pydantic models
class DetectionRequest(BaseModel):
    video_url: str
    session_id: str

def get_log_file_path(session_id):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{session_id}.log")

async def log_event(log_file_path, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

async def update_line_count(session_id, class_name, track_id, log_file_path, centroid_x):
    global cumulative_count
    
    if session_id not in sessions:
        sessions[session_id] = {"active": False, "detected_count": 0}

   
    if track_id in crossed_ids:
        return  

    if class_name in countable_classes and (line_x - 10 < centroid_x < line_x + 10):
        increment = class_count_mapping.get(class_name, 0)
        cumulative_count += increment
        sessions[session_id]["detected_count"] += 1
        class_counts_by_session[session_id][class_name] += increment
        crossed_ids.add(track_id)  

        print(f"Updated count for {class_name}: {class_counts_by_session[session_id][class_name]}")
        await log_event(log_file_path, f"Detected {class_name} (ID {track_id}), Count: {sessions[session_id]['detected_count']}")

async def process_frame(session_id, im0, frame_count, log_file_path):
    annotator = Annotator(im0, line_width=2)
    results = model.track(im0, persist=True)
    
    if results[0].boxes.id is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask, cls, track_id in zip(masks, clss, track_ids):
            color = colors(int(track_id), True)
            class_name = model.model.names[int(cls)]
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask=mask, mask_color=color, label=class_name, txt_color=txt_color)

            mask_array = np.array(mask, dtype=np.int32)
            M = cv2.moments(mask_array)
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
                cv2.circle(im0, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

               
                await update_line_count(session_id, class_name, track_id, log_file_path, centroid_x)

    cv2.putText(im0, f"Cumulative: {cumulative_count}, Detected: {sessions[session_id]['detected_count']}",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Instance Segmentation Object Tracking", im0)
    cv2.waitKey(1) 

async def video_capture_loop(session_id, video_url, log_file_path):
    await log_event(log_file_path, "Start Stream")
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open video stream")

    frame_index = 0
    video_writer = None
    os.makedirs("saved_video", exist_ok=True)
    video_file_path = f"saved_video/{session_id}.mp4"
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 30)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    try:
        video_writer = cv2.VideoWriter(video_file_path, fourcc, fps, (frame_width, frame_height))
        video_writers[session_id] = video_writer
        while cap.isOpened() and sessions.get(session_id, {}).get("active", False):
            ret, frame = cap.read()
            if not ret:
                break

            await process_frame(session_id, frame, frame_index, log_file_path)
            frame_index += 1

            if video_writer:
                video_writer.write(frame)

            if frame_index % 50 == 0:
                gc.collect()

            await asyncio.sleep(0)

    except Exception as e:
        await log_event(log_file_path, f"Error: {e}")
    finally:
        if video_writer:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        await log_event(log_file_path, "Processing Stopped")

@app.post("/start_detection/")
async def start_detection(request: DetectionRequest):
    session_id, video_url = request.session_id, request.video_url.strip()
    if session_id in sessions and sessions[session_id]["active"]:
        return {"status": "Already running", "count": sessions[session_id]["detected_count"]}
    
    sessions[session_id] = {"active": True, "detected_count": 0}
    asyncio.create_task(video_capture_loop(session_id, video_url, get_log_file_path(session_id)))
    return {"status": "Started", "session_id": session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
