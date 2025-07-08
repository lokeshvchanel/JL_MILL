
# import cv2
# import pandas as pd
# from ultralytics import solutions
# from datetime import datetime
# import os
# id = pd.Series(dtype=int)
# tracked_ids = []
# class_mapping = {"bale": 1,
#     "bag": 5,
#     "tbale": 2,
#     "sbale":2,
#     "m_box":6,
#     "fbale":2,
#     "c_box":5
# }

# cap = cv2.VideoCapture("rtmp://localhost/live/stream4")

# if not cap.isOpened():
#     print("Error: Cannot open video stream.")
#     exit()
# saved_video_dir = "March_videos"
# os.makedirs(saved_video_dir, exist_ok=True)
# timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
# video_file_path = os.path.join(saved_video_dir, f"output_jlNewf{timestamp}.mp4")
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 30
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_writer = cv2.VideoWriter(video_file_path, fourcc, fps, (frame_width, frame_height))
# region_points = [(650, 10), (650, 650)]

# counter = solutions.ObjectCounter(
#     show=False,
#     region=region_points,
#     model="jlobb.pt",
#     classes=[0,1,2,3,4,5,6],
#     show_in=True,
#     show_out=True,
#     line_width=0,
#     conf=0.9
# )


# while cap.isOpened():
#     success, im0 = cap.read()
    
#     if not success:
#         print("Video frame is empty or processing completed.")
#         break

#     im0 = counter.count(im0)
#     cc = counter.classwise_counts
#     df = pd.DataFrame.from_dict(cc)

#     total_sum = 0  # Default if no valid calculation
#     if df.shape[0] > 1:
#         test = df.iloc[1].to_dict()
#         series = pd.Series(test)
#         mapped_values = series.index.map(lambda x: class_mapping.get(x, 0))
#         result = series.values * mapped_values
#         df2 = pd.DataFrame({'Class': series.index, 'Value': series.values, 'Mapped': mapped_values, 'Result': result})
#         total_sum = df2["Result"].sum()

#     # Display total_sum on frame
#     text = f"Trolley Count: {len(counter.counted_ids)} - Object Count: {total_sum}"
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     font_thickness = 2
#     text_color = (0, 255, 0)
#     bg_color = (0, 0, 0)
#     position = (50, 500)

#     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
#     x, y = position
#     rect_top_left = (x - 10, y - text_height - 10)
#     rect_bottom_right = (x + text_width + 10, y + baseline + 10)

#     # cv2.rectangle(im0, rect_top_left, rect_bottom_right, bg_color, -1)
#     # cv2.putText(im0, text, position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

#     # Show frame
#     re=cv2.resize(im0,(640,640))
#     cv2.imshow("Frame", re)

#     # Save frame to video
#     video_writer.write(im0)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()
# ===============================================

# import cv2
# import pandas as pd
# from ultralytics import solutions
# from datetime import datetime
# import os

# id = pd.Series(dtype=int)
# tracked_ids = []
# class_mapping = {
#     "bale": 1,
#     "bag": 5,
#     "tbale": 2,
#     "sbale": 2,
#     "m_box": 6,
#     "fbale": 2,
#     "c_box": 5
# }

# cap = cv2.VideoCapture("rtmp://localhost/live/stream4")

# if not cap.isOpened():
#     print("Error: Cannot open video stream.")
#     exit()

# saved_video_dir = "April_videos"
# os.makedirs(saved_video_dir, exist_ok=True)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# video_file_path = os.path.join(saved_video_dir, f"output_jlNewf{timestamp}.mp4")
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 30
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_writer = cv2.VideoWriter(video_file_path, fourcc, fps, (frame_width, frame_height))
# region_points = [(650, 10), (650, 650)]

# counter = solutions.ObjectCounter(
#     show=False,
#     region=region_points,
#     model="jlobb.pt",
#     classes=[0, 1, 2, 3, 4, 5, 6],
#     show_in=True,
#     show_out=True,
#     line_width=0,
#     conf=0.9
# )

# while cap.isOpened():
#     success, im0 = cap.read()
    
#     if not success:
#         print("Video frame is empty or processing completed.")
#         break
    
#     # Duplicate frame
#     im0copy = im0.copy()
    
#     im0 = counter.count(im0)
#     cc = counter.classwise_counts
#     df = pd.DataFrame.from_dict(cc)
    
#     total_sum = 0  # Default if no valid calculation
#     if df.shape[0] > 1:
#         test = df.iloc[1].to_dict()
#         series = pd.Series(test)
#         mapped_values = series.index.map(lambda x: class_mapping.get(x, 0))
#         result = series.values * mapped_values
#         df2 = pd.DataFrame({'Class': series.index, 'Value': series.values, 'Mapped': mapped_values, 'Result': result})
#         total_sum = df2["Result"].sum()
    
#     # Display total_sum on frame
#     text = f"Trolley Count: {len(counter.counted_ids)} - Object Count: {total_sum}"
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     font_thickness = 2
#     text_color = (0, 255, 0)
#     position = (50, 500)
    
#     cv2.putText(im0, text, position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
#     # Show frame
#     re = cv2.resize(im0, (640, 640))
#     cv2.imshow("Frame", re)
    
#     # Save duplicated frame to video
#     video_writer.write(im0copy)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()
# ==================check+++++++++++++++++

import cv2
import pandas as pd
from ultralytics import solutions
from datetime import datetime
import os
import smtplib
from email.message import EmailMessage
import torch
import signal
import sys

# === Configuration ===
EMAIL_ADDRESS = "lokeshchinnadurai5@gmail.com"
EMAIL_PASSWORD = "vhrn rdgv uyvx vawq"  # App password from Google
RECEIVER_EMAIL = "growth@vchanel.com"
VIDEO_SOURCE = "rtmp://localhost/live/stream4"

# === Class Mapping for Weighted Totals ===
class_mapping = {
    "bale": 1,
    "bag": 5,
    "tbale": 2,
    "sbale": 2,
    "m_box": 6,
    "fbale": 2,
    "c_box": 5
}

# === Setup Paths and Video Writer ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
saved_video_dir = "April_videos"
os.makedirs(saved_video_dir, exist_ok=True)
video_file_path = os.path.join(saved_video_dir, f"output_jlNewf{timestamp}.mp4")
report_file_path = os.path.join(saved_video_dir, f"report_{timestamp}.txt")

# === Open Video Stream ===
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: Cannot open video stream.")
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_file_path, fourcc, fps, (frame_width, frame_height))

# === Setup YOLO Counter ===
region_points = [(650, 10), (650, 650)]
print("CUDA available:", torch.cuda.is_available())
print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

counter = solutions.ObjectCounter(
    show=False,
    region=region_points,
    model="jlobb.pt",
    classes=[0, 1, 2, 3, 4, 5, 6],
    show_in=True,
    show_out=True,
    line_width=0,
    conf=0.9
)

if torch.cuda.is_available():
    counter.model.to('cuda')
if hasattr(counter.model, 'device'):
    print("Model is running on:", counter.model.device)

# === Global Report Storage ===
df2 = pd.DataFrame()
report = ""
send_email_on_exit = True  # Avoid multiple sends

# === Email Sending Function ===
def send_email(report):
    global send_email_on_exit
    if not send_email_on_exit:
        return
    send_email_on_exit = False

    msg = EmailMessage()
    msg['Subject'] = 'Detailed Object Counting Report'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECEIVER_EMAIL
    msg.set_content(f"""Hello,

The video stream has ended. Below is the final report:

{report}

Regards,
Object Counter System
""")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# === Exit Handler ===
def exit_handler(sig=None, frame=None):
    print("\n[INFO] Program interrupted. Generating report and sending email...")
    finalize_and_send_report()
    sys.exit(0)

signal.signal(signal.SIGINT, exit_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, exit_handler)  # kill

# === Finalize Report and Send Email ===
def finalize_and_send_report():
    global report

    report = f"Object Counting Report - {timestamp}\n\n"
    if not df2.empty:
        report += "Object Counts:\n\n"
        for _, row in df2.iterrows():
            report += f"- {row['Class']}: {int(row['Count'])} × {row['Mapped Value']} = {int(row['Result'])}\n"
        report += f"\nTotal Object Count (Weighted): {int(df2['Result'].sum())}\n"
    else:
        report += "No objects were detected.\n"

    report += f"Trolley Count (Unique IDs): {len(counter.counted_ids)}\n"

    # Save to text file
    with open(report_file_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_file_path}")

    # Send email
    send_email(report)

# === Main Processing Loop ===
try:
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing completed.")
            break

        im0copy = im0.copy()
        im0 = counter.count(im0)
        cc = counter.classwise_counts
        df = pd.DataFrame.from_dict(cc)

        # Weighted count logic
        if df.shape[0] > 1:
            test = df.iloc[1].to_dict()
            series = pd.Series(test)
            mapped_values = series.index.map(lambda x: class_mapping.get(x, 0))
            result = series.values * mapped_values
            df2 = pd.DataFrame({
                'Class': series.index,
                'Count': series.values,
                'Mapped Value': mapped_values,
                'Result': result
            })
            total_sum = int(df2["Result"].sum())
        else:
            total_sum = 0

        # Overlay on frame
        text = f"Trolley Count: {len(counter.counted_ids)} - Object Count: {total_sum}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im0, text, (50, 500), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        re = cv2.resize(im0, (640, 640))
        cv2.imshow("Frame", re)

        # Save original frame
        video_writer.write(im0copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exit key pressed.")
            break

finally:
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    finalize_and_send_report()
