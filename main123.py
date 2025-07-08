# import cv2
# import pandas as pd
# from ultralytics import solutions

# id = pd.Series(dtype=int)
# tracked_ids = []
# class_mapping = {
#     "bale1": 1,
#     "bag5": 5,
#     "tbale2": 2
# }

# cap = cv2.VideoCapture("rtmp://localhost/live/stream4")
# assert cap.isOpened(), "Error reading video file"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# region_points = [(650, 10), (650, 650)]

# video_writer = cv2.VideoWriter("object_counting_output.mp4", cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

# counter = solutions.ObjectCounter(
#     show=False,
#     region=region_points,
#     model="vchanel/jlobb/js.pt",
#     classes=[0, 1, 2, 3],
#     show_in=True,
#     show_out=True,
#     line_width=2,
# )

# # Process video
# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break

#     im0 = counter.count(im0)
#     # print(len(counter.counted_ids))
#     cc = counter.classwise_counts
#     df = pd.DataFrame.from_dict(cc)

#     total_sum = 0  # Default value if no valid calculation is made
#     if df.shape[0] > 1:
#         test = df.iloc[1].to_dict()

#         series = pd.Series(test)
#         mapped_values = series.index.map(lambda x: class_mapping.get(x, 0))
#         result = series.values * mapped_values
#         df2 = pd.DataFrame({'Class': series.index, 'Value': series.values, 'Mapped': mapped_values, 'Result': result})
#         total_sum = df2["Result"].sum()

#     # Display total_sum on the frame
#     text = f"trolly_count{len(counter.counted_ids)} - object count: {total_sum} "
#     # Define text parameters
#     # text = f"Total Sum: {total_sum}"
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     font_thickness = 2
#     text_color = (0, 255, 0)  # Green
#     bg_color = (0, 0, 0)  # Black background
#     position = (50, 500)

#     # Get text size
#     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

#     # Define the rectangle background coordinates
#     x, y = position
#     rect_top_left = (x - 10, y - text_height - 10)
#     rect_bottom_right = (x + text_width + 10, y + baseline + 10)

#     # Draw the rectangle (background)
#     cv2.rectangle(im0, rect_top_left, rect_bottom_right, bg_color, -1)

#     # Draw the text on top of the background
#     cv2.putText(im0, text, position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)


#     # Show the frame live
#     cv2.imshow("Frame", im0)

#     # Write the frame to the output video
#     video_writer.write(im0)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()


# import cv2
# import pandas as pd
# from ultralytics import solutions

# # Object ID tracking
# id = pd.Series(dtype=int)
# tracked_ids = []

# # Class mapping
# class_mapping = {
#     "bale1": 1,
#     "bag5": 5,
#     "tbale2": 2
# }

# # Open video stream
# cap = cv2.VideoCapture("rtmp://localhost/live/stream4")
# assert cap.isOpened(), "Error reading video file"

# # Get video properties
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
#                                        cv2.CAP_PROP_FRAME_HEIGHT,
#                                        cv2.CAP_PROP_FPS))

# # Ensure FPS is a valid number
# fps = max(fps, 10)  # If FPS is 0, set it to 10

# # Debugging output
# print(f"Video Resolution: {w}x{h}, FPS: {fps}")

# # Define region for counting
# region_points = [(650, 10), (650, 650)]

# # Initialize ObjectCounter
# counter = solutions.ObjectCounter(
#     show=False,
#     region=region_points,
#     model="vchanel/fastback/js.pt",
#     classes=[0, 1, 2, 3],
#     show_in=True,
#     show_out=True,
#     line_width=2,
# )

# # Set video writer
# video_writer = cv2.VideoWriter("object_counting_output.avi",  # Change extension if needed
#                                cv2.VideoWriter_fourcc(*'XVID'),  # Try 'XVID' or 'MJPG' for better support
#                                fps, (w, h))

# # Verify if VideoWriter is working
# if not video_writer.isOpened():
#     print("Error: VideoWriter not initialized. Check codec or permissions.")
#     cap.release()
#     exit()

# # Process video
# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break

#     # Apply object counting
#     im0 = counter.count(im0)
    
#     # Check if frame is valid
#     if im0 is None:
#         print("Warning: Received an empty frame.")
#         continue

#     # Get classwise count
#     cc = counter.classwise_counts
#     df = pd.DataFrame.from_dict(cc)

#     total_sum = 0
#     if df.shape[0] > 1:
#         test = df.iloc[1].to_dict()
#         series = pd.Series(test)
#         mapped_values = series.index.map(lambda x: class_mapping.get(x, 0))
#         result = series.values * mapped_values
#         df2 = pd.DataFrame({'Class': series.index, 'Value': series.values, 'Mapped': mapped_values, 'Result': result})
#         total_sum = df2["Result"].sum()

#     # Display total count on frame
#     text = f"Trolly Count {len(counter.counted_ids)} - Object Count: {total_sum}"
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     font_thickness = 2
#     text_color = (0, 255, 0)  # Green
#     bg_color = (0, 0, 0)  # Black background
#     position = (50, 500)

#     # Get text size
#     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

#     # Define the rectangle background coordinates
#     x, y = position
#     rect_top_left = (x - 10, y - text_height - 10)
#     rect_bottom_right = (x + text_width + 10, y + baseline + 10)

#     # Draw the rectangle background
#     cv2.rectangle(im0, rect_top_left, rect_bottom_right, bg_color, -1)

#     # Draw the text on top of the background
#     cv2.putText(im0, text, position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

#     # Show the frame live
#     cv2.imshow("Frame", im0)

#     # Save frame to video
#     video_writer.write(im0)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()
# print("Video saved successfully!")


import cv2
import pandas as pd
from ultralytics import solutions
from datetime import datetime
import os
id = pd.Series(dtype=int)
tracked_ids = []
class_mapping = {"bale": 1,
    "bag": 5,
    "tbale": 2,
    "sbale":2,
    "m_box":6,
    "fbale":2,
    "c_box":5
}

cap = cv2.VideoCapture("rtmp://localhost/live/stream4")

if not cap.isOpened():
    print("Error: Cannot open video stream.")
    exit()
saved_video_dir = "march_videos"
os.makedirs(saved_video_dir, exist_ok=True)
timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
video_file_path = os.path.join(saved_video_dir, f"output_jlNewf{timestamp}.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_file_path, fourcc, fps, (frame_width, frame_height))
region_points = [(650, 10), (650, 650)]

counter = solutions.ObjectCounter(
    show=False,
    region=region_points,
    model="jl_obb_l_org.pt",
    classes=[0, 1, 2, 3],
    # show_in=True,
    # show_out=True,
    line_width=0,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing completed.")
        break

    im0 = counter.count(im0)
    cc = counter.classwise_counts
    df = pd.DataFrame.from_dict(cc)

    total_sum = 0  # Default if no valid calculation
    if df.shape[0] > 1:
        test = df.iloc[1].to_dict()
        series = pd.Series(test)
        mapped_values = series.index.map(lambda x: class_mapping.get(x, 0))
        result = series.values * mapped_values
        df2 = pd.DataFrame({'Class': series.index, 'Value': series.values, 'Mapped': mapped_values, 'Result': result})
        total_sum = df2["Result"].sum()

    # Display total_sum on frame
    text = f"Trolley Count: {len(counter.counted_ids)} - Object Count: {total_sum}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 255, 0)
    bg_color = (0, 0, 0)
    position = (50, 500)

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    x, y = position
    rect_top_left = (x - 10, y - text_height - 10)
    rect_bottom_right = (x + text_width + 10, y + baseline + 10)

    # cv2.rectangle(im0, rect_top_left, rect_bottom_right, bg_color, -1)
    # cv2.putText(im0, text, position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Show frame
    re=cv2.resize(im0,(640,640))
    cv2.imshow("Frame", re)

    # Save frame to video
    video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
