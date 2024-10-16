import torch
import cv2
import numpy as np

# Load MiDaS model for depth estimation
model_type = "DPT_Large"  # Or "MiDaS_small" for faster but less accurate
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

transform = midas_transforms.default_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

def estimate_depth(frame):
    # Convert frame to the correct format
    input_batch = transform(frame).to(device)

    # Estimate depth
    with torch.no_grad():
        prediction = midas(input_batch)

    # Convert the output to a NumPy array
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=frame.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()

    return depth_map

# Example usage in video frame processing:
cap = cv2.VideoCapture('input_videos/demo3.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Estimate depth for the current frame
    depth_map = estimate_depth(frame)

    # Normalize depth map for visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)

    # Show depth map alongside the original frame
    cv2.imshow('Depth Map', depth_map_normalized)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
