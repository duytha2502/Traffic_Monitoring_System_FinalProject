from ultralytics import YOLO
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from main import processVideo, save_processed_video, load_class_names, plot_vehicle_pie_chart, plot_congestion_line_chart, plot_asp_ocp_chart

# Khởi tạo FastAPI app
app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn, hoặc thay bằng một list nguồn cụ thể
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, ...)
    allow_headers=["*"],  # Cho phép tất cả các headers
)

# Load class names
class_file = 'classes_name.txt'
class_names = load_class_names(class_file)

# Đường dẫn lưu trữ video upload và video sau khi xử lý
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'static/tracked_videos/'
CHART_FOLDER = 'charts/'

# Kiểm tra và tạo các folder nếu không tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)

# Tải model (giả sử model đã được lưu với định dạng .h5)
model = YOLO("models/yolov8x/yolov8x.pt")

def allowed_file(filename: str):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'mp4'

@app.post("/track")
async def track(file: UploadFile = File(...)):
    # Kiểm tra định dạng file
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file format. Only .mp4 is allowed.")

    # Lưu file video đã upload
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Đường dẫn video đã xử lý
    processed_video_path = os.path.join(PROCESSED_FOLDER, file.filename)

    # Gọi hàm track_video để xử lý video
    processed_video, vehicle_count, congestion_rate, average_speed, occupancy = processVideo(file_path, model, class_names)

    # Lưu video đã xử lí
    saved_video, output_name = save_processed_video(processed_video, file_path, PROCESSED_FOLDER)

    # Save charts
    pie_file_path = plot_vehicle_pie_chart(vehicle_count, output_name, CHART_FOLDER)
    line_file_path = plot_congestion_line_chart(congestion_rate, output_name, CHART_FOLDER)
    asp_ocp_file_path = plot_asp_ocp_chart(average_speed, occupancy, output_name, CHART_FOLDER)
    
    # Trả về file video đã tracking
    return {"message": "Tracking completed", "video_url": f"{saved_video}", "pie_url": f"{pie_file_path}", "line_url": f"{line_file_path}", "hybrid_url": f"{asp_ocp_file_path}"}

# Route để lấy video đã xử lý
@app.get("/static/tracked_videos/{filename}")
def get_processed_video(filename: str):
    video_path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    else:
        raise HTTPException(status_code=404, detail="File not found")

# Route để lấy danh sách các file trong thư mục tương ứng với video
@app.get("/charts/{video_folder}")
async def list_charts_in_video(video_folder: str):
    video_folder_path = os.path.join(CHART_FOLDER, video_folder)
    if os.path.exists(video_folder_path) and os.path.isdir(video_folder_path):
        try:
            # Lấy danh sách file trong thư mục video_folder (chỉ các file hình ảnh)
            image_files = [f for f in os.listdir(video_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            return {"images": image_files}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=404, detail="Video folder not found")

# Route để lấy file ảnh cụ thể từ thư mục video
@app.get("/charts/{video_folder}/{filename}")
async def get_chart_from_video(video_folder: str, filename: str):
    file_path = os.path.join(CHART_FOLDER, video_folder, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")