import os
from pathlib import Path


DEFAULT_M20_IP = "10.21.31.103"
DEFAULT_NOS_IP = "10.21.31.106"
DEFAULT_GOS_IP = "10.21.31.104"
DEFAULT_NOS_USER = "user"

M20_IP = os.getenv("M20_IP", DEFAULT_M20_IP)
NOS_IP = os.getenv("M20_NOS_IP", DEFAULT_NOS_IP)
GOS_IP = os.getenv("M20_GOS_IP", DEFAULT_GOS_IP)
NOS_USER = os.getenv("M20_NOS_USER", DEFAULT_NOS_USER)
UDP_PORT = int(os.getenv("M20_UDP_PORT", "30000"))
TCP_PORT = int(os.getenv("M20_TCP_PORT", "30001"))
RTSP_PORT = int(os.getenv("M20_RTSP_PORT", "8554"))
# Optional center WebSocket endpoint for the H.264 camera publisher.  Keep it
# empty by default because the center's actual IP/port/path is deployment
# specific; m20_h264_websocket.py requires an explicit value in publish mode.
H264_WS_SERVER = os.getenv("M20_H264_WS_SERVER", "")
# 使用带 COCO 类别表的本地分割模型。它原生包含 "bottle" 和 "cup"，
# 不依赖 YOLOE 的文本提示词/CLIP 下载。
YOLO_MODEL = os.getenv("M20_YOLO_MODEL", "models/yolo11l-seg.pt")
YOLO_INFERENCE_SIZE = 640  # 双路实时检测的低延迟尺寸；近距离小目标漏检时可改回 960。
YOLO_CONFIDENCE = 0.25  # 低于常规 0.25，以便先显示待人工确认的瓶子候选。
GARBAGE_TARGET_CLASSES = ("bottle", "wine glass", "cup", "bowl",
    "fork", "knife", "spoon",
    "banana", "apple", "orange", "sandwich",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake")  # 当前 COCO 模型能稳定命名的垃圾候选类别。
# False 时检测窗口只绘制上述目标的红框；True 时还会显示其他 COCO 类别的绿框。
SHOW_NON_TARGET_DETECTIONS = False
# 只对接触平地的目标显示单目粗略距离。它不是标定后的三维深度。
SHOW_ROUGH_GROUND_DISTANCE = True

# 官方《相机与视频流》页给出的前相机 Base Link 外参，单位米。
# Base Link 坐标约定：x 向前、y 向左、z 向上。官方没有给出相机姿态矩阵。
ROUGH_CAMERA_OFFSET_X_M = 0.37646
ROUGH_CAMERA_OFFSET_Y_M = 0.0
ROUGH_CAMERA_OFFSET_Z_M = 0.03738

# 前相机粗略地面投影参数：只用于报告，不能替代正式标定。
# 注意：ROUGH_CAMERA_HEIGHT_M 是镜头到“地面”的高度，不能直接用官方的
# OFFSET_Z_M 替代，因为官方 z 是相对 Base Link 的高度，不是离地高度。
ROUGH_CAMERA_HEIGHT_M = 0.40  # 以机器狗实际站立姿态用卷尺测量，单位米。
# 官方未提供相机俯仰/偏航姿态；正前方且水平安装时可暂用 0。
ROUGH_CAMERA_PITCH_DOWN_DEG = 0.0
ROUGH_CAMERA_HFOV_DEG = 90.0  # 官方未提供；优先使用厂家规格、墙面实测或内参标定结果。
ROUGH_CAMERA_YAW_OFFSET_DEG = 0.0
ROUGH_MIN_RANGE_M = 0.4  # 仅报告大于等于此距离的候选，单位米。
ROUGH_MAX_RANGE_M = 6.0  # 仅报告小于等于此距离的候选，单位米。

# Intel RealSense D455 单次塑料瓶定位。D455 的深度和 RGB 帧由设备自身同步，
# 不使用机器狗 RTSP 流或单目估距。输出坐标原点是 D455 彩色镜头中心，轴方向
# 已按机器狗约定转换为 x 向前、y 向左；尚未加入镜头到 base_link 的平移外参。
D455_COLOR_WIDTH = int(os.getenv("M20_D455_COLOR_WIDTH", "848"))
D455_COLOR_HEIGHT = int(os.getenv("M20_D455_COLOR_HEIGHT", "480"))
D455_FPS = int(os.getenv("M20_D455_FPS", "30"))
D455_YOLO_MODEL = os.getenv("M20_D455_YOLO_MODEL", YOLO_MODEL)
D455_YOLO_CONFIDENCE = float(os.getenv("M20_D455_YOLO_CONFIDENCE", "0.30"))
D455_YOLO_INFERENCE_SIZE = int(os.getenv("M20_D455_YOLO_INFERENCE_SIZE", "640"))
D455_TARGET_CLASS = "bottle"
D455_MIN_DEPTH_M = float(os.getenv("M20_D455_MIN_DEPTH_M", "0.20"))
D455_MAX_DEPTH_M = float(os.getenv("M20_D455_MAX_DEPTH_M", "4.00"))
D455_DEPTH_PATCH_RADIUS_PX = int(os.getenv("M20_D455_DEPTH_PATCH_RADIUS_PX", "3"))
D455_MIN_VALID_DEPTH_SAMPLES = int(os.getenv("M20_D455_MIN_VALID_DEPTH_SAMPLES", "5"))
# 当前 D455 安装假设：相机光轴相对机器狗前方固定向下 45 度，无偏航和横滚。
D455_PITCH_DOWN_DEG = float(os.getenv("M20_D455_PITCH_DOWN_DEG", "45.0"))
# D455 镜头中心在 base_link 坐标系中的平移。默认 0 表示直接把相机坐标
# 传入 approach-garbage；要严格按机器狗中心导航，测量后填写这两个值。
D455_BASE_LINK_OFFSET_X_M = float(os.getenv("M20_D455_BASE_LINK_OFFSET_X_M", "0.0"))
D455_BASE_LINK_OFFSET_Y_M = float(os.getenv("M20_D455_BASE_LINK_OFFSET_Y_M", "0.0"))
D455_RESULT_PATH = os.getenv("M20_D455_RESULT_PATH", "runs/d455_bottle/latest_detection.json")
D455_EVIDENCE_PATH = os.getenv("M20_D455_EVIDENCE_PATH", "runs/d455_bottle/latest_detection.jpg")
D455_DISPLAY_SECONDS = float(os.getenv("M20_D455_DISPLAY_SECONDS", "2.0"))

# 所有 approach-garbage 流程在最终趴下前沿机器人当前朝向统一前移的距离。
APPROACH_PRE_LIE_FORWARD_M = float(os.getenv("M20_APPROACH_PRE_LIE_FORWARD_M", "0.15"))

MAPS_ROOT = os.getenv("M20_MAPS_ROOT", "/var/opt/robot/data/maps")
ACTIVE_MAP_PATH = f"{MAPS_ROOT.rstrip('/')}/active"


def rtsp_url(path: str, ip: str = M20_IP, port: int = RTSP_PORT) -> str:
    return f"rtsp://{ip}:{port}/{path.lstrip('/')}"


def require_local_yolo_model(model_path: str) -> str:
    """Reject missing or partial weights instead of blocking on an automatic download."""
    path = Path(model_path).expanduser()
    if not path.is_file() or path.stat().st_size < 100_000:
        raise RuntimeError(
            f"YOLO 权重文件不可用: {path}。当前网络无法稳定访问 GitHub，"
            "请先将完整的 models/yolo11l-seg.pt 放到项目目录，"
            "或在 m20_config.py 的 YOLO_MODEL 中填写本地权重路径。"
        )
    return str(path)
