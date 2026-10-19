# M20 Pro 自动巡检与抓取

本项目运行于 Jetson Orin NX，控制 M20 Pro 机器狗、DM-B601 机械臂、D455 和 D435，执行室内垃圾巡检、抓取与投放。

## 环境准备

仓库不包含本地环境、模型权重和 Jetson wheel：

- `.venv/`、缓存、日志和测试输出；
- `models/*.pt`、`weights/*.ts`；
- `wheelhouse/**/*.whl`。

请先按 [Jetson 依赖说明](docs/jetson_dependencies_zh.md) 准备环境，再将模型放回 `models/` 和 `weights/`。

## 巡检区域

路线由 [config/m20/routes.json](config/m20/routes.json) 管理，路线点坐标由 [config/m20/route_points.json](config/m20/route_points.json) 提供。现场应根据当前地图填写实际坐标。

- `route1`：靠近 3D 打印机的一小块空地。
- `route2`：会议桌周围一圈空地。
- `route3`：靠近实验室介绍公告的 3 张桌子附近，不进入狭窄长廊。
- `route4`：沿 `pointc1` 至 `pointc14` 的路线，已完成路线标注。

## 运行巡检

先执行干运行检查路线、配置、模型和标定文件；该命令不会连接真实设备：

```bash
PYTHONPATH=src .venv/bin/python \
  -m m20_rebot_moving_grasp.route_main \
  --dry-run \
  --route-name route1
```

确认现场安全、设备已连接且 M20 位于 `10.21.31.0/24` 网络后，执行一轮真实巡检：

```bash
PYTHONPATH=src .venv/bin/python \
  -m m20_rebot_moving_grasp.route_main \
  --run-real \
  --route-name route1 \
  --route-loops 1
```

将 `route1` 替换为 `route2`、`route3` 或 `route4` 即可巡检其他区域。常用参数：

- `--route-loops`：路线循环次数，默认 `1`；
- `--route-nav-timeout`：单个路线点导航超时时间，默认 300 秒；
- `--pose-interval`：定位缓存间隔，默认 1 秒；
- `--pose-log`、`--resume-log`：指定定位和中断恢复日志路径。

路线点和投放点使用 `x/y/z/yaw` 坐标导航。`routes.json` 只描述路线顺序，例如：

```json
{"route1": ["point1", "point2"]}
```

## 启动前检查

1. D455、D435 和 DM-B601 已连接 Jetson，镜头方向正确，机械臂、机器狗路线和投放区域无人无障碍。
2. M20 已加载地图并保持有效定位；确认 Jetson 地址为 `10.21.31.50`，M20 AOS 地址为 `10.21.31.103:30000`。
3. 实体急停可立即触及。软件停止不能替代实体急停。
4. 机械臂初始化后应保持 `safe_home`；机器狗到达目标并趴下后才进入抓取姿态。
5. 两台相机应分别连接稳定的 USB 端口。程序会先使用 D455 巡检，再使用 D435 抓取，不能同时采集。

可用以下命令检查网络和 D455：

```bash
ip -4 addr
ping -c 2 10.21.31.103
PYTHONPATH=src .venv/bin/python tests/d455_camera_check.py
PYTHONPATH=src .venv/bin/python tests/d455_camera_yolo_check.py
```

## 任务行为与故障处理

巡检过程中，机器狗在路线点之间扫描 D455；发现目标后会暂停导航、接近目标、趴下并执行抓取和投放，然后返回中断位置继续路线。D435 中心帧和补偿视角均无法得到可行抓取位姿时，程序会回到安全位、记录失败并跳过当前目标。

定位缓存、恢复记录和 D435 失败记录默认写入 `runs/route/`。恢复记录存在时，表示抓取、投放或返回未完整成功；检查现场和日志后再重新启动，不要直接删除记录。

按 `Ctrl+C` 或收到终止信号后，程序会取消导航并请求机器狗趴下，同时停止机械臂任务。发现异常运动、人员进入工作区或无法确认设备状态时，立即使用实体急停。

## 目录与配置

```text
config/m20/routes.json       路线名称和点位顺序
config/m20/route_points.json 路线点与投放点坐标
config/system.yaml           M20 网络和任务参数
config/arm.yaml              DM-B601 参数
config/vision.yaml           D435、YOLO 和手眼标定参数
config/d455.yaml             D455 巡检与接近参数
config/poses.yaml             机械臂姿态参数
src/m20_rebot_moving_grasp/  Python 源码
tests/                        不连接真实硬件的测试
```

开发机可运行单元测试：

```bash
python3 -m unittest discover -s tests -v
```

真实巡检前必须先通过 `--dry-run`，并确认所有路线坐标与当前地图一致。

## 局域网远程服务

前端开发接口和浏览器视频实现请参阅独立文档：[README_frontend_zh.md](README_frontend_zh.md)。

项目提供可由 systemd 开机启动的局域网 WebSocket 服务。服务启动后保持空闲，不会自动让机器狗运动；远程电脑发送启动命令后才会执行巡检。当前只提供一个视频入口 `/all`，其中同时包含 M20 前、后相机画面；不提供单独前相机或后相机入口。D435 仅用于本机抓取视觉，不通过该接口传输。

### Ubuntu 电脑运行

Jetson 服务端和远程 Ubuntu 电脑是两套环境。Jetson 上启动服务后，在远程电脑执行：

```bash
sudo apt install python3-venv ffmpeg
python3 -m venv .venv-client
.venv-client/bin/python -m pip install websockets
PYTHONPATH=. .venv-client/bin/python tools/m20_remote_client.py \
  --host <Jetson可访问的内网IP> --route route1 --loops 1
```

该客户端连接后会立即发送 `start` 并启动真实巡检，关闭客户端不会停止巡检；需要停止时使用 `/control` 发送 `stop`。Ubuntu 电脑不需要安装 YOLO、CUDA、RealSense 或机械臂依赖。

安装依赖并安装服务（将 service 文件中的用户和项目路径改为现场实际值）：

```bash
python -m pip install -r requirements-video.txt
sudo cp systemd/m20-remote.service /etc/systemd/system/m20-remote.service
sudo systemctl daemon-reload
sudo systemctl enable --now m20-remote.service
sudo systemctl status m20-remote.service
journalctl -u m20-remote.service -f
```

服务地址默认为 `ws://<Jetson内网地址>:8765`。控制端连接 `/control`，发送：

```json
{"action":"start","route":"route1","loops":1}
{"action":"status"}
{"action":"stop"}
```

可用路线为 `route1`、`route2`、`route3`、`route4`。远程停止会向巡检进程发送安全停止信号，取消导航并请求机器狗趴下；它不是实体急停。服务当前不提供认证，只应暴露在脑与智能科学中心内网。

简单客户端需要安装 `websockets` 和本机 `ffplay`，会固定打开前、后两个画面并打印状态：

```bash
PYTHONPATH=. python tools/m20_remote_client.py \
  --host 10.21.31.50 \
  --route route1
```

协议和视频服务入口：

- `/control`：JSON 启动、状态和安全停止；
- `/all`：前后相机 H.264 二进制帧，消息头中的 camera ID 区分两个画面。
