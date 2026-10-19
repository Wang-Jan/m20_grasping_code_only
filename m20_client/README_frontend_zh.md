# M20 局域网客户端运行与前端开发指南

本文档介绍如何在 Ubuntu 控制电脑上运行 Python 客户端，以及如何开发连接 Jetson 的局域网网页前端。

当前目录提供 Python 远程控制客户端和视频协议工具，**尚未提供可直接运行的网页前端**。Python 客户端用于启动巡检、查看前后相机视频和打印状态；网页开发者可使用本文中的 WebSocket 接口实现相同功能。

## 1. 系统架构与运行条件

```text
Ubuntu 控制电脑
├── Python 客户端 + 两个 ffplay 视频窗口
└── 浏览器网页（开发后通过 http://localhost 提供）
        │ WebSocket
        ├── ws://192.168.126.176:8765/control  JSON 控制和状态
        └── ws://192.168.126.176:8765/all      前后相机 H.264 二进制帧
                         │
                      Jetson 服务
                         ├── route_main 巡检子进程
                         └── M20 前后相机 RTSP
```

控制电脑与 Jetson 应处于可互通的局域网。本文使用 Jetson 地址 `192.168.126.176`、端口 `8765`；如果部署地址不同，请统一替换命令和代码中的地址。

- **控制电脑**：需要 Python 3、`websockets` 和 FFmpeg（包含 `ffplay`），显示视频时需要桌面图形环境；不需要安装 YOLO、CUDA、RealSense 或机械臂 SDK。
- **Jetson**：需要部署完整的机器人服务端项目及其运行依赖，并能访问机器人和相机。服务端代码不在当前客户端目录中。
- **网页前端**：推荐 Chromium/Chrome，需支持 WebCodecs，并运行在安全上下文中。

当前服务没有认证，仅应在脑与智能科学中心内网使用。多个客户端共享同一个巡检任务。

## 2. 项目文件

| 文件 | 用途 |
| --- | --- |
| `tools/m20_remote_client.py` | 连接控制与视频接口，启动巡检并显示前后视频 |
| `tools/m20_h264_websocket.py` | H.264 WebSocket 协议及视频工具，客户端使用其消息解析函数 |
| `m20_config.py` | 视频工具使用的配置模块 |
| `requirements-video.txt` | Python 依赖，包含 `websockets>=10.4` |
| `README_frontend_zh.md` | 本文档 |

复制项目到其他电脑时，应保留上述文件及目录结构。`__pycache__` 中的 `.pyc` 文件是 Python 自动生成的缓存，不需要复制或手动修改。

## 3. 启动 Jetson 服务

以下命令在 **Jetson 上**执行，假设完整服务端项目位于 `/home/g/m20_rebot_moving_grasp`，且已准备好 `.venv` 环境及机器人运行依赖：

```bash
cd /home/g/m20_rebot_moving_grasp
.venv/bin/python -m pip install -r requirements-video.txt
sudo apt install ffmpeg
PYTHONPATH=src .venv/bin/python -m m20_rebot_moving_grasp.remote_service
```

服务默认监听 `0.0.0.0:8765`。可使用 `hostname -I` 查看 Jetson 的 IP。

也可以使用服务端项目提供的 systemd 配置安装开机服务：

```bash
cd /home/g/m20_rebot_moving_grasp
sudo cp systemd/m20-remote.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now m20-remote.service
```

手动启动与 systemd 启动选择一种，避免重复占用端口。使用 systemd 时可检查状态和日志：

```bash
sudo systemctl status m20-remote.service
sudo journalctl -u m20-remote.service -f
```

## 4. 准备控制电脑

以下命令均在 **Ubuntu 控制电脑上**执行。

### 4.1 安装依赖

```bash
cd ~/m20_client
sudo apt update
sudo apt install python3-venv ffmpeg netcat-openbsd

python3 -m venv .venv-client
source .venv-client/bin/activate
python3 -m pip install -r requirements-video.txt
```

已有可用虚拟环境时，无需重新创建。以后每次打开新终端，只需：

```bash
cd ~/m20_client
source .venv-client/bin/activate
```

也可以直接使用 `.venv-client/bin/python` 运行脚本，无需激活环境。

### 4.2 检查网络

```bash
ping -c 3 192.168.126.176
nc -vz 192.168.126.176 8765
```

`ping` 用于检查主机可达性，`nc` 用于检查服务端口。如果端口连接失败，请检查 Jetson IP、服务是否启动，以及网络和防火墙是否允许访问 `8765`。

### 4.3 查询状态，不启动巡检

先执行只读状态查询，确认控制接口正常：

```bash
.venv-client/bin/python - <<'PY'
import asyncio
import json
import websockets

JETSON_IP = "192.168.126.176"

async def main():
    async with websockets.connect(f"ws://{JETSON_IP}:8765/control") as ws:
        print("初始状态:", await ws.recv())
        await ws.send(json.dumps({
            "action": "status",
            "request_id": "status-001"
        }))
        async for message in ws:
            print(message)
            data = json.loads(message)
            if data.get("type") == "status" and data.get("request_id") == "status-001":
                break

asyncio.run(main())
PY
```

收到带 `request_id: "status-001"` 的状态 JSON，说明控制接口查询成功；视频和机器人动作仍需单独验证。

## 5. 运行与停止巡检

### 5.1 启动一轮巡检

**该命令会立即启动真实巡检，请先确认现场安全。** 在控制电脑的桌面终端执行：

```bash
cd ~/m20_client
.venv-client/bin/python tools/m20_remote_client.py \
  --host 192.168.126.176 \
  --route route1 \
  --loops 1
```

客户端连接 `/control` 和 `/all`，打开前、后两个 `ffplay` 视频窗口，发送启动请求，并在终端打印服务端状态。

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--host` | Jetson IP 或主机名，必填 | 无 |
| `--port` | WebSocket 服务端口 | `8765` |
| `--route` | 巡检路线 | `route1` |
| `--loops` | 巡检总循环次数，应为大于等于 1 的整数 | `1` |
| `--player` | 视频播放器可执行文件 | `ffplay` |

可运行路线为 `route1`、`route2`、`route3`、`route4`。`route4` 使用 `pointc1` 至 `pointc14` 的已标注点位。

### 5.2 停止巡检

**关闭视频窗口、退出客户端或断开 WebSocket 都不会停止巡检。** 需要停止时，在控制电脑另开终端执行：

```bash
cd ~/m20_client
.venv-client/bin/python - <<'PY'
import asyncio
import json
import websockets

JETSON_IP = "192.168.126.176"

async def main():
    async with websockets.connect(f"ws://{JETSON_IP}:8765/control") as ws:
        print("初始状态:", await ws.recv())
        await ws.send(json.dumps({
            "action": "stop",
            "request_id": "stop-001"
        }))
        async for message in ws:
            print(message)
            data = json.loads(message)
            if data.get("type") == "response" and data.get("request_id") == "stop-001":
                if not data.get("ok"):
                    print("停止请求未成功，请根据响应检查当前状态。")
                    break
            if data.get("type") == "status" and data.get("state") == "stopped":
                print("巡检已停止。")
                break

asyncio.run(main())
PY
```

`state: "stopping"` 只表示已发出停止请求，应继续等待状态变为 `stopped`。上述脚本会持续监听；若任务已结束或服务端未推送 `stopped`，请结合返回状态判断，并使用第 4.3 节重新查询。实体急停优先于远程停止。

建议操作顺序：检查网络 → 查询状态 → 确认现场安全 → 启动巡检 → 观察双路视频与状态 → 发送停止请求并确认结果。

## 6. 网页前端接入

网页需要自行实现，通过以下两个 WebSocket 与 Jetson 通信：

| 接口 | 地址 | 用途 |
| --- | --- | --- |
| 控制接口 | `ws://192.168.126.176:8765/control` | JSON 启动、停止和状态消息 |
| 视频接口 | `ws://192.168.126.176:8765/all` | 前后相机混合 H.264 二进制消息 |

建议在 Ubuntu 本机通过 `http://localhost` 提供网页，并使用 Chromium/Chrome。普通 `http://localhost` 通常属于安全上下文，通过内网 IP 提供的普通 HTTP 页面可能不属于安全上下文。

如果网页改为 HTTPS，连接 `ws://` 会触发混合内容限制，需要同步配置 WSS 反向代理。

## 7. `/control` 控制协议

连接后，服务立即发送一个状态快照，之后在状态变化时推送状态消息。建议每次请求携带唯一的 `request_id`，用于关联响应。

### 7.1 启动巡检

请求：

```json
{"action":"start","request_id":"start-001","route":"route1","loops":1}
```

`route` 可写 `route1`、`route2`、`route3`、`route4`；`loops` 必须是大于等于 1 的整数。

成功响应示例：

```json
{"type":"response","request_id":"start-001","ok":true,"state":"starting","route":"route1","loops":1}
```

`route4` 已配置，启动成功响应示例：

```json
{"type":"response","request_id":"start-001","ok":true,"state":"starting","route":"route4","loops":1}
```

同一时间只允许一个巡检进程；重复启动返回 `ok:false` 和“已有巡检任务运行中”。

### 7.2 查询与接收状态

请求：

```json
{"action":"status","request_id":"status-001"}
```

服务返回 `type:"status"` 的快照，并带上请求中的 `request_id`：

```json
{
  "type": "status",
  "request_id": "status-001",
  "state": "running",
  "route": "route1",
  "loops": 1,
  "pid": 12345,
  "message": "巡检进行中",
  "failure": null,
  "logs": [],
  "updated_at": 1760000000.0
}
```

| 字段 | 说明 |
| --- | --- |
| `state` | `idle`、`starting`、`running`、`stopping`、`succeeded`、`failed`、`stopped`、`unavailable` |
| `route` | 当前或最近一次路线；空闲时可能为 `null` |
| `loops` | 启动时请求的总循环次数，没有当前循环编号字段 |
| `pid` | 当前巡检子进程 PID；进程结束后为 `null` |
| `message` | 最近状态文字或最近一行日志 |
| `failure` | 异常退出原因，正常时为 `null` |
| `logs` | 状态快照中的最近最多 10 条日志行 |
| `updated_at` | 当前快照生成时间，Unix 秒时间戳 |

当前状态表示进程生命周期，没有“接近、抓取、投放”等结构化阶段事件。断线重连后应重新发送 `status`，不要只依赖断线前缓存。

### 7.3 安全停止

请求：

```json
{"action":"stop","request_id":"stop-001"}
```

成功响应只表示服务已向巡检进程发出 `SIGINT` 安全停止请求：

```json
{"type":"response","request_id":"stop-001","ok":true,"state":"stopping","message":"正在执行安全停止"}
```

前端应继续接收状态推送，等待 `stopped`，不能收到成功响应后立即显示“已停止”。关闭网页或断开连接不会停止巡检。

### 7.4 错误与接口范围

当前只支持 `start`、`status`、`stop`。非法 JSON 或未知 `action` 返回 `ok:false`。

接口没有 REST、登录、token、暂停、继续、路线编辑或参数修改功能。

## 8. `/all` 双路视频协议

`/all` 是唯一正式视频入口，前后相机必须同时显示；`/front` 和 `/back` 会被拒绝。D435 抓取视频不通过该接口传输。

每条 WebSocket 二进制消息是一帧 H.264 Annex-B access unit，消息头使用大端序：

| 偏移 | 长度 | 类型 | 含义 |
| ---: | ---: | --- | --- |
| 0 | 8 | bytes | `M20H264\0` |
| 8 | 1 | uint8 | 协议版本，当前为 `1` |
| 9 | 1 | uint8 | camera ID：`0` 前相机，`1` 后相机 |
| 10 | 2 | uint16 | flags，最低位 `0x0001` 表示关键帧 |
| 12 | 4 | uint32 | 该相机独立递增序号 |
| 16 | 8 | uint64 | Unix 毫秒时间戳 |
| 24 | 剩余 | bytes | H.264 Annex-B access unit |

浏览器不能把这些字节直接设置为 `<video src>`。推荐使用两个 `VideoDecoder` 和两个 `<canvas>`：

1. 设置 WebSocket 的 `binaryType = "arraybuffer"`。
2. 校验消息头，按 camera ID 将帧分流到前后相机解码器。
3. 缓存 SPS/PPS，收到关键帧后创建或重建解码器。
4. 丢帧或重连后等待下一个关键帧。
5. 绘制完每个 `VideoFrame` 后调用 `close()`，释放资源。

浏览器端应先检查环境：

```js
if (!window.VideoDecoder || !window.isSecureContext) {
  throw new Error("当前浏览器环境不支持 WebCodecs 或不是安全上下文");
}
```

## 9. 最小前端实现流程

1. 页面加载后建立 `/control` 和 `/all` 两个 WebSocket。
2. 收到首个 `type:"status"` 快照后更新状态面板。
3. 路线下拉框显示四条可启动路线。
4. 用户点击启动时发送带唯一 `request_id` 的 `start` 请求，处理成功和失败响应。
5. 将 camera ID 为 `0` 和 `1` 的视频帧分别送入前、后解码器和 canvas。
6. 持续显示 `state`、`message`、`failure` 和最近日志。
7. 用户点击停止时发送 `stop`，显示停止中，等待后续 `stopped` 状态。
8. 断线后重连并发送 `status`，恢复当前状态显示；视频重新等待关键帧。

开发前可先按第 4、5 节使用 Python 客户端验证服务。注意该客户端没有“只看视频不启动”的参数，运行后会发送真实巡检启动请求。

## 10. 常见问题与验收

| 现象 | 检查方向 |
| --- | --- |
| 连接拒绝或超时 | Jetson IP、`8765` 端口、服务进程、局域网和防火墙 |
| `No module named websockets` | 使用 `.venv-client/bin/python`，并在该环境安装 `requirements-video.txt` |
| 找不到 `ffplay` 或无法打开视频窗口 | 安装 `ffmpeg`，确认在桌面图形环境运行 |
| 启动路线返回 `unavailable` | 检查服务端 `routes.json` 是否包含该路线且列表非空；当前配置包含四条路线 |
| 提示已有巡检任务运行中 | 先查询共享任务状态，需要时发送停止请求 |
| 关闭客户端后机器人仍在运行 | 断开客户端不会停止巡检，按第 5.2 节发送 `stop` |
| 浏览器无 `VideoDecoder` | 检查浏览器支持情况和安全上下文，优先使用 Chromium/Chrome 与 `http://localhost` |
| HTTPS 网页无法连接 `ws://` | 配置 WSS，避免混合内容限制 |

实际部署时，应分别验证网络和控制接口、前后视频、启动及停止流程、断线重连后的状态恢复。本文档不代表真实局域网连通性、浏览器 H.264 解码或机器狗动作已完成验收；这些需要在 Jetson、相机、RTSP 服务和实际网络上测试。

## 浏览器验证台与停止修复

已提供无需前端构建的双路视频和巡检控制页面，见 [验证台使用说明](README.md)。打开页面不会自动启动巡检。

停止请求在动作边界退出，导航等待会检查停止标志；只有清理确认且进程正常退出才返回 `stopped`，清理失败返回 `failed`。更新服务端后需在无任务时重启服务，旧进程不会自动加载修复。
