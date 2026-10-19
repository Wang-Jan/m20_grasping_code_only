# M20 局域网前端开发说明

本文档面向开发局域网网页前端的人员。前端运行在 Ubuntu 电脑的浏览器中，Jetson 运行项目的 WebSocket 服务。前端不需要安装 YOLO、CUDA、RealSense 或机械臂 SDK。

## 架构与启动

```text
浏览器网页（Ubuntu 本机 http://localhost）
        │ WebSocket
        ├── ws://<192.168.126.176>:8765/control  JSON 控制和状态
        └── ws://<192.168.126.176>:8765/all      前后相机 H.264 二进制帧
                         │
                         ├── route_main 巡检子进程
                         └── M20 前后相机 RTSP
```

Jetson 上启动服务：

```bash
cd /home/g/m20_rebot_moving_grasp
.venv/bin/python -m pip install -r requirements-video.txt
sudo apt install ffmpeg
PYTHONPATH=src .venv/bin/python -m m20_rebot_moving_grasp.remote_service
```

也可以安装开机服务：

```bash
sudo cp systemd/m20-remote.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now m20-remote.service
hostname -I
```

服务默认监听 `0.0.0.0:8765`。网页建议在 Ubuntu 本机以 `http://localhost` 提供；推荐 Chromium/Chrome。当前服务没有认证，只应在脑与智能科学中心内网使用。

## `/control` 控制接口

连接：`ws://<192.168.126.176>:8765/control`。连接后服务立即发送一个状态快照，之后在状态变化时推送状态消息。多个客户端共享同一个巡检任务。

### 启动巡检

请求：

```json
{"action":"start","request_id":"start-001","route":"route1","loops":1}
```

`route` 可写 `route1`、`route2`、`route3`、`route4`。四条路线均已在路线文件中配置。`loops` 必须是大于等于 1 的整数。

成功响应示例：

```json
{"type":"response","request_id":"start-001","ok":true,"state":"starting","route":"route1","loops":1}
```

路线配置缺失时的失败示例（当前 route4 已配置）：

```json
{"type":"response","request_id":"start-001","ok":false,"state":"unavailable","route":"route4","error":"该路线尚未完成路线标注"}
```

同一时间只允许一个巡检进程；重复启动返回 `ok:false` 和“已有巡检任务运行中”。

### 查询状态

请求：

```json
{"action":"status","request_id":"status-001"}
```

当前实现返回 `type:"status"` 的快照，并带上请求中的 `request_id`：

```json
{
  "type":"status",
  "request_id":"status-001",
  "state":"running",
  "route":"route1",
  "loops":1,
  "pid":12345,
  "message":"巡检进行中",
  "failure":null,
  "logs":[],
  "updated_at":1760000000.0
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `state` | `idle`、`starting`、`running`、`stopping`、`succeeded`、`failed`、`stopped`、`unavailable` |
| `route` | 当前或最近一次路线；空闲时可能为 `null` |
| `loops` | 启动时请求的总循环次数，没有当前循环编号字段 |
| `pid` | 当前巡检子进程 PID；进程结束后为 `null` |
| `message` | 最近状态文字或最近一行日志 |
| `failure` | 异常退出原因，正常时为 `null` |
| `logs` | 最近最多 10 条状态快照中的日志行 |
| `updated_at` | 当前快照生成时间，Unix 秒时间戳 |

当前状态是进程生命周期状态，没有“接近、抓取、投放”等结构化阶段事件。断线重连后应重新发送 `status`，不要只依赖断线前缓存。

### 安全停止

请求：

```json
{"action":"stop","request_id":"stop-001"}
```

成功响应只表示服务已经向巡检进程发出 `SIGINT` 安全停止请求：

```json
{"type":"response","request_id":"stop-001","ok":true,"state":"stopping","message":"正在执行安全停止"}
```

随后等待状态推送变为 `stopped`。关闭网页或断开 WebSocket 不会停止巡检；实体急停仍是最高优先级。

### 错误请求

当前只支持 `start`、`status`、`stop`。非法 JSON 或未知 action 返回 `ok:false`。接口没有 REST、登录、token、暂停、继续、路线编辑或参数修改接口。

## `/all` 双路视频接口

连接：`ws://<192.168.126.176>:8765/all`。这是唯一正式视频入口，前后相机必须同时显示；`/front` 和 `/back` 会被拒绝。D435 抓取视频不通过该接口传输。

每条 WebSocket 二进制消息是一帧 H.264 Annex-B access unit，消息头为大端序：

| 偏移 | 长度 | 类型 | 含义 |
| ---: | ---: | --- | --- |
| 0 | 8 | bytes | `M20H264\0` |
| 8 | 1 | uint8 | 协议版本，当前为 `1` |
| 9 | 1 | uint8 | camera ID：`0` 前相机，`1` 后相机 |
| 10 | 2 | uint16 | flags，最低位 `0x0001` 表示关键帧 |
| 12 | 4 | uint32 | 该相机独立递增序号 |
| 16 | 8 | uint64 | Unix 毫秒时间戳 |
| 24 | 剩余 | bytes | H.264 Annex-B access unit |

浏览器不能把这些字节直接设置为 `<video src>`。推荐使用两个 `VideoDecoder` 和两个 `<canvas>`：设置 WebSocket 的 `binaryType="arraybuffer"`，按 camera ID 分流；缓存 SPS/PPS，收到关键帧后创建/重建解码器；丢帧或重连后等待下一个关键帧；绘制完每个 `VideoFrame` 后调用 `close()`。

浏览器端应先检查：

```js
if (!window.VideoDecoder || !window.isSecureContext) {
  throw new Error("当前浏览器环境不支持 WebCodecs 或不是安全上下文");
}
```

普通 `http://localhost` 通常属于安全上下文，但通过内网 IP 的普通 HTTP 页面可能不属于安全上下文。若改用 HTTPS，连接 `ws://` 会触发混合内容限制，应同步使用 WSS 反向代理或在本机 localhost 页面开发。

## 最小前端流程

1. 页面加载后建立 `/control` 和 `/all` 两个 WebSocket。
2. 收到首个 `type:"status"` 快照后更新状态面板。
3. 路线下拉框显示四条可启动路线。
4. 点击启动时发送带唯一 `request_id` 的 `start` 请求。
5. 将 `/all` 中 camera ID 为 0 和 1 的帧分别送入前、后两个解码器和 canvas。
6. 接收状态推送并显示 `state`、`message`、`failure`、最近日志。
7. 点击停止发送 `stop`，等待后续 `stopped` 状态；断线后重新连接并发送 `status`。

可先用仓库内 Python 客户端验证服务：

```bash
sudo apt install python3-venv ffmpeg
python3 -m venv .venv-client
.venv-client/bin/python -m pip install websockets
PYTHONPATH=. .venv-client/bin/python tools/m20_remote_client.py \
  --host <192.168.126.176> --route route1 --loops 1
```

该客户端连接后会立即启动真实巡检，并打开前后两个 `ffplay` 进程；关闭客户端不会停止巡检。网页开发完成后应使用 `/control` 单独发送 `stop`。

## 当前限制与验收

- route4 使用 `pointc1` 至 `pointc14`，启动前可用 `route_main --dry-run --route-name route4` 检查路线和依赖。
- 视频和控制当前均为局域网免认证。
- 本文档不承诺真实局域网连通性、浏览器 H.264 解码和机器狗动作已验收；这些必须在 Jetson、相机、RTSP 服务和实际网络上单独测试。

## 浏览器验证台与停止修复

已提供无需前端构建的双路视频和巡检控制页面，见 [验证台使用说明](m20_client/README.md)。打开页面不会自动启动巡检。

停止请求在动作边界退出，导航等待会检查停止标志；只有清理确认且进程正常退出才返回 `stopped`，清理失败返回 `failed`。更新服务端后需在无任务时重启服务，旧进程不会自动加载修复。
