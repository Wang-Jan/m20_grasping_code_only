# M20 浏览器验证台

在控制电脑（和 Jetson 同一局域网）运行，推荐 Chrome / Edge。

```bash
cd ~/m20_client
python3 serve_frontend.py
```

如果直接在本项目目录使用：

```bash
cd /home/g/m20_rebot_moving_grasp
python3 m20_client/serve_frontend.py
```

在**运行上述命令的电脑**打开 `http://localhost:8080`。不需要 npm，也不需要安装前端依赖。`localhost` 可提供 WebCodecs 所需的安全上下文；不要改成 `http://<局域网IP>:8080`。浏览器询问本地网络访问权限时，允许访问 Jetson。

1. 填写 `ws://192.168.126.176:8765`，点击“连接”。连接不会启动任务。
2. 检查状态面板和前后画面。每路显示收到帧数、已显示帧数以及是否等待关键帧。
3. 选择 `route1` / `route2` / `route3` / `route4` 和循环次数，点击“启动巡检”才会执行真实任务。
4. 点击“查询状态”可手动同步状态；平时服务端会主动推送。
5. 点击“停止巡检”，等待状态从 `stopping` 变为 `stopped`。若变成 `failed`，查看日志和设备状态，不能当作已安全停止。
6. 断线会自动重连并查询状态，**不会重发启动请求**。点击“断开”不会停止巡检。

停止不是立即断电：导航等待会检查停止标志，单次机械臂动作先执行到动作边界，再取消导航、请求趴下并关闭设备。停止耗时受当前动作和设备通信超时影响。

## 修复服务端后如何生效

原来的 `stop` JSON 不需要更改。旧代码只在 D455 接近路径检查停止标志，路线巡航主循环及导航等待没有检查，可能一直运行。更新后必须确保 Jetson 服务加载新代码；**已有旧巡检进程不会因文件更新自动修复**。

在确认当前没有运行中的任务后，在 Jetson 执行：

```bash
sudo systemctl restart m20-remote.service
systemctl status m20-remote.service --no-pager
```

`/control` 只管理该服务启动的巡检进程。如果巡检是在另一个终端直接运行 `route_main` / `main.py` 启动的，服务会回复“当前没有运行中的巡检”；应在原终端停止该任务，后续统一从页面启动。

## 黑屏排查

- “控制已连接”不代表视频源可用；视频是独立连接。
- “未收到视频”时在 Jetson 查看 `journalctl -u m20-remote.service -n 50 --no-pager`。
- 若日志出现 `RTSP ... 404 Not Found`，表示相机地址没有提供对应视频流，需检查机器狗相机服务和 RTSP 地址；修改前端不能修复源端 404。
- 已收帧数增长但未显示：检查浏览器 WebCodecs / H.264 支持和页面底部解码错误；使用 Chrome / Edge 的 localhost 页面。
- 如果曾看到 `A key frame is required after configure()`，请刷新验证页面使用最新的 `app.js`。页面会把服务端 Annex-B 帧转换成带 `avcC description` 的 AVC 样本后再交给 WebCodecs，并在丢帧后等待下一个关键帧。
- 两路均通过 `/all`，不要改为 `/front`、`/back`。

原有 `tools/m20_remote_client.py` 仍是会自动启动巡检的旧命令行客户端；仅查看视频和状态请使用本验证台。
