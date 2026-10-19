/* Plain WebCodecs client; serve on localhost for a secure context. */
"use strict";
const $ = id => document.getElementById(id);
const names = ["front", "back"];
const labels = {idle:"空闲",starting:"启动中",running:"运行中",stopping:"停止中",stopped:"已停止",succeeded:"已完成",failed:"失败",unavailable:"路线不可用"};
let control, stream, retry, enabled = false, generation = 0, lastState = null, pending = new Map();
const cameras = names.map(name => ({name, decoder:null, ready:false, sequence:null, sps:null, pps:null, codec:null, description:null, received:0, drawn:0, last:0}));
function eventLog(text) {
  $("events").textContent = `${new Date().toLocaleTimeString()} ${text}\n` + $("events").textContent.slice(0,14000);
}
function notice(text) { $("notice").textContent = text; }
function buttons() {
  const open = control?.readyState === WebSocket.OPEN;
  const busy = ["starting","running","stopping"].includes(lastState);
  $("start").disabled = !open || lastState === null || busy || [...pending.values()].some(p=>p.action==="start");
  $("status").disabled = !open;
  $("stop").disabled = !open || !busy || lastState === "stopping";
}
function send(action) {
  if (control?.readyState !== WebSocket.OPEN) return notice("控制连接未建立。");
  const request = {action, request_id:crypto.randomUUID()};
  if (action === "start") {
    const loops = Number($("loops").value);
    if (!Number.isSafeInteger(loops) || loops < 1) return notice("循环次数必须为大于等于 1 的整数。");
    Object.assign(request, {route:$("route").value, loops});
  }
  const timer = setTimeout(()=> {
    if (pending.delete(request.request_id)) {
      notice("请求响应超时，请查询状态；不要直接重复启动。");
      buttons();
    }
  },10000);
  pending.set(request.request_id, {action,timer});
  control.send(JSON.stringify(request)); eventLog("发送 " + JSON.stringify(request)); buttons();
}
function resetCamera(c, clearParameters = false) {
  if (c.decoder && c.decoder.state !== "closed") c.decoder.close();
  c.decoder = null; c.ready = false;
  if (clearParameters) {
    c.sps = c.pps = c.codec = c.description = c.sequence = null; c.last = 0; c.received = c.drawn = 0;
    const canvas = $(c.name); canvas.getContext("2d").clearRect(0,0,canvas.width,canvas.height);
  }
}
function nalUnits(data) {
  const starts = [];
  for (let i=0;i<data.length-2;i++) {
    if (data[i]===0 && data[i+1]===0) {
      const size = data[i+2]===1 ? 3 : (data[i+2]===0 && data[i+3]===1 ? 4 : 0);
      if (size) { starts.push([i,i+size]); i+=size-1; }
    }
  }
  return starts.map((s,i)=>data.subarray(s[1],i+1<starts.length ? starts[i+1][0] : data.length));
}
function concatenate(parts) {
  const result = new Uint8Array(parts.reduce((n,p)=>n+p.length,0));
  let offset=0; for (const part of parts) {result.set(part,offset);offset+=part.length;} return result;
}
function avcDescription(sps, pps) {
  // AVCDecoderConfigurationRecord (ISO/IEC 14496-15), length-size = 4.
  return concatenate([
    new Uint8Array([1, sps[1], sps[2], sps[3], 0xff, 0xe1,
      (sps.length >>> 8) & 255, sps.length & 255]), sps,
    new Uint8Array([1, (pps.length >>> 8) & 255, pps.length & 255]), pps
  ]);
}
function toAvcSample(units) {
  const size = units.reduce((total, nal) => total + 4 + nal.length, 0);
  const result = new Uint8Array(size);
  const view = new DataView(result.buffer);
  let offset = 0;
  for (const nal of units) {
    view.setUint32(offset, nal.length, false); offset += 4;
    result.set(nal, offset); offset += nal.length;
  }
  return result;
}
function videoMessage(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength <= 24) throw Error("视频消息长度错误");
  const b = new Uint8Array(buffer), v = new DataView(buffer);
  if (![77,50,48,72,50,54,52,0].every((x,i)=>b[i]===x) || b[8]!==1 || b[9]>1) throw Error("视频协议头无效");
  const c = cameras[b[9]], sequence = v.getUint32(12,false);
  c.last = Date.now(); c.received++;
  if (c.sequence !== null && sequence !== ((c.sequence+1)>>>0)) resetCamera(c);
  c.sequence = sequence;
  let data = b.subarray(24);
  const units = nalUnits(data);
  for (const nal of units) {
    if ((nal[0]&31)===7 && nal.length>=4) {
      const codec = "avc1." + [...nal.subarray(1,4)].map(x=>x.toString(16).padStart(2,"0")).join("");
      if (c.sps && (c.sps.length!==nal.length || c.sps.some((x,i)=>x!==nal[i]))) resetCamera(c);
      c.sps = nal.slice(); c.codec = codec;
    }
    if ((nal[0]&31)===8) {
      if (c.pps && (c.pps.length!==nal.length || c.pps.some((x,i)=>x!==nal[i]))) resetCamera(c);
      c.pps = nal.slice();
    }
  }
  const key = units.some(n=>(n[0]&31)===5);
  if (c.decoder?.decodeQueueSize > 6) resetCamera(c);
  if (!c.ready) {
    if (!key || !c.sps || !c.pps) return;
    c.decoder = new VideoDecoder({
      output(frame) {
        try {
          const canvas = $(c.name);
          if (canvas.width!==frame.displayWidth || canvas.height!==frame.displayHeight) {
            canvas.width=frame.displayWidth; canvas.height=frame.displayHeight;
          }
          canvas.getContext("2d").drawImage(frame,0,0); c.drawn++;
        } finally { frame.close(); }
      },
      error(error) { resetCamera(c); eventLog(c.name+" 解码失败："+error.message); }
    });
    c.description = avcDescription(c.sps, c.pps);
    c.decoder.configure({codec:c.codec, description:c.description, optimizeForLatency:true});
    c.ready=true;
    // WebCodecs avc1 input uses AVC length-prefixed samples, not Annex-B.
  }
  c.decoder.decode(new EncodedVideoChunk({type:key?"key":"delta",timestamp:Number(v.getBigUint64(16,false)*1000n),data:toAvcSample(units)}));
}
function closeSockets() {
  generation++; clearTimeout(retry);
  for (const ws of [control,stream]) if (ws) {ws.onclose=null;ws.close();}
  for (const p of pending.values()) clearTimeout(p.timer);
  pending.clear(); cameras.forEach(c=>resetCamera(c,true)); lastState=null; buttons();
}
function connect() {
  closeSockets(); enabled=true;
  let base;
  try {
    const url=new URL($("address").value.trim());
    if (!["ws:","wss:"].includes(url.protocol)) throw Error("请使用 ws:// 或 wss:// 地址");
    base=url.origin;
  } catch (e) {enabled=false;return notice("服务地址无效："+e.message);}
  const epoch=generation;
  $("connection").textContent="连接中";
  control=new WebSocket(base+"/control");
  control.onopen=()=> {$("connection").textContent="控制已连接";notice("已连接；启动巡检会执行真实任务。");send("status");};
  control.onmessage=event=> {
    try {
      const data=JSON.parse(event.data);
      if (data.request_id && pending.has(data.request_id)) {clearTimeout(pending.get(data.request_id).timer);pending.delete(data.request_id);}
      if (data.type==="status") {
        lastState=data.state;
        $("state").textContent=labels[data.state]||data.state;
        $("task").textContent=`路线 ${data.route||"—"} · 总循环 ${data.loops??"—"} · PID ${data.pid??"—"}`;
        $("message").textContent=data.message||""; $("failure").textContent=data.failure||"";
        $("logs").textContent=(data.logs||[]).join("\n");
        if (data.state==="stopped") notice("巡检已停止。");
        if (data.state==="failed") notice("任务或停止清理失败，请检查状态和日志。");
      } else {
        eventLog("接收 "+JSON.stringify(data));
        if (data.ok===false) notice(data.error||"请求失败");
        else if (data.state==="stopping") notice("正在停止，等待设备清理完成；尚未确认已停止。");
      }
      buttons();
    } catch(e) {eventLog("状态消息解析失败："+e.message);}
  };
  control.onerror=()=>notice("控制连接失败，请检查服务地址、端口及浏览器网络权限。");
  control.onclose=()=> {
    if (epoch!==generation) return;
    $("connection").textContent="控制已断开，3 秒后重连"; lastState=null;buttons();
    $("state").textContent="状态未知（连接断开）";
    if (enabled) retry=setTimeout(connect,3000);
  };
  if (!window.isSecureContext || !window.VideoDecoder) {
    cameras.forEach(c=>$(c.name+"Info").textContent="视频不可用：请用支持 WebCodecs 的 Chrome/Edge 打开 http://localhost:8080");
    return;
  }
  function openVideo() {
    if (epoch!==generation || !enabled) return;
    cameras.forEach(c=>resetCamera(c,true));
    stream=new WebSocket(base+"/all"); stream.binaryType="arraybuffer";
    stream.onopen=()=>cameras.forEach(c=>$(c.name+"Info").textContent="视频连接已建立，等待关键帧");
    stream.onmessage=e=> {try {videoMessage(e.data);}catch(error){cameras.forEach(c=>resetCamera(c));eventLog("视频错误："+error.message);}};
    stream.onerror=()=>eventLog("视频连接错误");
    stream.onclose=()=> {if(epoch===generation && enabled) {cameras.forEach(c=>$(c.name+"Info").textContent="视频断开，等待重连");setTimeout(openVideo,3000);}};
  }
  openVideo();
}
setInterval(()=> {
  if (stream?.readyState!==WebSocket.OPEN) return;
  for (const c of cameras) {
    const age=c.last ? Math.floor((Date.now()-c.last)/1000) : null;
    $(c.name+"Info").textContent=`已收 ${c.received} 帧 / 已显示 ${c.drawn} 帧 · `+(age===null?"未收到视频，请检查 RTSP 源":age>3?`${age} 秒未收到新帧`:c.ready?"正在播放":"等待关键帧 / SPS / PPS");
  }
},1000);
$("connect").onclick=connect;
$("disconnect").onclick=()=> {enabled=false;closeSockets();$("connection").textContent="已断开";$("state").textContent="状态未知（连接断开）";notice("已断开连接；不会停止巡检。");cameras.forEach(c=>$(c.name+"Info").textContent="已断开");};
for (const action of ["start","status","stop"]) $(action).onclick=()=>send(action);
window.addEventListener("beforeunload",()=>{enabled=false;closeSockets();});
