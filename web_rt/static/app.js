const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const liveLog = document.getElementById('liveLog');
const langSelect = document.getElementById('langSelect');

let ws = null;
let audioCtx = null;
let processor = null;
let sourceNode = null;
let mediaStream = null;
let started = false;

function formatTime(ms) {
  const totalSec = ms / 1000;
  const minutes = Math.floor(totalSec / 60);
  const seconds = totalSec % 60;
  const m = String(minutes).padStart(2, '0');
  const s = seconds.toFixed(2).padStart(5, '0');
  return `${m}:${s}`;
}

function downsampleBuffer(buffer, inputRate, outputRate) {
  if (outputRate === inputRate) return buffer;
  const ratio = inputRate / outputRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offset = 0;
  for (let i = 0; i < newLength; i++) {
    const nextOffset = Math.round((i + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let j = offset; j < nextOffset && j < buffer.length; j++) {
      accum += buffer[j];
      count++;
    }
    result[i] = accum / Math.max(1, count);
    offset = nextOffset;
  }
  return result;
}

function floatTo16BitPCM(float32Array) {
  const out = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function addLogItem(startMs, endMs, text, translation) {
  const item = document.createElement('div');
  item.className = 'log-item';
  const time = document.createElement('div');
  time.className = 'time';
  time.textContent = `${formatTime(startMs)} - ${formatTime(endMs)}`;
  const asr = document.createElement('div');
  asr.className = 'asr';
  asr.textContent = `ASR: ${text}`;
  const trans = document.createElement('div');
  trans.className = 'trans';
  trans.textContent = `TR: ${translation}`;
  item.appendChild(time);
  item.appendChild(asr);
  item.appendChild(trans);
  liveLog.appendChild(item);
  liveLog.scrollTop = liveLog.scrollHeight;
}

function sendConfig() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'config', language: langSelect.value }));
  }
}

async function startMeeting() {
  if (started) return;
  liveLog.innerHTML = '';

  const wsProtocol = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${wsProtocol}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';

  ws.onmessage = (event) => {
    if (typeof event.data !== 'string') return;
    const data = JSON.parse(event.data);
    if (data.type === 'transcript') {
      addLogItem(data.start_ms, data.end_ms, data.text || '', data.translation || '');
    }
  };

  ws.onopen = async () => {
    sendConfig();
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      alert('麦克风权限请求失败，请检查浏览器权限。');
      return;
    }
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioCtx.createMediaStreamSource(mediaStream);

    const bufferSize = 4096;
    processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
    processor.onaudioprocess = (event) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const input = event.inputBuffer.getChannelData(0);
      const downsampled = downsampleBuffer(input, audioCtx.sampleRate, 16000);
      const pcm16 = floatTo16BitPCM(downsampled);
      ws.send(pcm16.buffer);
    };

    sourceNode.connect(processor);
    processor.connect(audioCtx.destination);

    started = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
  };
}

async function stopMeeting() {
  if (!started) return;
  stopBtn.disabled = true;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'end' }));
  }
  if (processor) {
    processor.disconnect();
    processor.onaudioprocess = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
  }
  if (audioCtx) {
    await audioCtx.close();
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
  }
  started = false;
  startBtn.disabled = false;
}

startBtn.addEventListener('click', startMeeting);
stopBtn.addEventListener('click', stopMeeting);
langSelect.addEventListener('change', sendConfig);
