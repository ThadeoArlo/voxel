# ESP32-CAM Multi-Camera Setup Guide

## Overview

This setup creates a web dashboard that displays live feeds from 3 ESP32-CAM
devices in a grid layout.

## Hardware Requirements

- 3x ESP32-CAM modules (AI-Thinker with OV2640)
- 3x MicroSD cards (optional, for storage)
- 3x USB-C cables for programming
- WiFi network

## Software Setup

### 1. ESP32-CAM Configuration

#### Update WiFi Credentials

Edit `cam/cam.ino` and update these lines:

```cpp
const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";
```

#### Upload Code to ESP32-CAMs

1. Open Arduino IDE
2. Install ESP32 board package:
   - File → Preferences → Additional Board Manager URLs
   - Add:
     `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Tools → Board → Boards Manager → Search "ESP32" → Install
3. Select board: Tools → Board → ESP32 Arduino → AI Thinker ESP32-CAM
4. Upload `cam/cam.ino` to each ESP32-CAM
5. Note the IP addresses shown in Serial Monitor

### 2. Server Setup

#### Install Dependencies

```bash
cd camServer
npm install
```

#### Configure Camera IPs

Edit `camServer/server.js` and update the CAM_IPS array:

```javascript
const CAM_IPS = [
	'192.168.1.100', // Replace with actual ESP32-CAM 1 IP
	'192.168.1.101', // Replace with actual ESP32-CAM 2 IP
	'192.168.1.102', // Replace with actual ESP32-CAM 3 IP
];
```

#### Start Server

```bash
npm start
```

## Usage

### 1. Power On ESP32-CAMs

- Connect ESP32-CAMs to power (5V via USB or external supply)
- Wait for WiFi connection (LED should stabilize)
- Check Serial Monitor for IP addresses

### 2. Start Server

```bash
cd camServer
npm start
```

### 3. Open Dashboard

- Navigate to: `http://localhost:3000`
- You should see a 3-camera grid layout
- Cameras appear automatically when they come online

## Features

### Dashboard Features

- **Live Grid**: 3-camera feed in responsive grid
- **Auto-Discovery**: Cameras appear automatically when powered on
- **Status Indicators**: Online/offline status with visual indicators
- **Individual Controls**: Refresh/toggle stream per camera
- **Mobile Responsive**: Works on phones/tablets

### Camera Features

- **MJPEG Stream**: `/stream` endpoint for live video
- **Single Frame**: `/jpg` endpoint for still images
- **Auto-Reconnect**: Handles camera disconnections gracefully
- **Error Handling**: Shows offline status when cameras disconnect

## Troubleshooting

### ESP32-CAM Issues

1. **No WiFi Connection**:

   - Check SSID/password in code
   - Verify WiFi signal strength
   - Check Serial Monitor for error messages

2. **Camera Not Initializing**:

   - Verify camera module is properly connected
   - Check if PSRAM is detected (affects resolution)
   - Try restarting ESP32-CAM

3. **Poor Video Quality**:
   - Adjust `jpeg_quality` in camera config (lower = better quality)
   - Check WiFi bandwidth
   - Reduce frame rate by increasing delay in stream loop

### Server Issues

1. **Cameras Not Appearing**:

   - Verify IP addresses in `server.js`
   - Check if ESP32-CAMs are on same network
   - Test individual camera URLs: `http://CAM_IP/stream`

2. **Stream Not Loading**:

   - Check browser console for errors
   - Verify CORS is enabled
   - Try refreshing individual cameras

3. **Performance Issues**:
   - Reduce camera resolution in ESP32 code
   - Increase delay between frames
   - Check network bandwidth

## Network Configuration

### Finding ESP32-CAM IPs

1. **Serial Monitor**: Check Arduino IDE Serial Monitor
2. **Router Admin**: Check connected devices in router settings
3. **Network Scanner**: Use apps like Fing or Advanced IP Scanner
4. **mDNS**: Access via `http://esp32cam.local` (if supported)

### Port Configuration

- **ESP32-CAM**: Port 80 (default)
- **Server**: Port 3000 (configurable in server.js)
- **Dashboard**: Accessible via `http://localhost:3000`

## Advanced Configuration

### Camera Settings

Modify in `cam/cam.ino`:

```cpp
config.frame_size = FRAMESIZE_VGA;  // Resolution: QVGA, VGA, SVGA, etc.
config.jpeg_quality = 10;           // Quality: 0-63 (lower = better)
```

### Server Settings

Modify in `camServer/server.js`:

```javascript
const PORT = 3000; // Change server port
const threshold = 50; // Distance threshold for arrow visibility
```

## Next Steps

- Add recording functionality
- Implement motion detection
- Add camera controls (brightness, contrast)
- Create mobile app
- Add cloud storage integration
