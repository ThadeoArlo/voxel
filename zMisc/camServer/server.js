const express = require('express');
const http = require('http');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.static(path.join(__dirname)));

// ESP32-CAM IP addresses (updated to match your network: 10.104.x.x)
const CAM_IPS = [
	'10.104.14.201', // ESP32-CAM 1
	'10.104.14.202', // ESP32-CAM 2
	'10.104.14.203', // ESP32-CAM 3
];

// Proxy stream from ESP32-CAM to client
app.get('/stream/:camId', (req, res) => {
	const camId = parseInt(req.params.camId);
	if (camId < 0 || camId >= CAM_IPS.length) {
		return res.status(404).send('Camera not found');
	}

	const camIP = CAM_IPS[camId];
	const streamUrl = `http://${camIP}/stream`;

	console.log(`Proxying stream from camera ${camId} (${camIP})`);

	// Set headers for streaming
	res.setHeader('Content-Type', 'multipart/x-mixed-replace; boundary=frame');
	res.setHeader('Cache-Control', 'no-cache');
	res.setHeader('Connection', 'keep-alive');

	// Create HTTP request to ESP32-CAM
	const options = {
		hostname: camIP,
		port: 80,
		path: '/stream',
		method: 'GET',
		timeout: 5000,
	};

	const proxyReq = http.request(options, (proxyRes) => {
		// Pipe the response from ESP32-CAM to client
		proxyRes.pipe(res);

		proxyRes.on('error', (err) => {
			console.error(`Stream error for camera ${camId}:`, err.message);
			res.end();
		});
	});

	proxyReq.on('error', (err) => {
		console.error(`Request error for camera ${camId}:`, err.message);
		if (!res.headersSent) {
			res.status(500).send('Camera connection failed');
		} else {
			res.end();
		}
	});

	proxyReq.end();
});

// Get single frame from camera
app.get('/frame/:camId', (req, res) => {
	const camId = parseInt(req.params.camId);
	if (camId < 0 || camId >= CAM_IPS.length) {
		return res.status(404).send('Camera not found');
	}

	const camIP = CAM_IPS[camId];
	const frameUrl = `http://${camIP}/jpg`;

	console.log(`Getting frame from camera ${camId} (${camIP})`);

	const options = {
		hostname: camIP,
		port: 80,
		path: '/jpg',
		method: 'GET',
		timeout: 3000,
	};

	const proxyReq = http.request(options, (proxyRes) => {
		res.setHeader('Content-Type', 'image/jpeg');
		proxyRes.pipe(res);

		proxyRes.on('error', (err) => {
			console.error(`Frame error for camera ${camId}:`, err.message);
			res.end();
		});
	});

	proxyReq.on('error', (err) => {
		console.error(`Request error for camera ${camId}:`, err.message);
		if (!res.headersSent) {
			res.status(500).send('Camera connection failed');
		} else {
			res.end();
		}
	});

	proxyReq.end();
});

// API endpoint to get camera status
app.get('/api/cameras', (req, res) => {
	res.json({
		cameras: CAM_IPS.map((ip, index) => ({
			id: index,
			ip: ip,
			name: `Camera ${index + 1}`,
			streamUrl: `/stream/${index}`,
			frameUrl: `/frame/${index}`,
		})),
	});
});

// Serve dashboard
app.get('/', (req, res) => {
	res.sendFile(path.join(__dirname, 'dashboard.html'));
});

// Start server
app.listen(PORT, () => {
	console.log(`\n🚀 Camera Server running on http://localhost:${PORT}`);
	console.log(`📹 Monitoring ${CAM_IPS.length} cameras:`);
	CAM_IPS.forEach((ip, i) => {
		console.log(`   Camera ${i + 1}: http://${ip}`);
	});
	console.log(`\n📱 Open dashboard: http://localhost:${PORT}\n`);
});

// Graceful shutdown
process.on('SIGINT', () => {
	console.log('\n👋 Shutting down camera server...');
	process.exit(0);
});
