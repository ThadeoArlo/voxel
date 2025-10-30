			import * as THREE from 'three';
			import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';

			// HUD collapse/expand
			const hudEl = document.getElementById('hud');
			const toggleBtn = document.getElementById('toggleHud');
			if (toggleBtn && hudEl) {
				toggleBtn.onclick = () => {
					hudEl.classList.toggle('collapsed');
		toggleBtn.textContent = hudEl.classList.contains('collapsed') ? 'Show' : 'Hide';
				};
			}

			// Cameras overlay wiring
			const camOverlay = document.getElementById('camOverlay');
			const camGrid = document.getElementById('camGrid');
			const showCamerasBtn = document.getElementById('showCameras');
			const closeCamOverlayBtn = document.getElementById('closeCamOverlay');
			async function refreshCameras() {
				if (!camGrid) return;
				camGrid.innerHTML = '';
				try {
					const res = await fetch('/api/cameras');
					const cams = (await res.json()) || [];
					for (const c of cams) {
						const tile = document.createElement('div');
						tile.className = 'cam-tile';
						const title = document.createElement('div');
						title.className = 'cam-title';
						title.textContent = `${c.name ?? 'Camera'} (${c.idx})`;
						tile.appendChild(title);
						const img = document.createElement('img');
						img.className = 'cam-video';
						img.src = `/stream/${c.idx}`;
						img.loading = 'eager';
						img.decoding = 'async';
						tile.appendChild(img);
						camGrid.appendChild(tile);
					}
				} catch (e) {
					const err = document.createElement('div');
					err.style.color = '#f66';
					err.textContent = 'Failed to load cameras';
					camGrid.appendChild(err);
				}
			}
			function openCamOverlay() {
				if (!camOverlay) return;
				camOverlay.style.display = 'flex';
				refreshCameras();
			}
			function closeCamOverlay() {
				if (!camOverlay) return;
				if (camGrid) camGrid.innerHTML = '';
				camOverlay.style.display = 'none';
			}
			if (showCamerasBtn) showCamerasBtn.onclick = openCamOverlay;
			if (closeCamOverlayBtn) closeCamOverlayBtn.onclick = closeCamOverlay;

// Scene setup (centimeter units)
			THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
			const scene = new THREE.Scene();
			scene.background = new THREE.Color(0x0f0f12);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 1, 1e7);
			const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
			renderer.setSize(innerWidth, innerHeight);
			document.body.appendChild(renderer.domElement);

			const controls = new OrbitControls(camera, renderer.domElement);
			controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.target.set(0, 0, 200);

const GRID_SIZE = 6000; // 60m square expressed in cm
const GRID_DIVISIONS = 120; // 50cm grid cells
const grid = new THREE.GridHelper(GRID_SIZE, GRID_DIVISIONS, 0x666666, 0x333333);
			grid.rotateX(Math.PI / 2);
			scene.add(grid);

			const plane = new THREE.Mesh(
	new THREE.PlaneGeometry(GRID_SIZE, GRID_SIZE),
	new THREE.MeshBasicMaterial({ color: 0x0a3a6a, transparent: true, opacity: 0.08 })
			);
			scene.add(plane);

function makeTextSprite(text, color, worldHeight = 18) {
				const dpr = Math.max(1, window.devicePixelRatio || 1);
	const padX = 4;
	const padY = 3;
				const cvs = document.createElement('canvas');
				const ctx = cvs.getContext('2d');
				ctx.font = '12px system-ui';
				const textW = ctx.measureText(text).width;
				const w = Math.ceil(textW + padX * 2);
				const h = 16 + padY * 2;
				cvs.width = w * dpr;
				cvs.height = h * dpr;
				ctx.scale(dpr, dpr);
				ctx.font = '12px system-ui';
	ctx.fillStyle = 'rgba(0,0,0,0.6)';
				ctx.fillRect(0, 0, w, h);
				ctx.fillStyle = new THREE.Color(color).getStyle();
				ctx.textBaseline = 'middle';
				ctx.fillText(text, padX, h * 0.5);
				const tex = new THREE.CanvasTexture(cvs);
				tex.minFilter = THREE.LinearFilter;
				tex.magFilter = THREE.LinearFilter;
				tex.generateMipmaps = false;
				const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
				const spr = new THREE.Sprite(mat);
				const aspect = w / h;
				const heightWorld = worldHeight;
				const widthWorld = heightWorld * aspect;
				spr.scale.set(widthWorld, heightWorld, 1);
				return spr;
			}

function addAxis(dir, color, label) {
	const ARROW_LENGTH = 400;
	const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), ARROW_LENGTH, color, 80, 40);
	scene.add(arrow);
	const sprite = makeTextSprite(label, color, 22);
	sprite.position.copy(dir.clone().multiplyScalar(ARROW_LENGTH + 40));
	scene.add(sprite);
}
addAxis(new THREE.Vector3(1, 0, 0), 0xff5555, 'X (cm)');
addAxis(new THREE.Vector3(0, 1, 0), 0x55ff55, 'Y (cm)');
addAxis(new THREE.Vector3(0, 0, 1), 0x5599ff, 'Z (cm)');

camera.position.set(0, -1800, 900);
camera.lookAt(0, 0, 200);

// Track playback state
			let fileTracks = [];
			let fileMeta = [];
			let fileActors = [];
			let filePathLines = [];
			let fileIdx = 0;
			let fileMaxLen = 0;
			let filePlaybackFps = 30;
			let fileAcc = 0;
			let filePlaying = false;

const palette = [
	0x55ffcc,
	0x33ddaa,
	0x88ffee,
	0x22aa88,
	0xbbffee,
	0x99ffdd,
	0xffdd55,
	0xffaa88,
	0xaaddff,
];
function nextColor(idx) {
	return palette[idx % palette.length];
}

function clearFileActors() {
	for (const m of fileActors) if (m && m.parent) m.parent.remove(m);
				fileActors = [];
	for (const l of filePathLines) if (l && l.parent) l.parent.remove(l);
				filePathLines = [];
			}

			function ensureFileActors() {
				while (fileActors.length < fileTracks.length) {
					const idx = fileActors.length;
					const color = fileMeta[idx]?.color || 0x55ffcc;
		const mesh = new THREE.Mesh(
			new THREE.SphereGeometry(12, 20, 16),
						new THREE.MeshBasicMaterial({ color })
					);
		scene.add(mesh);
		fileActors.push(mesh);
				}
				for (let i = 0; i < fileActors.length; i++) {
		const visible = Boolean(fileMeta[i]?.visible);
		fileActors[i].visible = visible;
		if (fileMeta[i]?.color && fileActors[i]?.material) {
						fileActors[i].material.color = new THREE.Color(fileMeta[i].color);
		}
				}
			}

			function buildFilePaths() {
	for (const l of filePathLines) if (l && l.parent) l.parent.remove(l);
				filePathLines = [];
				const show = document.getElementById('showPaths')?.checked;
				if (!show) return;
				for (let i = 0; i < fileTracks.length; i++) {
		const track = fileTracks[i];
		if (!track || track.length < 2) continue;
					if (!fileMeta[i]?.visible) continue;
		const points = track.map((p) => new THREE.Vector3(p.x, p.y, p.z));
		const geom = new THREE.BufferGeometry().setFromPoints(points);
		const mat = new THREE.LineBasicMaterial({
			color: fileMeta[i]?.color || 0x55ffcc,
			linewidth: 2,
		});
					const line = new THREE.Line(geom, mat);
					scene.add(line);
					filePathLines.push(line);
				}
			}

			function rebuildTrackListUI() {
				const list = document.getElementById('trackList');
				if (!list) return;
				list.innerHTML = '';
				for (let i = 0; i < fileTracks.length; i++) {
					const row = document.createElement('div');
					row.style.display = 'flex';
					row.style.alignItems = 'center';
					row.style.marginTop = '4px';
					const sw = document.createElement('span');
					sw.className = 'swatch';
		sw.style.background = new THREE.Color(fileMeta[i]?.color || 0x55ffcc).getStyle();
					row.appendChild(sw);
					const label = document.createElement('span');
					label.textContent = fileMeta[i]?.name || `Track ${i + 1}`;
					label.style.marginRight = '8px';
					row.appendChild(label);
					const vis = document.createElement('input');
					vis.type = 'checkbox';
					vis.checked = Boolean(fileMeta[i]?.visible);
					vis.onchange = () => {
						fileMeta[i].visible = vis.checked;
						ensureFileActors();
						buildFilePaths();
					};
					row.appendChild(vis);
					const rm = document.createElement('button');
					rm.textContent = 'Remove';
					rm.style.marginLeft = '6px';
					rm.onclick = () => {
			if (fileActors[i] && fileActors[i].parent) fileActors[i].parent.remove(fileActors[i]);
						fileActors.splice(i, 1);
			if (filePathLines[i] && filePathLines[i].parent) filePathLines[i].parent.remove(filePathLines[i]);
						filePathLines.splice(i, 1);
						fileTracks.splice(i, 1);
						fileMeta.splice(i, 1);
						fileMaxLen = fileTracks.reduce((m, t) => Math.max(m, t.length), 0);
						rebuildTrackListUI();
						ensureFileActors();
						buildFilePaths();
					};
					row.appendChild(rm);
					list.appendChild(row);
				}
			}

			function addTracksFromData(name, data) {
				let tracks = [];
				if (Array.isArray(data)) {
					tracks = [data];
				} else if (data && Array.isArray(data.tracks)) {
		tracks = data.tracks.map((t) => (Array.isArray(t) ? t : t.points || []));
				} else {
					return false;
				}
				for (let k = 0; k < tracks.length; k++) {
					const idx = fileTracks.length;
					fileTracks.push(tracks[k]);
					fileMeta.push({
						name: tracks.length > 1 ? `${name}[${k}]` : name,
						color: nextColor(idx),
						visible: true,
					});
				}
				fileMaxLen = fileTracks.reduce((m, t) => Math.max(m, t.length), 0);
	fileIdx = 0;
	fileAcc = 0;
	filePlaying = false;
				ensureFileActors();
				buildFilePaths();
				rebuildTrackListUI();
				return true;
			}

			function flash(color = '#ffffff') {
				const f = document.getElementById('flash');
	if (!f) return;
				f.style.background = color;
				f.classList.add('on');
				setTimeout(() => f.classList.remove('on'), 180);
			}

function updateStopButton() {
					const stopBtn = document.getElementById('stop');
	if (!stopBtn) return;
	stopBtn.textContent = filePlaying ? 'Pause' : 'Resume';
}

function startPlayback() {
	if (fileTracks.length === 0) return;
					fileIdx = 0;
					fileAcc = 0;
					filePlaying = true;
	updateStopButton();
	ensureFileActors();
	flash('#ffffff');
}

function togglePlayback() {
	if (fileTracks.length === 0) return;
	filePlaying = !filePlaying;
	updateStopButton();
	if (filePlaying) flash('#66ff66');
}

const cameraIndicators = [];
function clearCameraIndicators() {
	for (const entry of cameraIndicators) {
		if (!entry) continue;
		if (entry.arrow && entry.arrow.parent) entry.arrow.parent.remove(entry.arrow);
		if (entry.label && entry.label.parent) entry.label.parent.remove(entry.label);
		if (entry.icon && entry.icon.parent) entry.icon.parent.remove(entry.icon);
	}
	cameraIndicators.length = 0;
}

function addCameraIndicator(cam) {
	if (!cam) return;
	const pos = new THREE.Vector3(
		Number(cam.position?.[0] || 0),
		Number(cam.position?.[1] || 0),
		Number(cam.position?.[2] || 0)
	);
	const dir = new THREE.Vector3(
		Number(cam.forward?.[0] || 0),
		Number(cam.forward?.[1] || 0),
		Number(cam.forward?.[2] || 0)
	).normalize();
	const ARROW_LENGTH = 300;
	const arrow = new THREE.ArrowHelper(dir, pos, ARROW_LENGTH, 0xffaa33, 60, 30);
	scene.add(arrow);
	const icon = new THREE.Mesh(
		new THREE.ConeGeometry(18, 40, 20),
		new THREE.MeshBasicMaterial({ color: 0xffaa33 })
	);
	icon.position.copy(pos);
	icon.up.set(0, 0, 1);
	icon.lookAt(pos.clone().add(dir));
	scene.add(icon);
	const label = makeTextSprite(cam.name || 'camera', 0xffaa33, 20);
	label.position.copy(pos.clone().add(new THREE.Vector3(0, 0, 60)));
	scene.add(label);
	cameraIndicators.push({ arrow, label, icon });
}

async function loadCameraConfig() {
	try {
		const res = await fetch('/render/cam_config.json', { cache: 'no-store' });
		if (!res.ok) return;
		const json = await res.json();
		const cams = Array.isArray(json?.cameras) ? json.cameras : Array.isArray(json) ? json : [];
		const test1Cams = cams
			.filter((c) => typeof c?.name === 'string' && c.name.toLowerCase().startsWith('test1_'))
			.sort((a, b) => a.name.localeCompare(b.name));
		clearCameraIndicators();
		if (test1Cams.length >= 3) {
			for (const cam of test1Cams.slice(0, 3)) addCameraIndicator(cam);
		} else {
			// fallback: use first three cameras if available
			for (const cam of cams.slice(0, 3)) addCameraIndicator(cam);
		}
	} catch (e) {
		console.warn('Failed to load camera config', e);
	}
}

			function animate() {
				requestAnimationFrame(animate);
	const now = performance.now();
	const dt = animate.prev ? (now - animate.prev) / 1000 : 0;
	animate.prev = now;

				if (fileTracks.length > 0) {
					for (let i = 0; i < fileTracks.length; i++) {
			const track = fileTracks[i];
			if (!track || track.length === 0) continue;
			const idx = Math.min(fileIdx, track.length - 1);
			const p = track[idx];
						if (p && fileActors[i]) fileActors[i].position.set(p.x, p.y, p.z);
					}
		if (filePlaying && fileMaxLen > 0) {
			const step = 1 / Math.max(1, filePlaybackFps);
						fileAcc += dt;
						while (fileAcc >= step) {
							fileAcc -= step;
				if (fileIdx < fileMaxLen - 1) {
					fileIdx += 1;
					} else {
					filePlaying = false;
					updateStopButton();
					flash('#ff3355');
					break;
				}
			}
		}
	}

				controls.update();
				renderer.render(scene, camera);
			}
			animate();

// UI wiring
const fileInput = document.getElementById('file');
if (fileInput) {
	fileInput.addEventListener('change', (ev) => {
				const files = ev.target.files;
				if (!files || files.length === 0) return;
				for (let i = 0; i < files.length; i++) {
					const f = files[i];
					const reader = new FileReader();
					reader.onload = () => {
						try {
							const data = JSON.parse(reader.result);
							if (!addTracksFromData(f.name.replace(/\.json$/i, ''), data)) {
						alert(`Unsupported track format in ${f.name}`);
							}
						} catch (e) {
					alert(`Bad JSON in ${f.name}`);
						}
					};
					reader.readAsText(f);
				}
		// allow selecting same file again later
		fileInput.value = '';
	});
}

const startBtn = document.getElementById('start');
if (startBtn) startBtn.onclick = () => startPlayback();

			const stopBtn = document.getElementById('stop');
			if (stopBtn) {
	stopBtn.onclick = () => togglePlayback();
	updateStopButton();
}

const showPathsChk = document.getElementById('showPaths');
if (showPathsChk) {
	showPathsChk.addEventListener('change', () => buildFilePaths());
}

			addEventListener('resize', () => {
				camera.aspect = innerWidth / innerHeight;
				camera.updateProjectionMatrix();
				renderer.setSize(innerWidth, innerHeight);
			});

			addEventListener('keydown', (ev) => {
				if (ev.code === 'Space' && !ev.metaKey && !ev.ctrlKey && !ev.altKey) {
					const tag = (ev.target && ev.target.tagName) || '';
		if (!['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(tag)) {
						ev.preventDefault();
			togglePlayback();
					}
				}
			});

loadCameraConfig();
		
