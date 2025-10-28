			import * as THREE from 'three';
			import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';

			// HUD collapse/expand
			const hudEl = document.getElementById('hud');
			const toggleBtn = document.getElementById('toggleHud');
			if (toggleBtn && hudEl) {
				toggleBtn.onclick = () => {
					hudEl.classList.toggle('collapsed');
					toggleBtn.textContent = hudEl.classList.contains('collapsed')
						? 'Show'
						: 'Hide';
				};
			}

			// Persist UI state across reloads
			const LS_KEY_SETUP3 = 'vp_setup3';
			const LS_KEY_SETUP4 = 'vp_setup2';
			function persistSetup(key, obj) {
				try {
					localStorage.setItem(key, JSON.stringify(obj));
				} catch (e) {}
			}
			function restoreSetup(key) {
				try {
					const s = localStorage.getItem(key);
					return s ? JSON.parse(s) : null;
				} catch (e) {
					return null;
				}
			}
			function mergeInto(dst, src) {
				if (!src || typeof src !== 'object') return;
				for (const k of Object.keys(src)) {
					if (k in dst && typeof src[k] === 'number') dst[k] = src[k];
				}
			}

			// Slider metadata and clamping for easier input
			const setup1ParamMeta = {
				heightZ: { min: 0, max: 2000, step: 1 },
				angleBetweenDeg: { min: 0, max: 120, step: 1 }, // equal spacing max 120
				distanceToCenter: { min: 1, max: 5000, step: 1 },
				rotationDeg: { min: 0, max: 360, step: 1 },
			};
			const setup2ParamMeta = {
				heightZ: { min: 0, max: 2000, step: 1 },
				distanceBetween: { min: 0, max: 5000, step: 1 },
				flareAngleDeg: { min: -90, max: 90, step: 1 },
				distanceToCenter: { min: 1, max: 5000, step: 1 },
				rotationDeg: { min: 0, max: 360, step: 1 },
			};
			function clampValue(meta, v) {
				if (!meta) return Number(v);
				let x = Number(v);
				if (Number.isNaN(x)) x = meta.min ?? 0;
				const min = meta.min ?? -Infinity;
				const max = meta.max ?? Infinity;
				x = Math.min(max, Math.max(min, x));
				if (meta.step != null) {
					const step = Number(meta.step) || 1;
					const base = Number.isFinite(min) ? min : 0;
					x = base + Math.round((x - base) / step) * step;
					x = Math.min(max, Math.max(min, x));
					const decimals = (String(step).split('.')[1] || '').length;
					x = decimals > 0 ? Number(x.toFixed(decimals)) : Math.round(x);
				}
				return x;
			}

			// Cache File System Access handles across reloads
			function openDb() {
				return new Promise((resolve) => {
					const req = indexedDB.open('vp_db', 1);
					req.onupgradeneeded = () => {
						const db = req.result;
						if (!db.objectStoreNames.contains('handles'))
							db.createObjectStore('handles');
					};
					req.onsuccess = () => resolve(req.result);
					req.onerror = () => resolve(null);
				});
			}
			async function dbGetHandle(name) {
				const db = await openDb();
				if (!db) return null;
				return new Promise((resolve) => {
					const tx = db.transaction('handles');
					const os = tx.objectStore('handles');
					const rq = os.get(name);
					rq.onsuccess = () => resolve(rq.result || null);
					rq.onerror = () => resolve(null);
				});
			}
			async function dbSetHandle(name, handle) {
				const db = await openDb();
				if (!db) return;
				return new Promise((resolve) => {
					const tx = db.transaction('handles', 'readwrite');
					const os = tx.objectStore('handles');
					os.put(handle, name);
					tx.oncomplete = () => resolve();
					tx.onerror = () => resolve();
				});
			}

			THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
			const scene = new THREE.Scene();
			scene.background = new THREE.Color(0x0f0f12);
			const camera = new THREE.PerspectiveCamera(
				60,
				innerWidth / innerHeight,
				0.1,
				1e7
			);
			const renderer = new THREE.WebGLRenderer({ antialias: true });
			renderer.setSize(innerWidth, innerHeight);
			document.body.appendChild(renderer.domElement);
			const controls = new OrbitControls(camera, renderer.domElement);
			controls.enableDamping = true;

			const grid = new THREE.GridHelper(2000, 80, 0x666666, 0x333333);
			grid.rotateX(Math.PI / 2);
			scene.add(grid);
			const plane = new THREE.Mesh(
				new THREE.PlaneGeometry(2000, 2000),
				new THREE.MeshBasicMaterial({
					color: 0x0a3a6a,
					transparent: true,
					opacity: 0.08,
				})
			);
			scene.add(plane);
			function addAxis(dir, color, label) {
				const a = new THREE.ArrowHelper(
					dir,
					new THREE.Vector3(0, 0, 0),
					300,
					color,
					60,
					30
				);
				scene.add(a);
				const sprite = makeTextSprite(label, color);
				sprite.position.copy(dir.clone().multiplyScalar(320));
				scene.add(sprite);
			}
			function makeTextSprite(text, color, worldHeight = 14) {
				const dpr = Math.max(1, window.devicePixelRatio || 1);
				const padX = 4,
					padY = 3;
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
				ctx.fillStyle = 'rgba(0,0,0,0.5)';
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
			addAxis(new THREE.Vector3(1, 0, 0), 0xff5555, 'X');
			addAxis(new THREE.Vector3(0, 1, 0), 0x55ff55, 'Y');
			addAxis(new THREE.Vector3(0, 0, 1), 0x5599ff, 'Z');

			camera.position.set(0, -900, 200);
			camera.lookAt(0, 0, 300);

			// removed standalone dot: reconstruction uses fileActors, simulation uses simActors

			// --- Loaded track playback (supports single or multiple tracks) ---
			let fileTracks = [];
			let fileMeta = [];
			let fileActors = [];
			let filePathLines = [];
			let fileIdx = 0;
			let fileMaxLen = 0;
			let filePlaybackFps = 30;
			let fileAcc = 0;
			let filePlaying = false;
			function clearFileActors() {
				for (const m of fileActors) {
					if (m && m.parent) m.parent.remove(m);
				}
				fileActors = [];
				for (const l of filePathLines) {
					if (l && l.parent) l.parent.remove(l);
				}
				filePathLines = [];
			}
			function ensureFileActors() {
				while (fileActors.length < fileTracks.length) {
					const idx = fileActors.length;
					const color = fileMeta[idx]?.color || 0x55ffcc;
					const m = new THREE.Mesh(
						new THREE.SphereGeometry(8, 16, 12),
						new THREE.MeshBasicMaterial({ color })
					);
					scene.add(m);
					fileActors.push(m);
				}
				for (let i = 0; i < fileActors.length; i++) {
					fileActors[i].visible = Boolean(fileMeta[i]?.visible);
					if (fileMeta[i]?.color && fileActors[i]?.material)
						fileActors[i].material.color = new THREE.Color(fileMeta[i].color);
				}
			}

			function buildFilePaths() {
				// Remove existing
				for (const l of filePathLines) {
					if (l && l.parent) l.parent.remove(l);
				}
				filePathLines = [];
				const show = document.getElementById('showPaths')?.checked;
				if (!show) return;
				for (let i = 0; i < fileTracks.length; i++) {
					const tr = fileTracks[i];
					if (!tr || tr.length < 2) continue;
					if (!fileMeta[i]?.visible) continue;
					const pts = [];
					for (const p of tr) {
						pts.push(new THREE.Vector3(p.x, p.y, p.z));
					}
					const geom = new THREE.BufferGeometry().setFromPoints(pts);
					const color = fileMeta[i]?.color || 0x55ffcc;
					const mat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
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
					sw.style.background = new THREE.Color(
						fileMeta[i]?.color || 0x55ffcc
					).getStyle();
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
						if (fileActors[i] && fileActors[i].parent)
							fileActors[i].parent.remove(fileActors[i]);
						fileActors.splice(i, 1);
						if (filePathLines[i] && filePathLines[i].parent)
							filePathLines[i].parent.remove(filePathLines[i]);
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

			const palette = [
				0x55ffcc, 0x33ddaa, 0x88ffee, 0x22aa88, 0xbbffee, 0x99ffdd, 0xffdd55,
				0xffaa88, 0xaaddff,
			];
			function nextColor(idx) {
				return palette[idx % palette.length];
			}
			function addTracksFromData(name, data) {
				let tracks = [];
				if (Array.isArray(data)) {
					tracks = [data];
				} else if (data && Array.isArray(data.tracks)) {
					tracks = data.tracks.map((t) =>
						Array.isArray(t) ? t : t.points || []
					);
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
				ensureFileActors();
				buildFilePaths();
				rebuildTrackListUI();
				return true;
			}

			// --- Simulation scenarios (procedural flight) ---
			let simActors = []; // { mesh, pathFn }
			let simPathLines = [];
			let simPlaying = false;
			let simStart = 0;
			let simDurationSec = 8.0;
			function clearSimActors() {
				for (const a of simActors) {
					if (a.mesh && a.mesh.parent) a.mesh.parent.remove(a.mesh);
				}
				simActors = [];
				for (const l of simPathLines) {
					if (l && l.parent) l.parent.remove(l);
				}
				simPathLines = [];
			}
			function addActor(color = 0xffcc00, size = 8) {
				const m = new THREE.Mesh(
					new THREE.SphereGeometry(size, 16, 12),
					new THREE.MeshBasicMaterial({ color })
				);
				scene.add(m);
				return m;
			}

			// Rebuild both setups' arrows from current params
			function rebuildArrows() {
				setup3Arrows.forEach((a) => a.parent && a.parent.remove(a));
				setup3Arrows.length = 0;
				computeSetup3Positions().forEach((entry, i) => {
					const a = new THREE.ArrowHelper(
						entry.dir,
						entry.pos.clone(),
						70,
						0x00aaff,
						14,
						7
					);
					a.userData = { setup: 'setup3', index: i };
					scene.add(a);
					setup3Arrows.push(a);
				});
				setup2Arrows.forEach((a) => a.parent && a.parent.remove(a));
				setup2Arrows.length = 0;
				computeSetup2Positions().forEach((entry, i) => {
					const a = new THREE.ArrowHelper(
						entry.dir,
						entry.pos.clone(),
						60,
						0xff4444,
						12,
						6
					);
					a.userData = { setup: 'setup2', index: i };
					scene.add(a);
					setup2Arrows.push(a);
				});
				selectedPOV = null;
				updateArrowVisibility();
			}
			function buildScenario(name) {
				clearSimActors();
				simDurationSec = 8.0;
				if (name === 'straight') {
					const m = addActor(0xff5555, 8);
					const p0 = new THREE.Vector3(-200, 50, 50);
					const p1 = new THREE.Vector3(200, 80, 650);
					simActors.push({
						mesh: m,
						pathFn: (t) => new THREE.Vector3().lerpVectors(p0, p1, t),
					});
				} else if (name === 'curved') {
					const m = addActor(0xff5555, 8);
					simActors.push({
						mesh: m,
						pathFn: (t) => {
							const x = -250 + 500 * t;
							const y = 80 + 100 * Math.sin(Math.PI * 1.5 * t);
							const z = 100 + 520 * t + 80 * Math.sin(Math.PI * 2 * t);
							return new THREE.Vector3(x, y, z);
						},
					});
				} else if (name === 'curved3') {
					const m1 = addActor(0xff5555, 8);
					const m2 = addActor(0xff5555, 8);
					const m3 = addActor(0xff5555, 8);
					// Two similar paths side-by-side (Y offset)
					simActors.push({
						mesh: m1,
						pathFn: (t) => {
							const x = -260 + 520 * t;
							const y = 60 + 90 * Math.sin(Math.PI * 1.6 * t) + 40;
							const z = 90 + 540 * t + 70 * Math.sin(Math.PI * 2.2 * t);
							return new THREE.Vector3(x, y, z);
						},
					});
					simActors.push({
						mesh: m2,
						pathFn: (t) => {
							const x = -260 + 520 * t;
							const y = 60 + 90 * Math.sin(Math.PI * 1.6 * t) - 40;
							const z = 90 + 540 * t + 70 * Math.sin(Math.PI * 2.2 * t);
							return new THREE.Vector3(x, y, z);
						},
					});
					// Third different path
					simActors.push({
						mesh: m3,
						pathFn: (t) => {
							const x = -300 + 600 * t;
							const y = 50 - 120 * Math.cos(Math.PI * t);
							const z = 80 + 520 * t + 120 * Math.sin(Math.PI * 1.2 * t);
							return new THREE.Vector3(x, y, z);
						},
					});
					simDurationSec = 9.0;
				} else {
					// default to straight
					return buildScenario('straight');
				}
			}

			function buildSimPaths() {
				for (const l of simPathLines) {
					if (l && l.parent) l.parent.remove(l);
				}
				simPathLines = [];
				const show = document.getElementById('showPaths')?.checked;
				if (!show) return;
				// Bake ~200 points along each sim path
				const SAMPLES = 200;
				for (const a of simActors) {
					const pts = [];
					for (let i = 0; i <= SAMPLES; i++) {
						const t = i / SAMPLES;
						const p = a.pathFn(t);
						if (p) pts.push(p.clone());
					}
					if (pts.length >= 2) {
						const geom = new THREE.BufferGeometry().setFromPoints(pts);
						const mat = new THREE.LineBasicMaterial({
							color: 0xff5555,
							linewidth: 2,
						});
						const line = new THREE.Line(geom, mat);
						scene.add(line);
						simPathLines.push(line);
					}
				}
			}
			function flash(color = '#ffffff') {
				const f = document.getElementById('flash');
				f.style.background = color;
				f.classList.add('on');
				setTimeout(() => f.classList.remove('on'), 180);
			}
			function startFlight() {
				const sel = document.getElementById('scenario');
				const includeSim = document.getElementById('includeSim');
				const includeSimRow = document.getElementById('includeSimRow');
				const name = sel ? sel.value : 'straight';
				const hasTracks = fileTracks.length > 0;
				const shouldPlaySim = hasTracks
					? includeSim && includeSim.checked
					: true;
				if (shouldPlaySim) {
					buildScenario(name);
					simStart = performance.now();
					simPlaying = true;
					const stopBtn = document.getElementById('stop');
					if (stopBtn) stopBtn.textContent = 'Pause';
					flash('#ffffff');
					buildSimPaths();
				} else {
					clearSimActors();
					simPlaying = false;
				}
				if (hasTracks) {
					fileIdx = 0;
					fileAcc = 0;
					ensureFileActors(fileTracks.length);
					filePlaying = true;
					buildFilePaths();
				}
				// Hide sim include when there is no track
				includeSimRow.style.display = hasTracks ? 'block' : 'none';
			}

			// --- POV integration from Simulation ---
			// Removed Setup 1 and 2
			const center = new THREE.Vector3(0, 0, 300);
			const setup3Arrows = [];
			const setup2Arrows = [];
			// Setup 4 (customizable): parameters and positions
			const defaultSetup2 = {
				heightZ: 120,
				distanceBetween: 600,
				flareAngleDeg: 30,
				distanceToCenter: 800,
				rotationDeg: 0,
			};
			let setup2 = {
				heightZ: 120,
				distanceBetween: 600,
				flareAngleDeg: 30, // angle of left/right outward from center direction
				distanceToCenter: 800,
				rotationDeg: 0,
			};
			mergeInto(setup2, restoreSetup(LS_KEY_SETUP4));
			function computeSetup2Positions() {
				// Linear side-by-side around the mid arrow (not arc about grid center)
				const centerTarget = new THREE.Vector3(0, 0, 300);
				const dz = setup2.heightZ;
				const D = Math.max(1e-6, setup2.distanceToCenter);
				const rot = ((Number(setup2.rotationDeg) || 0) * Math.PI) / 180;
				const a0 = -Math.PI / 2 + rot; // keep existing baseline at -Y when rotation=0
				const baseC = new THREE.Vector3(D * Math.cos(a0), D * Math.sin(a0), dz);
				const dirC = centerTarget.clone().sub(baseC).normalize();
				let lat = new THREE.Vector3(-dirC.y, dirC.x, 0);
				if (lat.length() < 1e-9) lat = new THREE.Vector3(1, 0, 0);
				lat.normalize();
				const halfB = Math.max(0, Number(setup2.distanceBetween) || 0) * 0.5;
				const posL = baseC.clone().add(lat.clone().multiplyScalar(halfB));
				const posR = baseC.clone().add(lat.clone().multiplyScalar(-halfB));
				const flare = ((Number(setup2.flareAngleDeg) || 0) * Math.PI) / 180;
				function rotateZ(v, ang) {
					const x = v.x,
						y = v.y;
					const ca = Math.cos(ang),
						sa = Math.sin(ang);
					return new THREE.Vector3(
						x * ca - y * sa,
						x * sa + y * ca,
						v.z
					).normalize();
				}
				// Base orientations derived from mid direction only (decoupled from distance sliders)
				const dirL = rotateZ(dirC, +flare);
				const dirR = rotateZ(dirC, -flare);
				return [
					{ pos: posL, dir: dirL },
					{ pos: baseC, dir: dirC },
					{ pos: posR, dir: dirR },
				];
			}
			let selectedPOV = null;
			// Setup 1 (customizable around circle)
			const defaultSetup3 = {
				heightZ: 120,
				angleBetweenDeg: 120,
				distanceToCenter: 1200,
				rotationDeg: 0,
			};
			let setup3 = {
				heightZ: 120,
				angleBetweenDeg: 120,
				distanceToCenter: 1200,
				rotationDeg: 0,
			};
			mergeInto(setup3, restoreSetup(LS_KEY_SETUP3));
			function computeSetup3Positions() {
				const R = Math.max(1e-6, setup3.distanceToCenter);
				const delta =
					(clampValue(setup1ParamMeta.angleBetweenDeg, setup3.angleBetweenDeg) *
						Math.PI) /
					180;
				const rot = ((Number(setup3.rotationDeg) || 0) * Math.PI) / 180;
				const z = setup3.heightZ;
				const centerTarget = new THREE.Vector3(0, 0, 300);
				// Orientation: rotate the tripod around Z by rot (center at angle=rot)
				const c = new THREE.Vector3(
					R * Math.cos(rot + 0),
					R * Math.sin(rot + 0),
					z
				);
				const l = new THREE.Vector3(
					R * Math.cos(rot + delta),
					R * Math.sin(rot + delta),
					z
				);
				const r = new THREE.Vector3(
					R * Math.cos(rot - delta),
					R * Math.sin(rot - delta),
					z
				);
				return [l, c, r].map((p) => ({
					pos: p,
					dir: centerTarget.clone().sub(p).normalize(),
				}));
			}
			function makeArrows() {
				// Setup 1 arrows
				computeSetup3Positions().forEach((entry, i) => {
					const a = new THREE.ArrowHelper(
						entry.dir,
						entry.pos.clone(),
						70,
						0x00aaff,
						14,
						7
					);
					a.userData = { setup: 'setup3', index: i };
					scene.add(a);
					setup3Arrows.push(a);
				});
				// Setup 2 custom arrows
				computeSetup2Positions().forEach((entry, i) => {
					const a = new THREE.ArrowHelper(
						entry.dir,
						entry.pos.clone(),
						60,
						0xff4444,
						12,
						6
					);
					a.userData = { setup: 'setup2', index: i };
					scene.add(a);
					setup2Arrows.push(a);
				});
			}
			makeArrows();
			function setSetup3POV(i) {
				const pos =
					computeSetup3Positions()[i]?.pos ||
					new THREE.Vector3(1200, 0, setup3.centerHeightZ);
				camera.position.copy(pos);
				camera.up.set(0, 0, 1);
				camera.lookAt(center);
				controls.target.copy(center);
				controls.update();
				selectedPOV = { setup: 'setup3', index: i };
			}
			function updateArrowVisibility() {
				if (!selectedPOV) {
					setup3Arrows.forEach((a) => (a.visible = true));
					setup2Arrows.forEach((a) => (a.visible = true));
					return;
				}
				setup3Arrows.forEach((a) => (a.visible = false));
				setup2Arrows.forEach((a) => (a.visible = false));
				const threshold = 50;
				let base = null;
				if (selectedPOV.setup === 'setup3') {
					base = setup3Arrows[selectedPOV.index]?.position || null;
				}
				if (selectedPOV.setup === 'setup2') {
					base = setup2Arrows[selectedPOV.index]?.position || null;
				}
				if (base && camera.position.distanceTo(base) > threshold) {
					setup3Arrows.forEach((a) => (a.visible = true));
					setup2Arrows.forEach((a) => (a.visible = true));
				}
			}

			// --- Track playback ---
			let lastTs = performance.now();
			// With unified HUD, no layout adjustments are necessary
			function layoutPanels() {}
			function animate() {
				requestAnimationFrame(animate);
				const nowTs = performance.now();
				const dt = (nowTs - lastTs) / 1000.0;
				lastTs = nowTs;
				// File-based multi-track playback (paused until Start)
				if (fileTracks.length > 0) {
					for (let i = 0; i < fileTracks.length; i++) {
						const tr = fileTracks[i];
						if (!tr || tr.length === 0) continue;
						const p = tr[Math.min(fileIdx, tr.length - 1)];
						if (p && fileActors[i]) fileActors[i].position.set(p.x, p.y, p.z);
					}
					// advance at target playback fps only when playing
					if (filePlaying) {
						fileAcc += dt;
						const step = 1.0 / Math.max(1, filePlaybackFps);
						while (fileAcc >= step) {
							if (fileIdx < fileMaxLen - 1) fileIdx += 1;
							fileAcc -= step;
						}
					}
				}
				// Simulation playback
				if (simActors.length > 0) {
					let t;
					const slider = document.getElementById('simProgress');
					if (slider && slider.dataset.scrubbing === '1') {
						// when scrubbing, trust slider value (0..100)
						t = Math.min(1.0, Math.max(0.0, Number(slider.value) / 100));
					} else if (simPlaying) {
						const elapsed = (performance.now() - simStart) / 1000.0;
						t = Math.min(1.0, Math.max(0.0, elapsed / simDurationSec));
						if (slider) slider.value = String(t * 100);
					} else {
						// paused: keep slider as source of truth if available
						t = slider
							? Math.min(1.0, Math.max(0.0, Number(slider.value) / 100))
							: 0;
					}
					for (const a of simActors) {
						const p = a.pathFn(t);
						if (p) a.mesh.position.copy(p);
					}
					if (simPlaying && t >= 1.0) {
						simPlaying = false;
						const stopBtn = document.getElementById('stop');
						if (stopBtn) stopBtn.textContent = 'Resume';
						flash('#ff3355'); // end marker
					}
				}
				// (no legacy single-dot tracking)
				updateArrowVisibility();
				controls.update();
				renderer.render(scene, camera);
			}
			animate();

			document.getElementById('file').addEventListener('change', (ev) => {
				const files = ev.target.files;
				if (!files || files.length === 0) return;
				// Stop simulation playback and reset state
				clearSimActors();
				simPlaying = false;
				for (let i = 0; i < files.length; i++) {
					const f = files[i];
					const reader = new FileReader();
					reader.onload = () => {
						try {
							const data = JSON.parse(reader.result);
							if (!addTracksFromData(f.name.replace(/\.json$/i, ''), data)) {
								alert('Unsupported track format in ' + f.name);
							}
							const includeSimRow = document.getElementById('includeSimRow');
							if (includeSimRow) includeSimRow.style.display = 'block';
						} catch (e) {
							alert('Bad JSON in ' + f.name);
						}
					};
					reader.readAsText(f);
				}
			});

			// POV buttons removed for Setup 1 & 2 (deleted)
			// Removed static legacy handlers; dynamic controls below
			// Add customizable Setup 1 + Setup 2 controls into HUD
			(function addSetupControls() {
				const hudBody = document.getElementById('hudBody');
				let camConfigHandle = null; // File System Access handle
				let camConfigNameEl = null;
				async function writePayloadToHandle(handle) {
					const s1 = computeSetup3Positions();
					const s2 = computeSetup2Positions();
					const cams = [];
					s1.forEach((e, i) =>
						cams.push({
							name: `setup1_${i + 1}`,
							position: [e.pos.x, e.pos.y, e.pos.z],
							forward: [e.dir.x, e.dir.y, e.dir.z],
							up: [0, 0, 1],
							fov_deg: 70,
						})
					);
					s2.forEach((e, i) =>
						cams.push({
							name: `setup2_${i + 1}`,
							position: [e.pos.x, e.pos.y, e.pos.z],
							forward: [e.dir.x, e.dir.y, e.dir.z],
							up: [0, 0, 1],
							fov_deg: 70,
						})
					);
					const payload = { cameras: cams };
					if (
						(await handle.queryPermission({ mode: 'readwrite' })) !== 'granted'
					) {
						await handle.requestPermission({ mode: 'readwrite' });
					}
					const writable = await handle.createWritable();
					await writable.write(
						new Blob([JSON.stringify(payload, null, 2)], {
							type: 'application/json',
						})
					);
					await writable.close();
				}
				async function saveNow() {
					if (!window.showOpenFilePicker) return; // unsupported
					try {
						if (!camConfigHandle) {
							const [handle] = await window.showOpenFilePicker({
								multiple: false,
								types: [
									{
										description: 'JSON',
										accept: { 'application/json': ['.json'] },
									},
								],
								excludeAcceptAllOption: false,
								id: 'cam_config_picker',
							});
							camConfigHandle = handle;
							if (camConfigNameEl)
								camConfigNameEl.textContent = handle.name || 'cam_config.json';
						}
						await writePayloadToHandle(camConfigHandle);
						flash('#66ff66');
					} catch (e) {
						// canceled or failed
					}
				}
				async function saveAsNow() {
					try {
						let handle = null;
						if (window.showSaveFilePicker) {
							handle = await window.showSaveFilePicker({
								suggestedName: camConfigHandle?.name || 'cam_config.json',
								types: [
									{
										description: 'JSON',
										accept: { 'application/json': ['.json'] },
									},
								],
							});
						} else if (window.showOpenFilePicker) {
							const [h] = await window.showOpenFilePicker({
								multiple: false,
								types: [
									{
										description: 'JSON',
										accept: { 'application/json': ['.json'] },
									},
								],
								excludeAcceptAllOption: false,
								id: 'cam_config_saveas',
							});
							handle = h;
						}
						if (!handle) return;
						await writePayloadToHandle(handle);
						camConfigHandle = handle; // future Cmd+S targets this
						if (camConfigNameEl)
							camConfigNameEl.textContent = handle.name || 'cam_config.json';
						flash('#66ff66');
					} catch (e) {
						// canceled or failed
					}
				}
				// expose for keyboard handlers outside this closure
				window.__saveCamConfig = saveNow;
				window.__saveAsCamConfig = saveAsNow;
				// Load Config UI (sets save target only)
				const loadRow = document.createElement('div');
				loadRow.style.display = 'flex';
				loadRow.style.gap = '8px';
				loadRow.style.alignItems = 'center';
				const loadBtn = document.createElement('button');
				loadBtn.textContent = 'Load Camera Config…';
				loadBtn.onclick = async () => {
					if (!window.showOpenFilePicker) return;
					try {
						const [handle] = await window.showOpenFilePicker({
							multiple: false,
							types: [
								{
									description: 'JSON',
									accept: { 'application/json': ['.json'] },
								},
							],
							excludeAcceptAllOption: false,
							id: 'cam_config_load',
						});
						camConfigHandle = handle;
						if (camConfigNameEl)
							camConfigNameEl.textContent = handle.name || 'cam_config.json';
						// read and apply config values to UI
						try {
							const f = await handle.getFile();
							const txt = await f.text();
							const data = JSON.parse(txt);
							if (typeof applyCamConfigFromData === 'function')
								applyCamConfigFromData(data);
						} catch (e) {}
						flash('#55ccff');
					} catch (e) {}
				};
				const fileNameLabel = document.createElement('span');
				fileNameLabel.style.fontSize = '11px';
				fileNameLabel.style.color = '#aaa';
				fileNameLabel.textContent = '';
				camConfigNameEl = fileNameLabel;
				loadRow.appendChild(loadBtn);
				loadRow.appendChild(fileNameLabel);
				// moved to bottom near Save button
				const title3 = document.createElement('div');
				title3.textContent = 'Setup 1';
				title3.style.margin = '8px 0';
				title3.style.fontSize = '12px';
				title3.style.color = '#888';
				hudBody.appendChild(title3);
				const params3 = document.createElement('div');
				const setup3Inputs = {};
				params3.style.display = 'grid';
				params3.style.gridTemplateColumns = 'auto 80px 1fr';
				params3.style.alignItems = 'center';
				params3.style.columnGap = '8px';
				function addParam3(label, key) {
					const l = document.createElement('label');
					l.textContent = label;
					const inp = document.createElement('input');
					inp.type = 'number';
					inp.step = String((setup1ParamMeta[key] || {}).step ?? 1);
					inp.value = String(clampValue(setup1ParamMeta[key], setup3[key]));
					const slider = document.createElement('input');
					slider.type = 'range';
					if (setup1ParamMeta[key]) {
						slider.min = String(setup1ParamMeta[key].min);
						slider.max = String(setup1ParamMeta[key].max);
						slider.step = String(setup1ParamMeta[key].step);
					}
					slider.value = String(inp.value);
					setup3Inputs[key] = inp;
					function onChange(val) {
						const v = clampValue(setup1ParamMeta[key], val);
						inp.value = String(v);
						slider.value = String(v);
						setup3[key] = Number(v);
						// rebuild arrows
						setup3Arrows.forEach((a) => a.parent && a.parent.remove(a));
						setup3Arrows.length = 0;
						computeSetup3Positions().forEach((entry, i) => {
							const a = new THREE.ArrowHelper(
								entry.dir,
								entry.pos.clone(),
								70,
								0x00aaff,
								14,
								7
							);
							a.userData = { setup: 'setup3', index: i };
							scene.add(a);
							setup3Arrows.push(a);
						});
						rebuildArrows();
						persistSetup(LS_KEY_SETUP3, setup3);
					}
					inp.onchange = () => onChange(inp.value);
					slider.oninput = () => onChange(slider.value);
					params3.appendChild(l);
					params3.appendChild(inp);
					params3.appendChild(slider);
				}
				addParam3('Height Z (units)', 'heightZ');
				addParam3('Angle between (deg)', 'angleBetweenDeg');
				addParam3('Rotation (deg)', 'rotationDeg');
				addParam3('Distance to center (units)', 'distanceToCenter');
				hudBody.appendChild(params3);
				const row3 = document.createElement('div');
				row3.style.marginTop = '6px';
				function addBtn3(name, idx) {
					const b = document.createElement('button');
					b.className = 'setup3-btn';
					b.textContent = name;
					b.onclick = () => {
						const entry = computeSetup3Positions()[idx];
						if (entry) {
							const p = entry.pos;
							const t = entry.pos
								.clone()
								.add(entry.dir.clone().multiplyScalar(500));
							camera.position.copy(p);
							camera.up.set(0, 0, 1);
							camera.lookAt(t);
							controls.target.copy(t);
							controls.update();
							selectedPOV = { setup: 'setup3', index: idx };
							updateArrowVisibility();
						}
					};
					row3.appendChild(b);
				}
				addBtn3('POV 1', 0);
				addBtn3('POV 2', 1);
				addBtn3('POV 3', 2);
				hudBody.appendChild(row3);
				// Reset to defaults for Setup 1
				const reset3 = document.createElement('button');
				reset3.className = 'setup3-btn';
				reset3.textContent = 'Reset to defaults';
				reset3.style.marginTop = '6px';
				reset3.onclick = () => {
					for (const k of Object.keys(defaultSetup3)) {
						setup3[k] = defaultSetup3[k];
						if (setup3Inputs[k]) setup3Inputs[k].value = defaultSetup3[k];
					}
					setup3Arrows.forEach((a) => a.parent && a.parent.remove(a));
					setup3Arrows.length = 0;
					computeSetup3Positions().forEach((entry, i) => {
						const a = new THREE.ArrowHelper(
							entry.dir,
							entry.pos.clone(),
							70,
							0x00aaff,
							14,
							7
						);
						a.userData = { setup: 'setup3', index: i };
						scene.add(a);
						setup3Arrows.push(a);
					});
					updateArrowVisibility();
					persistSetup(LS_KEY_SETUP3, setup3);
				};
				hudBody.appendChild(reset3);

				// (per-setup save removed; unified save provided at bottom)

				const title4 = document.createElement('div');
				title4.textContent = 'Setup 2';
				title4.style.margin = '8px 0';
				title4.style.fontSize = '12px';
				title4.style.color = '#888';
				hudBody.appendChild(title4);
				const params = document.createElement('div');
				const setup2Inputs = {};
				params.style.display = 'grid';
				params.style.gridTemplateColumns = 'auto 80px 1fr';
				params.style.alignItems = 'center';
				params.style.columnGap = '8px';
				function addParam(label, key) {
					const l = document.createElement('label');
					l.textContent = label;
					const inp = document.createElement('input');
					inp.type = 'number';
					inp.step = String((setup2ParamMeta[key] || {}).step ?? 1);
					inp.value = String(clampValue(setup2ParamMeta[key], setup2[key]));
					const slider = document.createElement('input');
					slider.type = 'range';
					if (setup2ParamMeta[key]) {
						slider.min = String(setup2ParamMeta[key].min);
						slider.max = String(setup2ParamMeta[key].max);
						slider.step = String(setup2ParamMeta[key].step);
					}
					slider.value = String(inp.value);
					setup2Inputs[key] = inp;
					function onChange(val) {
						const v = clampValue(setup2ParamMeta[key], val);
						inp.value = String(v);
						slider.value = String(v);
						setup2[key] = Number(v);
						// Rebuild arrows
						setup2Arrows.forEach((a) => a.parent && a.parent.remove(a));
						setup2Arrows.length = 0;
						computeSetup2Positions().forEach((entry, i) => {
							const a = new THREE.ArrowHelper(
								entry.dir,
								entry.pos.clone(),
								60,
								0xff4444,
								12,
								6
							);
							a.userData = { setup: 'setup2', index: i };
							scene.add(a);
							setup2Arrows.push(a);
						});
						rebuildArrows();
						persistSetup(LS_KEY_SETUP4, setup2);
					}
					inp.onchange = () => onChange(inp.value);
					slider.oninput = () => onChange(slider.value);
					params.appendChild(l);
					params.appendChild(inp);
					params.appendChild(slider);
				}
				addParam('Height Z (units)', 'heightZ');
				addParam('Distance between (units)', 'distanceBetween');
				addParam('Flare angle (deg)', 'flareAngleDeg');
				addParam('Rotation (deg)', 'rotationDeg');
				addParam('Distance to center (units)', 'distanceToCenter');
				hudBody.appendChild(params);
				const row = document.createElement('div');
				row.style.marginTop = '6px';
				function addBtn(name, idx) {
					const b = document.createElement('button');
					b.className = 'setup2-btn';
					b.textContent = name;
					b.onclick = () => {
						const entry = computeSetup2Positions()[idx];
						if (!entry) return;
						const p = entry.pos;
						const t = entry.pos
							.clone()
							.add(entry.dir.clone().multiplyScalar(500));
						camera.position.copy(p);
						camera.up.set(0, 0, 1);
						camera.lookAt(t);
						controls.target.copy(t);
						controls.update();
						selectedPOV = { setup: 'setup2', index: idx };
						updateArrowVisibility();
					};
					row.appendChild(b);
				}
				addBtn('POV 1', 0);
				addBtn('POV 2', 1);
				addBtn('POV 3', 2);
				hudBody.appendChild(row);
				// Reset to defaults for Setup 2
				const reset4 = document.createElement('button');
				reset4.className = 'setup2-btn';
				reset4.textContent = 'Reset to defaults';
				reset4.style.marginTop = '6px';
				reset4.onclick = () => {
					for (const k of Object.keys(defaultSetup2)) {
						setup2[k] = defaultSetup2[k];
						if (setup2Inputs[k]) setup2Inputs[k].value = defaultSetup2[k];
					}
					setup2Arrows.forEach((a) => a.parent && a.parent.remove(a));
					setup2Arrows.length = 0;
					computeSetup2Positions().forEach((entry, i) => {
						const a = new THREE.ArrowHelper(
							entry.dir,
							entry.pos.clone(),
							60,
							0xff4444,
							12,
							6
						);
						a.userData = { setup: 'setup2', index: i };
						scene.add(a);
						setup2Arrows.push(a);
					});
					updateArrowVisibility();
					persistSetup(LS_KEY_SETUP4, setup2);
				};
				hudBody.appendChild(reset4);
				// (per-setup save removed; unified save provided at bottom)

				// Apply a loaded cam_config.json into Setup 1/2 UI
				function applyCamConfigFromData(data) {
					if (!data) return;
					const cams = Array.isArray(data.cameras)
						? data.cameras
						: Array.isArray(data)
						? data
						: [];
					if (cams.length === 0) return;
					const byName = new Map();
					for (const c of cams) if (c && c.name) byName.set(String(c.name), c);
					const s1 = [
						byName.get('setup1_1'),
						byName.get('setup1_2'),
						byName.get('setup1_3'),
					].filter(Boolean);
					const s2 = [
						byName.get('setup2_1'),
						byName.get('setup2_2'),
						byName.get('setup2_3'),
					].filter(Boolean);
					function arrToVec(a) {
						return new THREE.Vector3(
							Number(a?.[0] || 0),
							Number(a?.[1] || 0),
							Number(a?.[2] || 0)
						);
					}
					function clamp(meta, v) {
						return clampValue(meta, v);
					}
					// Setup 1
					if (s1.length === 3) {
						const pL = arrToVec(s1[0].position);
						const pC = arrToVec(s1[1].position);
						const pR = arrToVec(s1[2].position);
						const zAvg = (pL.z + pC.z + pR.z) / 3;
						const D = Math.sqrt(pC.x * pC.x + pC.y * pC.y);
						const aL = Math.atan2(pL.y, pL.x);
						const aC = Math.atan2(pC.y, pC.x);
						let deltaDeg = Math.abs(((aL - aC) * 180) / Math.PI);
						if (deltaDeg > 180) deltaDeg = 360 - deltaDeg;
						let rot1 = (aC * 180) / Math.PI;
						if (rot1 < 0) rot1 += 360;
						const s1Vals = {
							heightZ: clamp(setup1ParamMeta.heightZ, zAvg),
							angleBetweenDeg: clamp(setup1ParamMeta.angleBetweenDeg, deltaDeg),
							distanceToCenter: clamp(setup1ParamMeta.distanceToCenter, D),
							rotationDeg: clamp(setup1ParamMeta.rotationDeg, rot1),
						};
						for (const k of Object.keys(s1Vals)) {
							setup3[k] = Number(s1Vals[k]);
							if (setup3Inputs && setup3Inputs[k]) {
								setup3Inputs[k].value = String(s1Vals[k]);
								if (typeof setup3Inputs[k].onchange === 'function')
									setup3Inputs[k].onchange();
							}
						}
					}
					// Setup 2
					if (s2.length === 3) {
						const pL = arrToVec(s2[0].position);
						const pC = arrToVec(s2[1].position);
						const pR = arrToVec(s2[2].position);
						const fL = arrToVec(s2[0].forward);
						const fC = arrToVec(s2[1].forward);
						const zAvg = (pL.z + pC.z + pR.z) / 3;
						const D = Math.sqrt(pC.x * pC.x + pC.y * pC.y);
						const distBetween = pL.distanceTo(pR);
						// signed azimuth angle around Z from center dir to left dir (XY plane only)
						const dotXY = fC.x * fL.x + fC.y * fL.y;
						const crossZ = fC.x * fL.y - fC.y * fL.x;
						let ang = (Math.atan2(crossZ, dotXY) * 180) / Math.PI;
						// clamp flare to [-90, 90]
						if (ang > 180) ang -= 360;
						if (ang < -180) ang += 360;
						ang = Math.max(-90, Math.min(90, ang));
						// rotation around center (0 deg keeps center at -Y); infer from center position angle
						let aC = Math.atan2(pC.y, pC.x); // actual center azimuth
						let rot2 = (aC * 180) / Math.PI + 90; // compensate baseline of -90 deg
						rot2 = ((rot2 % 360) + 360) % 360;
						const s2Vals = {
							heightZ: clamp(setup2ParamMeta.heightZ, zAvg),
							distanceBetween: clamp(
								setup2ParamMeta.distanceBetween,
								distBetween
							),
							flareAngleDeg: clamp(setup2ParamMeta.flareAngleDeg, ang),
							distanceToCenter: clamp(setup2ParamMeta.distanceToCenter, D),
							rotationDeg: clamp(setup2ParamMeta.rotationDeg, rot2),
						};
						for (const k of Object.keys(s2Vals)) {
							setup2[k] = Number(s2Vals[k]);
							if (setup2Inputs && setup2Inputs[k]) {
								setup2Inputs[k].value = String(s2Vals[k]);
								if (typeof setup2Inputs[k].onchange === 'function')
									setup2Inputs[k].onchange();
							}
						}
					}
				}

				// Unified save button (bottom of HUD)
				const saveAllBtn = document.createElement('button');
				saveAllBtn.id = 'saveAllSetupsBtn';
				saveAllBtn.className = 'save-btn';
				saveAllBtn.textContent = 'Save Camera Configs';
				saveAllBtn.style.marginTop = '10px';
				saveAllBtn.onclick = async () => {
					await saveNow();
				};
				hudBody.appendChild(saveAllBtn);
				// place Load + filename just below Save
				loadRow.style.marginTop = '8px';
				hudBody.appendChild(loadRow);
			})();
			document.getElementById('start').onclick = () => startFlight();
			const simSlider = document.getElementById('simProgress');
			const stopBtn = document.getElementById('stop');
			if (simSlider) {
				simSlider.addEventListener('input', () => {
					// mark scrubbing and pause playback while dragging
					simSlider.dataset.scrubbing = '1';
					simPlaying = false;
				});
				simSlider.addEventListener('change', () => {
					// finished scrubbing
					simSlider.dataset.scrubbing = '0';
				});
			}
			if (stopBtn) {
				const updateStopLabel = () => {
					stopBtn.textContent = simPlaying ? 'Pause' : 'Resume';
				};
				stopBtn.onclick = () => {
					// toggle pause/resume
					simPlaying = !simPlaying;
					// when resuming, sync simStart so current slider position stays
					const slider = document.getElementById('simProgress');
					if (simPlaying) {
						const t = slider
							? Math.min(1.0, Math.max(0.0, Number(slider.value) / 100))
							: 0;
						simStart = performance.now() - t * simDurationSec * 1000.0;
						slider && (slider.dataset.scrubbing = '0');
					}
					updateStopLabel();
				};
				updateStopLabel();
			}
			document
				.getElementById('scenario')
				.addEventListener('change', layoutPanels);
			const simChk = document.getElementById('includeSim');
			if (simChk) simChk.addEventListener('change', layoutPanels);
			const pathsChk = document.getElementById('showPaths');
			if (pathsChk)
				pathsChk.addEventListener('change', () => {
					buildFilePaths();
					buildSimPaths();
				});

			// Default camera
			camera.position.set(0, -900, 200);
			camera.lookAt(0, 0, 300);

			addEventListener('resize', () => {
				camera.aspect = innerWidth / innerHeight;
				camera.updateProjectionMatrix();
				renderer.setSize(innerWidth, innerHeight);
				layoutPanels();
			});
			// Map Cmd+S to Save, Cmd+Shift+S to Save As
			addEventListener('keydown', (ev) => {
				if (ev.metaKey && (ev.key === 's' || ev.key === 'S')) {
					ev.preventDefault();
					if (ev.shiftKey) {
						if (window.__saveAsCamConfig) window.__saveAsCamConfig();
					} else {
						if (window.__saveCamConfig) window.__saveCamConfig();
					}
				}
				// Space = Start flight (ignore when typing)
				if (ev.code === 'Space' && !ev.metaKey && !ev.ctrlKey && !ev.altKey) {
					const tag = (ev.target && ev.target.tagName) || '';
					if (
						tag !== 'INPUT' &&
						tag !== 'TEXTAREA' &&
						tag !== 'SELECT' &&
						tag !== 'BUTTON'
					) {
						ev.preventDefault();
						startFlight();
					}
				}
			});
			layoutPanels();
		
