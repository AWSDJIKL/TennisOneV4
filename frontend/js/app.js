/**
 * Sport Vision — Main Application
 * WebSocket communication, UI orchestration, and event handling
 */

(function () {
    'use strict';

    // ============ DOM Elements ============
    const $ = (id) => document.getElementById(id);

    const heroSection = $('hero-section');
    const analysisSection = $('analysis-section');
    const videoCanvas = $('video-canvas');
    const videoOverlay = $('video-overlay');
    const progressFill = $('progress-fill');
    const statusDot = $('status-dot');
    const statusText = $('status-text');
    const actionIcon = $('action-icon');
    const actionName = $('action-name');
    const confidenceFill = $('confidence-fill');
    const confidenceText = $('confidence-text');
    const actionCard = $('action-card');
    const timelineContainer = $('timeline-container');
    const uploaInput = $('video-upload');
    const uploadBtn = $('upload-btn');
    const demoBtn = $('demo-btn');
    const cameraBtn = $('camera-btn');
    const demoSelector = $('demo-selector');
    const demoGrid = $('demo-grid');
    const btnStop = $('btn-stop');
    const btnBack = $('btn-back');

    const videoCtx = videoCanvas.getContext('2d');

    // ============ State ============
    let ws = null;
    let isAnalyzing = false;
    let dashboard = null;
    let currentSport = 'badminton';
    let frameImage = new Image();
    let lastActionColor = '#00f0ff';
    let videoModal = null;

    // ============ Initialize ============
    function init() {
        dashboard = new Dashboard();
        bindEvents();
        setStatus('ready', '就绪');
        loadDemos();
    }

    // ============ Event Binding ============
    function bindEvents() {
        // Sport toggle
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentSport = btn.dataset.sport;
            });
        });

        // Upload
        if (uploaInput) {
            uploaInput.addEventListener('change', handleUpload);
        }

        // Demo button
        if (demoBtn && demoSelector) {
            demoBtn.addEventListener('click', () => {
                demoSelector.style.display = demoSelector.style.display === 'none' ? 'block' : 'none';
            });
        }

        // Camera button
        if (cameraBtn) {
            cameraBtn.addEventListener('click', () => {
                startAnalysis('camera');
            });
        }

        // Stop / Back
        if (btnStop) btnStop.addEventListener('click', stopAnalysis);
        if (btnBack) btnBack.addEventListener('click', goBack);

        // ESC 关闭视频弹窗
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeVideoModal();
            }
        });
    }

    // ============ Status ============
    function setStatus(type, text) {
        if (!statusDot || !statusText) return;

        statusDot.className = 'status-dot';
        if (type === 'processing') {
            statusDot.classList.add('processing');
        } else if (type === 'active') {
            statusDot.classList.add('active');
        }
        statusText.textContent = text;
    }

    // ============ Demo Videos ============
    async function loadDemos() {
        try {
            const resp = await fetch('/api/demos');
            const data = await resp.json();

            if (!demoGrid) return;

            if (data.demos.length === 0) {
                demoGrid.innerHTML = `
                    <div class="demo-card" style="grid-column: 1/-1; text-align: center; color: var(--text-dim);">
                        <p>暂无 Demo 视频</p>
                        <p style="font-size: 0.7rem; margin-top: 8px;">
                            将 .mp4 文件放入 <code>demo_videos/</code> 目录
                        </p>
                    </div>
                `;
                return;
            }

            demoGrid.innerHTML = data.demos.map(d => `
                <div class="demo-card" data-demo-id="${d.id}" data-filename="${d.filename}">
                    <div class="demo-card-name">🎬 ${d.name}</div>
                    <div class="demo-card-size">${d.size_mb} MB</div>
                </div>
            `).join('');

            demoGrid.querySelectorAll('.demo-card').forEach(card => {
                card.addEventListener('click', () => {
                    const demoId = card.dataset.demoId;
                    if (demoId) startAnalysis('demo', demoId);
                });
            });
        } catch (err) {
            console.warn('Failed to load demos:', err);
        }
    }

    // ============ Upload ============
    async function handleUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        setStatus('processing', '上传中...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });
            const data = await resp.json();

            if (data.error) {
                alert(data.error);
                setStatus('ready', '就绪');
                return;
            }

            startAnalysis('upload', null, data.path);
        } catch (err) {
            alert('上传失败: ' + err.message);
            setStatus('ready', '就绪');
        } finally {
            e.target.value = '';
        }
    }

    // ============ WebSocket Analysis ============
    function startAnalysis(source, demoId, uploadPath) {
        if (!heroSection || !analysisSection || !videoOverlay) return;

        heroSection.style.display = 'none';
        analysisSection.style.display = 'block';
        videoOverlay.classList.remove('hidden');

        if (dashboard) dashboard.reset();
        if (timelineContainer) timelineContainer.innerHTML = '';
        if (progressFill) progressFill.style.width = '0%';
        isAnalyzing = true;
        closeVideoModal();

        setStatus('processing', '连接分析引擎...');

        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws/analyze`);

        ws.onopen = () => {
            setStatus('processing', '分析中...');

            const msg = { type: 'start', source };
            if (source === 'demo') msg.id = demoId;
            if (source === 'upload') msg.path = uploadPath;

            ws.send(JSON.stringify(msg));
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            handleWSMessage(msg);
        };

        ws.onerror = (err) => {
            console.error('WebSocket error:', err);
            setStatus('ready', '连接错误');
        };

        ws.onclose = () => {
            if (isAnalyzing) {
                setStatus('active', '分析完成');
                isAnalyzing = false;
            }
        };
    }

    function handleWSMessage(msg) {
        switch (msg.type) {
            case 'started':
                if (videoOverlay) videoOverlay.classList.add('hidden');
                setStatus('processing', '分析中...');
                break;

            case 'frame':
                renderFrame(msg.data);
                break;

            case 'complete':
                setStatus('active', '分析完成');
                isAnalyzing = false;
                break;

            case 'stopped':
                setStatus('ready', '已停止');
                isAnalyzing = false;
                break;

            case 'error':
                setStatus('ready', `错误: ${msg.message}`);
                isAnalyzing = false;
                break;
        }
    }

    // ============ Frame Rendering ============
    function renderFrame(data) {
        if (!data.frame_base64) return;

        if (progressFill) {
            progressFill.style.width = `${data.progress * 100}%`;
        }

        frameImage.onload = () => {
            if (!videoCanvas || !videoCtx) return;
            videoCanvas.width = frameImage.naturalWidth;
            videoCanvas.height = frameImage.naturalHeight;
            videoCtx.drawImage(frameImage, 0, 0);
        };
        frameImage.src = 'data:image/jpeg;base64,' + data.frame_base64;

        if (data.action) {
            updateActionDisplay(data.action);
        }

        if (data.pose && dashboard) {
            dashboard.updateAngles(data.pose.joint_angles);
            dashboard.updateBiomechanics(data.pose.biomechanics);
        }

        if (data.action && dashboard) {
            dashboard.updateStats(data.action.action_counts);
        }

        if (data.frame_number % 10 === 0 && data.heatmap_data && dashboard) {
            dashboard.updateHeatmap(data.heatmap_data, data.width, data.height);
        }
    }

    function updateActionDisplay(actionData) {
        const info = actionData.action_info;
        if (!info) return;

        if (actionIcon) actionIcon.textContent = info.icon;
        if (actionName) actionName.textContent = info.name;

        const pct = Math.round(actionData.confidence * 100);
        if (confidenceFill) confidenceFill.style.width = `${pct}%`;
        if (confidenceText) confidenceText.textContent = `${pct}%`;

        if (info.color !== lastActionColor) {
            if (actionCard) actionCard.style.borderColor = info.color;
            if (actionName) actionName.style.color = info.color;
            lastActionColor = info.color;
        }

        if (actionData.is_new_action) {
            addTimelineItem(actionData);
        }
    }

    function addTimelineItem(actionData) {
        if (!timelineContainer) return;

        const info = actionData.action_info;
        const videoUrl = actionData.clip?.video_url || '';
        const clipId = actionData.clip?.clip_id || '';

        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'timeline-item';
        item.style.borderColor = info.color;
        item.style.color = info.color;
        item.style.background = info.color + '15';
        item.textContent = `${info.icon} ${info.name}`;

        item.dataset.videoUrl = videoUrl;
        item.dataset.clipId = clipId;
        item.dataset.actionName = info.name;

        item.addEventListener('click', () => {
            if (!item.dataset.videoUrl) {
                alert('该动作还没有视频切片');
                return;
            }

            openVideoModal(item.dataset.videoUrl, info, item.dataset.clipId);
        });

        timelineContainer.appendChild(item);
        timelineContainer.scrollLeft = timelineContainer.scrollWidth;

        if (actionCard) {
            actionCard.classList.remove('flash');
            void actionCard.offsetWidth;
            actionCard.classList.add('flash');
        }
    }

    // ============ Report API ============
    async function requestAnalysisReport(videoUrl, clipId, actionName) {
        const resp = await fetch('/api/report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                video_url: videoUrl,
                clip_id: clipId,
                action_name: actionName
            })
        });

        let data = null;
        try {
            data = await resp.json();
        } catch (e) {
            data = null;
        }

        if (!resp.ok) {
            const message = data?.error || `请求失败 (${resp.status})`;
            throw new Error(message);
        }

        return data;
    }

    // ============ Video Modal ============
    function openVideoModal(videoUrl, info, clipId = '') {
        closeVideoModal();

        videoModal = document.createElement('div');
        videoModal.id = 'video-modal';
        videoModal.style.position = 'fixed';
        videoModal.style.inset = '0';
        videoModal.style.background = 'rgba(0, 0, 0, 0.75)';
        videoModal.style.display = 'flex';
        videoModal.style.alignItems = 'center';
        videoModal.style.justifyContent = 'center';
        videoModal.style.zIndex = '9999';
        videoModal.style.padding = '24px';

        videoModal.innerHTML = `
            <div style="
                position: relative;
                width: min(960px, 94vw);
                max-height: 90vh;
                background: #0b1220;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 16px;
                padding: 16px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.45);
                overflow: auto;
            ">
                <button id="close-video-modal" type="button" style="
                    position: absolute;
                    right: 12px;
                    top: 12px;
                    width: 36px;
                    height: 36px;
                    border: none;
                    border-radius: 50%;
                    background: rgba(255,255,255,0.1);
                    color: white;
                    font-size: 20px;
                    cursor: pointer;
                ">×</button>

                <div style="
                    display:flex;
                    align-items:flex-start;
                    justify-content:space-between;
                    gap:16px;
                    margin-bottom:12px;
                    padding-right:48px;
                ">
                    <div style="color: white; font-weight: 700;">
                        ${info.icon} ${info.name}
                        ${clipId ? `<span style="display:block; margin-top:6px; font-size:12px; color:rgba(255,255,255,0.65);">clip: ${clipId}</span>` : ''}
                    </div>

                    <button id="analyze-report-btn" type="button" style="
                        flex-shrink:0;
                        padding:10px 14px;
                        border:none;
                        border-radius:10px;
                        background: linear-gradient(135deg, #00f0ff, #3366ff);
                        color:#05101a;
                        font-weight:700;
                        cursor:pointer;
                    ">分析报告</button>
                </div>

                <video id="video-player" controls playsinline autoplay style="
                    width: 100%;
                    max-height: 54vh;
                    border-radius: 12px;
                    background: black;
                    display: block;
                "></video>

                <div id="video-error" style="
                    display:none;
                    margin-top:12px;
                    color:#ff8a8a;
                    font-size:14px;
                    line-height:1.5;
                ">
                    视频加载失败，请检查切片地址是否可访问。
                </div>

                <div id="report-status" style="
                    display:none;
                    margin-top:12px;
                    color:rgba(255,255,255,0.78);
                    font-size:14px;
                    line-height:1.6;
                "></div>

                <pre id="report-result" style="
                    display:none;
                    margin-top:12px;
                    padding:12px;
                    border-radius:12px;
                    background: rgba(255,255,255,0.05);
                    color:#dbeafe;
                    font-size:13px;
                    line-height:1.6;
                    white-space:pre-wrap;
                    word-break:break-word;
                "></pre>
            </div>
        `;

        document.body.appendChild(videoModal);

        const closeBtn = document.getElementById('close-video-modal');
        if (closeBtn) {
            closeBtn.addEventListener('click', closeVideoModal);
        }

        videoModal.addEventListener('click', (e) => {
            if (e.target === videoModal) {
                closeVideoModal();
            }
        });

        const player = videoModal.querySelector('#video-player');
        const errorBox = videoModal.querySelector('#video-error');
        const reportBtn = videoModal.querySelector('#analyze-report-btn');
        const reportStatus = videoModal.querySelector('#report-status');
        const reportResult = videoModal.querySelector('#report-result');

        if (reportBtn) {
            reportBtn.addEventListener('click', async () => {
                reportBtn.disabled = true;
                reportBtn.textContent = '生成中...';

                if (reportStatus) {
                    reportStatus.style.display = 'block';
                    reportStatus.style.color = 'rgba(255,255,255,0.78)';
                    reportStatus.textContent = '正在生成分析报告...';
                }

                if (reportResult) {
                    reportResult.style.display = 'none';
                    reportResult.textContent = '';
                }

                try {
                    const result = await requestAnalysisReport(videoUrl, clipId, info.name);

                    if (reportStatus) {
                        reportStatus.style.display = 'block';
                        reportStatus.style.color = '#86efac';
                        reportStatus.textContent = '分析报告生成成功';
                    }

                    if (reportResult) {
                        reportResult.style.display = 'block';
                        reportResult.textContent = JSON.stringify(result, null, 2);
                    }
                } catch (err) {
                    console.error('report request error:', err);
                    if (reportStatus) {
                        reportStatus.style.display = 'block';
                        reportStatus.style.color = '#fca5a5';
                        reportStatus.textContent = `分析报告生成失败：${err.message}`;
                    }
                } finally {
                    reportBtn.disabled = false;
                    reportBtn.textContent = '分析报告';
                }
            });
        }

        if (!player) return;

        player.onerror = () => {
            console.error('video load error:', videoUrl);
            if (errorBox) {
                errorBox.style.display = 'block';
                errorBox.textContent = `视频加载失败：${videoUrl}`;
            }
        };

        player.onloadeddata = () => {
            if (errorBox) errorBox.style.display = 'none';
        };

        player.src = videoUrl;
        player.load();

        const playPromise = player.play();
        if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch((err) => {
                console.warn('video autoplay blocked or failed:', err);
            });
        }
    }

    function closeVideoModal() {
        if (videoModal) {
            const player = videoModal.querySelector('#video-player');
            if (player) {
                try {
                    player.pause();
                    player.removeAttribute('src');
                    player.load();
                } catch (e) {
                    console.warn('video cleanup failed:', e);
                }
            }
            videoModal.remove();
            videoModal = null;
        }
    }

    // ============ Controls ============
    function stopAnalysis() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'stop' }));
        }
        isAnalyzing = false;
        setStatus('ready', '已停止');
    }

    function goBack() {
        stopAnalysis();
        closeVideoModal();

        if (ws) {
            ws.close();
            ws = null;
        }

        if (analysisSection) analysisSection.style.display = 'none';
        if (heroSection) heroSection.style.display = 'flex';
        setStatus('ready', '就绪');
    }

    // ============ Start ============
    document.addEventListener('DOMContentLoaded', init);
})();