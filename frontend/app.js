// ==========================================================================
// LecLens — Minimalist Single-Video Grounding RAG Assistant
// ==========================================================================

const state = {
    currentTime: 0.0,
    totalDuration: 425.0,
    isPlaying: false,
    activeSegment: null, // {start, end}
    subtitles: [],
    playerMode: 'virtual'
};

// DOM References
const elements = {
    // Stage & Dropzone Elements
    videoWrapper: document.getElementById('video-wrapper'),
    dropzoneStage: document.getElementById('dropzone-stage'),
    dropzoneCard: document.getElementById('dropzone-card'),
    btnBrowseVideo: document.getElementById('btn-browse-video'),
    dropFileInput: document.getElementById('drop-file-input'),
    dropYoutubeInput: document.getElementById('drop-youtube-input'),
    btnDropYoutubeSubmit: document.getElementById('btn-drop-youtube-submit'),
    btnChangeVideo: document.getElementById('btn-change-video'),
    stageLoadingOverlay: document.getElementById('stage-loading-overlay'),
    stageLoadingTitle: document.getElementById('stage-loading-title'),
    stageLoadingSub: document.getElementById('stage-loading-sub'),
    nativeVideo: document.getElementById('native-video'),

    // Player Controls
    btnPlayPause: document.getElementById('btn-play-pause'),
    timelineTrack: document.getElementById('timeline-track'),
    timelineProgress: document.getElementById('timeline-progress'),
    timelineMarkers: document.getElementById('timeline-markers'),
    timelineScrubber: document.getElementById('timeline-scrubber'),
    timeDisplay: document.getElementById('time-display'),

    // Query Area
    queryInput: document.getElementById('query-input'),
    btnSearch: document.getElementById('btn-search'),
    chips: document.querySelectorAll('.chip'),

    // Results Area
    resultsLoading: document.getElementById('results-loading'),
    resultsContainer: document.getElementById('results-container'),

    // Groq Header & Modal
    headerGroqBtn: document.getElementById('btn-open-groq-key'),
    headerGroqBadge: document.getElementById('header-groq-badge'),
    groqModal: document.getElementById('groq-modal'),
    btnCloseGroq: document.getElementById('btn-close-groq'),
    btnCancelGroq: document.getElementById('btn-cancel-groq'),
    inputModalGroqKey: document.getElementById('input-modal-groq-key'),
    btnSaveModalGroqKey: document.getElementById('btn-save-modal-groq-key'),
    groqModalStatusBadge: document.getElementById('groq-modal-status-badge')
};

function formatTime(sec) {
    if (isNaN(sec) || sec < 0) sec = 0;
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// Stage Loading Overlay Helpers
function showStageLoading(title, subtitle) {
    if (elements.stageLoadingTitle) elements.stageLoadingTitle.textContent = title;
    if (elements.stageLoadingSub) elements.stageLoadingSub.textContent = subtitle;
    if (elements.stageLoadingOverlay) elements.stageLoadingOverlay.style.display = 'flex';
}

function hideStageLoading() {
    if (elements.stageLoadingOverlay) elements.stageLoadingOverlay.style.display = 'none';
}

// ==========================================================================
// Player Engine & Seeking
// ==========================================================================

let masterTimer = null;

function setPlayState(play) {
    state.isPlaying = play;
    if (!elements.btnPlayPause) return;
    if (play) {
        elements.btnPlayPause.innerHTML = '<span class="material-symbols-outlined">pause</span>';
        elements.btnPlayPause.title = 'Tạm dừng';
        elements.btnPlayPause.style.background = '#8b5cf6';
        if (state.playerMode === 'native' && elements.nativeVideo && elements.nativeVideo.src) {
            elements.nativeVideo.play().catch(e => console.log('Video play error:', e));
        }
    } else {
        elements.btnPlayPause.innerHTML = '<span class="material-symbols-outlined">play_arrow</span>';
        elements.btnPlayPause.title = 'Phát';
        elements.btnPlayPause.style.background = 'var(--primary-cyan)';
        if (state.playerMode === 'native' && elements.nativeVideo && elements.nativeVideo.src) {
            elements.nativeVideo.pause();
        }
    }
}

function seekTo(seconds) {
    state.currentTime = Math.max(0, Math.min(seconds, state.totalDuration));
    if (state.playerMode === 'native' && elements.nativeVideo && elements.nativeVideo.src) {
        elements.nativeVideo.currentTime = state.currentTime;
    }
    updateTimeDisplay();
}

function updateTimeDisplay() {
    const cur = formatTime(state.currentTime);
    const tot = formatTime(state.totalDuration);
    if (elements.timeDisplay) {
        elements.timeDisplay.textContent = `${cur} / ${tot}`;
    }

    const pct = (state.currentTime / (state.totalDuration || 1)) * 100;
    if (elements.timelineProgress) elements.timelineProgress.style.width = `${pct}%`;
    if (elements.timelineScrubber) elements.timelineScrubber.style.left = `${pct}%`;

    // Auto-loop segment check
    if (state.activeSegment && state.isPlaying) {
        if (state.currentTime >= state.activeSegment.end) {
            state.currentTime = state.activeSegment.start;
            if (state.playerMode === 'native' && elements.nativeVideo && elements.nativeVideo.src) {
                elements.nativeVideo.currentTime = state.currentTime;
            }
        }
    }
}

function startClock() {
    if (masterTimer) clearInterval(masterTimer);
    masterTimer = setInterval(() => {
        if (state.isPlaying) {
            if (state.playerMode === 'virtual' || !elements.nativeVideo || !elements.nativeVideo.src) {
                state.currentTime += 0.1;
                if (state.currentTime >= state.totalDuration) {
                    state.currentTime = 0;
                    setPlayState(false);
                }
            } else {
                state.currentTime = elements.nativeVideo.currentTime;
            }
            updateTimeDisplay();
        }
    }, 100);
}

// ==========================================================================
// Video Grounding RAG Query Execution
// ==========================================================================

async function searchVideo(queryText) {
    if (!queryText || !queryText.trim()) return;

    if (elements.resultsLoading) elements.resultsLoading.style.display = 'flex';
    if (elements.resultsContainer) elements.resultsContainer.innerHTML = '';

    try {
        const resp = await fetch('/api/rag/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: queryText.trim(),
                alpha: 0.7,
                use_query_expansion: true
            })
        });

        const data = await resp.json();
        if (elements.resultsLoading) elements.resultsLoading.style.display = 'none';

        if (data.primary_grounding) {
            renderResults(data);
            highlightTimeline(data.primary_grounding);

            // Auto-play segment found
            playSegment(data.primary_grounding.start, data.primary_grounding.end);
        } else {
            if (elements.resultsContainer) {
                elements.resultsContainer.innerHTML = `
                    <div class="res-card">
                        <p style="color: var(--text-muted); font-size: 0.9rem;">Không tìm thấy đoạn video phù hợp cho câu hỏi này.</p>
                    </div>
                `;
            }
        }
    } catch (err) {
        if (elements.resultsLoading) elements.resultsLoading.style.display = 'none';
        console.error('Search error:', err);
    }
}

function renderResults(data) {
    try {
        const pg = data.primary_grounding || {};
        const citations = data.evidence_citations || [];
        const alts = data.alternative_moments || [];

        let citationsHtml = '';
        citations.forEach(c => {
            citationsHtml += `
                <div class="citation-row" onclick="playSegment(${c.start}, ${c.end})">
                    <span class="time-pill">
                        <span class="material-symbols-outlined" style="font-size: 14px; vertical-align: middle; margin-right: 2px;">play_arrow</span>${c.timestamp_str}
                    </span>
                    <span class="quote-text">"${c.quote}"</span>
                </div>
            `;
        });

        // Parse OCR keywords safely whether returned as Array or comma-delimited String
        let keywords = [];
        if (Array.isArray(pg.ocr_keywords)) {
            keywords = pg.ocr_keywords;
        } else if (typeof pg.ocr_keywords === 'string' && pg.ocr_keywords.trim()) {
            keywords = pg.ocr_keywords.split(',').map(s => s.trim()).filter(Boolean);
        }

        let visionPanelHtml = '';
        if (pg.keyframe_url) {
            visionPanelHtml = `
                <div class="res-vision-panel">
                    <div class="keyframe-card">
                        <img src="${pg.keyframe_url}" alt="Slide Keyframe" class="keyframe-thumb" loading="lazy">
                        <div class="keyframe-caption">
                            <span class="material-symbols-outlined" style="font-size: 14px; vertical-align: middle;">image</span>
                            <span>Keyframe @ ${pg.start_str || '00:00'}</span>
                        </div>
                    </div>
                    ${keywords.length > 0 ? `
                        <div class="ocr-keywords-box">
                            <div class="ocr-title">
                                <span class="material-symbols-outlined" style="font-size: 15px; vertical-align: middle; color: var(--accent-purple);">document_scanner</span>
                                <span>Từ khóa nhận diện từ Slide:</span>
                            </div>
                            <div class="ocr-tags">
                                ${keywords.map(kw => `<span class="ocr-tag">${kw}</span>`).join('')}
                            </div>
                        </div>
                    ` : ''}
                </div>
            `;
        }

        let altsHtml = '';
        if (alts.length > 0) {
            altsHtml = `
                <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border-color);">
                    <span style="font-size: 0.8rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Các phân đoạn thảo luận liên quan khác:</span>
                    <div style="display: flex; flex-direction: column; gap: 6px; margin-top: 8px;">
                        ${alts.map(a => `
                            <div class="alt-row" onclick="playSegment(${a.start}, ${a.end})">
                                <span class="time-pill alt-time">
                                    <span class="material-symbols-outlined" style="font-size: 14px; vertical-align: middle; margin-right: 2px;">play_arrow</span>${a.start_str} - ${a.end_str}
                                </span>
                                <span style="font-size: 0.85rem; color: var(--text-main); font-weight: 500;">${a.slide_title || a.title || a.snippet || 'Đoạn bài giảng'}</span>
                                <span style="font-size: 0.78rem; color: var(--text-muted); margin-left: auto;">(${Math.round((a.score || 0) * 100)}%)</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        const rawAnswer = data.answer || '';
        const formattedAnswer = rawAnswer
            .replace(/\n\n/g, '<br><br>')
            .replace(/> (.*?)(<br>|$)/g, '<blockquote>$1</blockquote>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');

        if (elements.resultsContainer) {
            elements.resultsContainer.innerHTML = `
                <div class="res-card">
                    <div class="res-header">
                        <div>
                            <span class="res-title">
                                <span class="material-symbols-outlined" style="font-size: 20px; vertical-align: middle; color: var(--primary-cyan); margin-right: 4px;">verified</span>${pg.slide_title || 'Phân đoạn trọng tâm'}
                            </span>
                            <span style="font-size: 0.8rem; color: var(--text-muted); margin-left: 8px;">(Độ tin cậy: ${Math.round((pg.confidence || 0) * 100)}%)</span>
                        </div>
                        <button class="res-btn-play" onclick="playSegment(${pg.start || 0}, ${pg.end || 0})">
                            <span class="material-symbols-outlined" style="font-size: 16px; vertical-align: middle; margin-right: 4px;">play_arrow</span>Phát Đoạn [${pg.start_str || '00:00'} - ${pg.end_str || '00:00'}] (${pg.duration || 0}s)
                        </button>
                    </div>

                    ${visionPanelHtml}

                    <div class="res-answer-box">
                        ${formattedAnswer}
                    </div>

                    ${citationsHtml ? `
                        <div style="font-size: 0.82rem; font-weight: 700; color: var(--text-muted); margin-top: 6px; display: flex; align-items: center; gap: 4px;">
                            <span class="material-symbols-outlined" style="font-size: 16px;">format_quote</span> Lời giảng chi tiết (Bấm để nghe đúng câu):
                        </div>
                        <div class="res-citations-list">
                            ${citationsHtml}
                        </div>
                    ` : ''}

                    ${altsHtml}
                </div>
            `;
        }
    } catch (err) {
        console.error('Error rendering results:', err);
        if (elements.resultsContainer) {
            elements.resultsContainer.innerHTML = `
                <div class="res-card">
                    <p style="color: #ef4444;">Lỗi hiển thị kết quả: ${err.message}</p>
                </div>
            `;
        }
    }
}

function highlightTimeline(pg) {
    if (!elements.timelineMarkers) return;
    elements.timelineMarkers.innerHTML = '';
    const leftPct = (pg.start / (state.totalDuration || 1)) * 100;
    const widthPct = Math.max(2, ((pg.end - pg.start) / (state.totalDuration || 1)) * 100);

    const mark = document.createElement('div');
    mark.className = 'marker-highlight';
    mark.style.left = `${leftPct}%`;
    mark.style.width = `${widthPct}%`;
    mark.onclick = (e) => {
        e.stopPropagation();
        playSegment(pg.start, pg.end);
    };
    elements.timelineMarkers.appendChild(mark);
}

window.playSegment = function(start, end) {
    state.activeSegment = { start: start, end: end };
    seekTo(start);
    setPlayState(true);
};

// ==========================================================================
// Setup & Lecture Loading
// ==========================================================================

async function loadLecture() {
    try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        state.totalDuration = data.current_video?.total_duration || 0;

        if (data.current_video && data.current_video.has_video_file) {
            state.playerMode = 'native';
            if (elements.dropzoneStage) elements.dropzoneStage.style.display = 'none';
            if (elements.nativeVideo) {
                elements.nativeVideo.style.display = 'block';
                elements.nativeVideo.src = `/api/video-stream/${data.current_video.id}`;
                elements.nativeVideo.ontimeupdate = () => {
                    state.currentTime = elements.nativeVideo.currentTime;
                    updateTimeDisplay();
                };
            }
            if (elements.btnChangeVideo) elements.btnChangeVideo.style.display = 'inline-flex';
        } else {
            state.playerMode = 'virtual';
            if (elements.dropzoneStage) elements.dropzoneStage.style.display = 'flex';
            if (elements.nativeVideo) {
                elements.nativeVideo.style.display = 'none';
                elements.nativeVideo.src = '';
            }
            if (elements.btnChangeVideo) elements.btnChangeVideo.style.display = 'none';
        }

        const subResp = await fetch('/api/subtitles');
        const subData = await subResp.json();
        state.subtitles = subData.subtitles || [];

        updateTimeDisplay();
    } catch (e) {
        console.error('loadLecture error:', e);
    }
}

// ==========================================================================
// Upload & YouTube Handlers (Dropzone Central Hub)
// ==========================================================================

async function uploadVideoFile(file) {
    if (!file) return;
    showStageLoading(
        'Đang xử lý & nhận dạng video bài giảng...',
        'Hệ thống tự động sử dụng Groq Whisper Cloud (nhanh 300x) để chuyển giọng nói thành phụ đề...'
    );

    const fd = new FormData();
    fd.append('video_file', file);
    fd.append('title', file.name.replace(/\.[^/.]+$/, ''));

    try {
        const resp = await fetch('/api/upload', { method: 'POST', body: fd });
        const data = await resp.json();
        hideStageLoading();
        if (data.success) {
            await loadLecture();
            seekTo(0);
            if (elements.resultsContainer) elements.resultsContainer.innerHTML = '';
            if (elements.timelineMarkers) elements.timelineMarkers.innerHTML = '';
        } else {
            alert('Lỗi tải video: ' + (data.detail || 'Không thể xử lý'));
        }
    } catch (err) {
        hideStageLoading();
        alert('Lỗi kết nối khi tải video: ' + err.message);
    }
}

async function loadYoutubeUrl(url) {
    if (!url || !url.trim()) return;
    showStageLoading(
        'Đang tải video & phụ đề từ YouTube...',
        'Trích xuất phụ đề và đồng bộ hóa các mốc thời gian bài giảng...'
    );

    try {
        const resp = await fetch('/api/youtube/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url.trim() })
        });
        const data = await resp.json();
        hideStageLoading();
        if (data.success) {
            await loadLecture();
            seekTo(0);
            if (elements.resultsContainer) elements.resultsContainer.innerHTML = '';
            if (elements.timelineMarkers) elements.timelineMarkers.innerHTML = '';
            if (elements.dropYoutubeInput) elements.dropYoutubeInput.value = '';
        } else {
            alert('Lỗi: ' + (data.detail || 'Không thể trích xuất phụ đề YouTube'));
        }
    } catch (err) {
        hideStageLoading();
        alert('Lỗi kết nối YouTube: ' + err.message);
    }
}

// ==========================================================================
// Event Listeners Initialization
// ==========================================================================

function initEvents() {
    // 1. Play / Pause
    elements.btnPlayPause?.addEventListener('click', () => setPlayState(!state.isPlaying));

    // 2. Timeline Click to Seek
    elements.timelineTrack?.addEventListener('click', (e) => {
        const rect = elements.timelineTrack.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        seekTo(ratio * state.totalDuration);
    });

    // 3. Search query input
    elements.btnSearch?.addEventListener('click', () => searchVideo(elements.queryInput?.value));
    elements.queryInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') searchVideo(elements.queryInput?.value);
    });

    // 4. Quick chips
    elements.chips?.forEach(chip => {
        chip.addEventListener('click', () => {
            const q = chip.getAttribute('data-query');
            if (elements.queryInput) elements.queryInput.value = q;
            searchVideo(q);
        });
    });

    // 5. Dropzone: Drag & Drop Files directly into video stage
    if (elements.videoWrapper) {
        ['dragenter', 'dragover'].forEach(evt => {
            elements.videoWrapper.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                elements.dropzoneCard?.classList.add('drag-active');
            }, false);
        });

        ['dragleave', 'drop'].forEach(evt => {
            elements.videoWrapper.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                elements.dropzoneCard?.classList.remove('drag-active');
            }, false);
        });

        elements.videoWrapper.addEventListener('drop', (e) => {
            const files = e.dataTransfer?.files;
            if (files && files.length > 0) {
                const file = files[0];
                if (file.name.match(/\.(mp4|webm)$/i)) {
                    uploadVideoFile(file);
                } else {
                    alert('Vui lòng chọn hoặc kéo thả file video .mp4 hoặc .webm!');
                }
            }
        });
    }

    // 6. Dropzone: File Browse button
    elements.btnBrowseVideo?.addEventListener('click', () => {
        elements.dropFileInput?.click();
    });

    elements.dropFileInput?.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            uploadVideoFile(e.target.files[0]);
            e.target.value = ''; // reset for re-selecting same file if needed
        }
    });

    // 7. Dropzone: YouTube URL Submit
    const handleYtSubmit = () => {
        const url = elements.dropYoutubeInput?.value.trim();
        if (!url) {
            alert('Vui lòng nhập đường dẫn video YouTube!');
            return;
        }
        loadYoutubeUrl(url);
    };

    elements.btnDropYoutubeSubmit?.addEventListener('click', handleYtSubmit);
    elements.dropYoutubeInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') handleYtSubmit();
    });

    // 8. Floating "Đổi Video" button (when a video is currently active)
    elements.btnChangeVideo?.addEventListener('click', () => {
        setPlayState(false);
        if (elements.nativeVideo) elements.nativeVideo.style.display = 'none';
        if (elements.btnChangeVideo) elements.btnChangeVideo.style.display = 'none';
        if (elements.dropzoneStage) elements.dropzoneStage.style.display = 'flex';
    });

    // 9. Groq Modal: Open & Close & Save
    elements.headerGroqBtn?.addEventListener('click', () => {
        if (elements.groqModal) elements.groqModal.style.display = 'flex';
        elements.inputModalGroqKey?.focus();
    });

    elements.btnCloseGroq?.addEventListener('click', () => {
        if (elements.groqModal) elements.groqModal.style.display = 'none';
    });

    elements.btnCancelGroq?.addEventListener('click', () => {
        if (elements.groqModal) elements.groqModal.style.display = 'none';
    });

    elements.btnSaveModalGroqKey?.addEventListener('click', async () => {
        const k = elements.inputModalGroqKey?.value.trim();
        if (!k) {
            alert('Vui lòng nhập API Key từ console.groq.com!');
            return;
        }
        try {
            const resp = await fetch('/api/groq/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_key: k })
            });
            const res = await resp.json();
            if (res.success) {
                alert('Đã lưu và kích hoạt Groq API Key thành công! Tốc độ trích xuất đạt ~300x.');
                if (elements.inputModalGroqKey) elements.inputModalGroqKey.value = '';
                if (elements.groqModal) elements.groqModal.style.display = 'none';
                await refreshGroqStatus();
            } else {
                alert('Lỗi: ' + (res.detail || 'Không thể lưu key'));
            }
        } catch (err) {
            alert('Lỗi kết nối: ' + err.message);
        }
    });

    // Close Groq modal if clicked on dark backdrop
    elements.groqModal?.addEventListener('click', (e) => {
        if (e.target === elements.groqModal) {
            elements.groqModal.style.display = 'none';
        }
    });
}

// ==========================================================================
// Groq Status Synchronization
// ==========================================================================

async function refreshGroqStatus() {
    try {
        const resp = await fetch('/api/groq/config');
        const data = await resp.json();

        // 1. Update Header Button Badge
        if (elements.headerGroqBadge) {
            if (data.has_key) {
                elements.headerGroqBadge.textContent = '● Active';
                elements.headerGroqBadge.className = 'header-groq-badge active';
                elements.headerGroqBadge.title = `Groq API Key đang kích hoạt: ${data.masked_key}`;
            } else {
                elements.headerGroqBadge.textContent = '● Deactive';
                elements.headerGroqBadge.className = 'header-groq-badge inactive';
                elements.headerGroqBadge.title = 'Bấm để cấu hình Groq API Key miễn phí';
            }
        }

        // 2. Update Inside Modal Status Badge
        if (elements.groqModalStatusBadge) {
            if (data.has_key) {
                elements.groqModalStatusBadge.textContent = `Active`;
                elements.groqModalStatusBadge.style.background = 'rgba(16, 185, 129, 0.18)';
                elements.groqModalStatusBadge.style.color = '#34d399';
                elements.groqModalStatusBadge.style.border = '1px solid #10b981';
                if (elements.inputModalGroqKey && !elements.inputModalGroqKey.value) {
                    elements.inputModalGroqKey.placeholder = `${data.masked_key}`;
                }
            } else {
                elements.groqModalStatusBadge.textContent = 'Deactive';
                elements.groqModalStatusBadge.style.background = 'rgba(245, 158, 11, 0.18)';
                elements.groqModalStatusBadge.style.color = '#fbbf24';
                elements.groqModalStatusBadge.style.border = '1px solid rgba(245, 158, 11, 0.4)';
            }
        }
    } catch (e) {
        console.warn('Groq config fetch error:', e);
    }
}

// ==========================================================================
// App Initialization
// ==========================================================================

window.addEventListener('DOMContentLoaded', async () => {
    initEvents();
    startClock();
    await refreshGroqStatus();
    await loadLecture();
});
