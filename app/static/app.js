/**
 * VocabMaster Frontend Application Logic
 * Supports 6 Learning Modes, Audio Speech Synthesis, and SRS SM-2 Tracking.
 */

const BG_PRESETS = {
  workspace: 'https://images.unsplash.com/photo-1499750310107-5fef28a66643?q=80&w=1600&auto=format&fit=crop',
  nature: 'https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?q=80&w=1600&auto=format&fit=crop',
  night: 'https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=1600&auto=format&fit=crop',
  sunset: 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?q=80&w=1600&auto=format&fit=crop'
};

// Application State
const state = {
  words: [],
  dueWords: [],
  currentCardIndex: 0,
  isCardFlipped: false,
  
  // SRS Session State
  srsIndex: 0,
  isSrsRevealed: false,

  // Quiz State
  quizQuestions: [],
  quizIndex: 0,
  quizScore: 0,
  quizStreak: 0,
  currentQuizWord: null,

  // Fill in the Blank State
  fillQuestions: [],
  fillIndex: 0,
  fillScore: 0,
  currentFillWord: null,

  // Matching Game State
  matchCards: [],
  selectedMatchCards: [],
  matchedPairsCount: 0,
  matchTimerInterval: null,
  matchSeconds: 0,

  // Reader State
  currentDoc: null,
  activeLookupWord: null,

  // Thesaurus State
  thesaurusData: null,

  // Settings
  geminiApiKey: localStorage.getItem('vocab_gemini_key') || '',
  geminiModel: localStorage.getItem('vocab_gemini_model') || 'gemini-3.7-flash',
  ttsVoice: localStorage.getItem('vocab_tts_voice') || 'en-US',
  themeMode: localStorage.getItem('vocabmaster_theme_mode') || 'system',
  themeColor: localStorage.getItem('vocabmaster_theme_color') || 'indigo',

  // Background Settings
  bgUrl: localStorage.getItem('vocabmaster_bg_url') || '',
  bgOverlay: parseInt(localStorage.getItem('vocabmaster_bg_overlay') || '35', 10),
  bgBlur: parseInt(localStorage.getItem('vocabmaster_bg_blur') || '0', 10),
  bgPreset: localStorage.getItem('vocabmaster_bg_preset') || 'none',

  // Submission & Timeout Safety Locks
  isCardSubmitting: false,
  srsReviewedCount: 0,
  matchTimeouts: []
};

// BỘ TĂNG TỐC: Cache kết quả thesaurus phía client — tra lại từ đã tra sẽ tức thời
const thesaurusCache = new Map();
const THESAURUS_CACHE_MAX = 200;
const translateCache = new Map();
const TRANSLATE_CACHE_MAX = 1000;

function escapeRegex(str) {
  if (!str) return '';
  return String(str).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function closeAllModals() {
  const upload = document.getElementById('modal-upload');
  if (upload && !upload.classList.contains('hidden')) closeUploadModal();

  const settings = document.getElementById('modal-settings');
  if (settings && !settings.classList.contains('hidden')) closeSettingsModal();

  const addWord = document.getElementById('modal-add-word');
  if (addWord && !addWord.classList.contains('hidden')) closeAddWordModal();

  const premade = document.getElementById('premade-modal');
  if (premade && !premade.classList.contains('hidden')) closePremadeModal();
}

// ==========================================
// Initialization & Global Event Listeners
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
  // Initialize Theme & Background
  initTheme();
  initBackground();

  // Load settings into inputs
  const keyInput = document.getElementById('settings-gemini-key');
  if (keyInput) keyInput.value = state.geminiApiKey;
  const modelInput = document.getElementById('settings-gemini-model');
  if (modelInput) modelInput.value = state.geminiModel;
  const voiceInput = document.getElementById('settings-tts-voice');
  if (voiceInput) voiceInput.value = state.ttsVoice;

  // Backdrop click listeners for all modals
  ['modal-upload', 'modal-settings', 'modal-add-word', 'premade-modal'].forEach(modalId => {
    const modal = document.getElementById(modalId);
    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) {
          if (modalId === 'modal-upload') closeUploadModal();
          else if (modalId === 'modal-settings') closeSettingsModal();
          else if (modalId === 'modal-add-word') closeAddWordModal();
          else if (modalId === 'premade-modal') closePremadeModal();
        }
      });
    }
  });

  // Fetch initial data
  loadStats();
  loadWords();
  loadDocuments();

  // Keyboard Shortcuts
  document.addEventListener('keydown', handleGlobalKeydown);
});

function handleGlobalKeydown(e) {
  // Close open modal on Escape (even if focus is currently inside an input)
  if (e.key === 'Escape') {
    closeAllModals();
    return;
  }

  // Ignore shortcuts when typing in inputs/textareas
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  // Ignore shortcuts with modifier keys (Ctrl, Alt, Meta)
  if (e.ctrlKey || e.altKey || e.metaKey) return;

  // If any modal is open, ignore learning shortcuts
  const openModal = document.querySelector('#modal-upload:not(.hidden), #modal-settings:not(.hidden), #modal-add-word:not(.hidden), #premade-modal:not(.hidden)');
  if (openModal) return;

  const activePane = document.querySelector('.tab-pane:not(.hidden)');
  if (!activePane) return;

  if (activePane.id === 'pane-flashcards') {
    if (e.code === 'Space') {
      e.preventDefault();
      flipCurrentCard();
    } else if (e.code === 'ArrowRight') {
      nextCard();
    } else if (e.code === 'ArrowLeft') {
      prevCard();
    } else if (e.key === '1') {
      rateCard(1);
    } else if (e.key === '2') {
      rateCard(2);
    } else if (e.key === '3') {
      rateCard(3);
    } else if (e.key === '4') {
      rateCard(4);
    } else if (e.key === '5') {
      rateCard(5);
    }
  } else if (activePane.id === 'pane-srs') {
    if (e.code === 'Space') {
      e.preventDefault();
      if (!state.isSrsRevealed) revealSrsAnswer();
    } else if (state.isSrsRevealed) {
      if (e.key === '1') submitSrsReview(1);
      else if (e.key === '2') submitSrsReview(2);
      else if (e.key === '3') submitSrsReview(3);
      else if (e.key === '4') submitSrsReview(4);
      else if (e.key === '5') submitSrsReview(5);
    }
  } else if (activePane.id === 'pane-quiz') {
    const nextBtn = document.getElementById('quiz-next-btn');
    if (nextBtn && !nextBtn.classList.contains('hidden')) {
      if (e.code === 'Enter' || e.code === 'Space') {
        e.preventDefault();
        nextQuizQuestion();
        return;
      }
    }
    const optButtons = document.querySelectorAll('.quiz-opt-btn');
    if (optButtons.length > 0 && !optButtons[0].disabled) {
      let optIdx = -1;
      const key = e.key.toUpperCase();
      if (key === 'A' || key === '1') optIdx = 0;
      else if (key === 'B' || key === '2') optIdx = 1;
      else if (key === 'C' || key === '3') optIdx = 2;
      else if (key === 'D' || key === '4') optIdx = 3;
      if (optIdx >= 0 && optIdx < optButtons.length) {
        e.preventDefault();
        optButtons[optIdx].click();
      }
    }
  } else if (activePane.id === 'pane-fill') {
    const nextBtn = document.getElementById('fill-next-btn');
    if (nextBtn && !nextBtn.classList.contains('hidden')) {
      if (e.code === 'Enter' || e.code === 'Space') {
        e.preventDefault();
        nextFillQuestion();
      }
    }
  }
}

// ==========================================
// Navigation & Tab Management
// ==========================================
function switchTab(tabName) {
  // Clear match timer & timeouts when navigating away
  if (tabName !== 'match') {
    if (state.matchTimerInterval) {
      clearInterval(state.matchTimerInterval);
      state.matchTimerInterval = null;
    }
    clearMatchTimeouts();
  }

  // Clear SRS typing timer when leaving SRS mode
  if (tabName !== 'srs') {
    if (typeof srsTypingTimeout !== 'undefined' && srsTypingTimeout) {
      clearTimeout(srsTypingTimeout);
      srsTypingTimeout = null;
    }
  }

  // Abort pending Thesaurus lookup if navigating away
  if (tabName !== 'thesaurus' && typeof thesaurusAbortCtrl !== 'undefined' && thesaurusAbortCtrl) {
    thesaurusAbortCtrl.abort();
    thesaurusAbortCtrl = null;
  }

  // Abort pending Reader document loading if navigating away
  if (tabName !== 'reader' && typeof readerAbortCtrl !== 'undefined' && readerAbortCtrl) {
    readerAbortCtrl.abort();
    readerAbortCtrl = null;
  }

  // Clear pending vocab filter debounce timer
  if (tabName !== 'manage' && typeof vocabDebounceTimer !== 'undefined' && vocabDebounceTimer) {
    clearTimeout(vocabDebounceTimer);
    vocabDebounceTimer = null;
  }

  // Update Tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  const activeBtn = document.getElementById(`tab-btn-${tabName}`);
  if (activeBtn) {
    activeBtn.classList.add('active');
  }

  // Hide all panes
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));

  // Show selected pane
  const pane = document.getElementById(`pane-${tabName}`);
  if (pane) pane.classList.remove('hidden');

  // Trigger mode-specific initializers
  if (tabName === 'flashcards') {
    renderFlashcard();
  } else if (tabName === 'srs') {
    initSrsSession();
  } else if (tabName === 'quiz') {
    initQuiz();
  } else if (tabName === 'fill') {
    initFillBlank();
  } else if (tabName === 'match') {
    startMatchGame();
  } else if (tabName === 'thesaurus') {
    initThesaurus();
  } else if (tabName === 'manage') {
    renderVocabTable();
  }

  lucide.createIcons();

  // Smooth scroll pane into view on mobile/tablet devices
  if (window.innerWidth < 1024 && pane) {
    pane.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ==========================================
// API Calls & Data Fetching
// ==========================================
async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    document.getElementById('stat-total').textContent = data.total_words || 0;
    document.getElementById('stat-due').textContent = data.due_reviews || 0;
    document.getElementById('stat-learning').textContent = data.learning_words || 0;
    document.getElementById('stat-mastered').textContent = data.mastered_words || 0;

    const navSrs = document.getElementById('nav-badge-srs');
    if (navSrs) navSrs.textContent = data.due_reviews || 0;
    const navTotal = document.getElementById('nav-badge-total');
    if (navTotal) navTotal.textContent = data.total_words || 0;
  } catch (err) {
    console.error('Error fetching stats:', err);
  }
}

async function loadWords() {
  try {
    const res = await fetch('/api/words');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.words = data.words || [];
    const navTotal = document.getElementById('nav-badge-total');
    if (navTotal) navTotal.textContent = state.words.length;
    renderFlashcard();
    renderVocabTable();
  } catch (err) {
    console.error('Error fetching words:', err);
    const tbody = document.getElementById('vocab-table-body');
    if (tbody && (!state.words || state.words.length === 0)) {
      tbody.innerHTML = `
        <tr>
          <td colspan="7" class="py-12 text-center text-rose-500 dark:text-rose-400">
            <div class="flex flex-col items-center justify-center gap-2.5">
              <span class="font-semibold text-sm">Không thể tải danh sách từ vựng từ máy chủ.</span>
              <button onclick="loadWords()" class="px-3.5 py-1.5 bg-indigo-50 dark:bg-zinc-800 text-indigo-600 dark:text-indigo-400 text-xs font-bold rounded-xl hover:bg-indigo-100 transition cursor-pointer">Thử lại</button>
            </div>
          </td>
        </tr>
      `;
    }
  }
}

async function loadDocuments() {
  try {
    const res = await fetch('/api/documents');
    const data = await res.json();
    const select = document.getElementById('reader-doc-select');
    if (!select) return;
    select.innerHTML = '<option value="">-- Chọn tài liệu đã tải lên --</option>';
    data.documents.forEach(doc => {
      const opt = document.createElement('option');
      opt.value = doc.id;
      opt.textContent = `${doc.filename} (${doc.word_count} từ)`;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error('Error loading documents:', err);
  }
}

// ==========================================
// Audio Pronunciation Engine (Web Audio API + DOM Audio + Web Speech)
// ==========================================
let audioCtx = null;
let activeWebAudioSource = null;
let activeSpeechUtterance = null; // Giữ tham chiếu chống Chromium Garbage Collection bug

function getAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume().catch(() => {});
  }
  return audioCtx;
}

// Tự động mở khóa AudioContext khi người dùng chạm/click vào bất kỳ đâu trên trang
document.addEventListener('click', () => {
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume().catch(() => {});
  }
}, { passive: true });

async function playWithWebAudio(audioUrl) {
  const ctx = getAudioContext();
  if (!ctx) throw new Error('AudioContext không được hỗ trợ');
  if (ctx.state === 'suspended') {
    await ctx.resume();
  }

  // Dừng nguồn âm thanh Web Audio đang phát trước đó
  if (activeWebAudioSource) {
    try {
      activeWebAudioSource.stop();
      activeWebAudioSource.disconnect();
    } catch (e) {}
    activeWebAudioSource = null;
  }

  const res = await fetch(audioUrl);
  if (!res.ok) throw new Error(`Lỗi máy chủ TTS (HTTP ${res.status})`);
  const arrayBuf = await res.arrayBuffer();
  if (!arrayBuf || arrayBuf.byteLength < 50) {
    throw new Error('Dữ liệu âm thanh nhận được không hợp lệ');
  }

  // Giải mã định dạng MP3 thành PCM AudioBuffer trong bộ nhớ
  const audioBuf = await new Promise((resolve, reject) => {
    ctx.decodeAudioData(arrayBuf.slice(0), resolve, (err) => {
      reject(err || new Error('Không thể giải mã file MP3'));
    });
  });

  const source = ctx.createBufferSource();
  source.buffer = audioBuf;
  source.connect(ctx.destination);
  activeWebAudioSource = source;
  source.onended = () => {
    if (activeWebAudioSource === source) activeWebAudioSource = null;
  };
  source.start(0);
  return source;
}

async function playWithDomAudio(audioUrl) {
  let player = document.getElementById('global-tts-audio');
  if (!player) {
    player = new Audio();
    player.id = 'global-tts-audio';
    document.body.appendChild(player);
  }
  try {
    player.pause();
    player.currentTime = 0;
  } catch (e) {}
  player.src = audioUrl;
  return await player.play();
}

async function speakText(text, langOverride = null) {
  if (!text) return;
  const clean = text.trim();
  if (!clean) return;

  const targetLang = langOverride || state.ttsVoice || 'en-US';
  const encodedText = encodeURIComponent(clean);
  const audioUrl = `/api/tts?text=${encodedText}&lang=${encodeURIComponent(targetLang)}`;

  // Dừng Web Speech API nếu đang đọc dở
  if ('speechSynthesis' in window) {
    try { window.speechSynthesis.cancel(); } catch (e) {}
  }

  // Ưu tiên 1: Web Audio API (Chơi trực tiếp qua DSP, hoàn toàn miễn nhiễm với Brave Shields / Autoplay Block)
  try {
    await playWithWebAudio(audioUrl);
    return;
  } catch (errWebAudio) {
    console.warn('Web Audio API gặp sự cố, chuyển sang DOM Audio:', errWebAudio);
  }

  // Ưu tiên 2: DOM Audio Player (Element gắn sẵn trong trang)
  try {
    await playWithDomAudio(audioUrl);
    return;
  } catch (errDom) {
    console.warn('DOM Audio gặp sự cố, chuyển sang Web Speech Fallback:', errDom);
  }

  // Ưu tiên 3: Web Speech API (Dự phòng khi mất kết nối máy chủ)
  speakWithWebSpeech(clean, targetLang);
}

function speakWithWebSpeech(text, lang) {
  if (!('speechSynthesis' in window)) {
    showToast('Thiết bị không hỗ trợ phát âm');
    return;
  }
  try {
    window.speechSynthesis.cancel();
    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
    }
    const utterance = new SpeechSynthesisUtterance(text);
    activeSpeechUtterance = utterance;
    utterance.lang = lang || 'en-US';
    utterance.rate = 0.9;
    utterance.onend = () => { activeSpeechUtterance = null; };
    utterance.onerror = (e) => {
      console.error('Web Speech error:', e);
      activeSpeechUtterance = null;
    };
    window.speechSynthesis.speak(utterance);
  } catch (e) {
    console.error('Speech synthesis error:', e);
  }
}

// Kiểm tra loa máy tính bằng tiếng chuông tổng hợp (Không cần mạng, không cần file)
function testSpeakerBeep() {
  const diag = document.getElementById('tts-diagnostic-msg');
  try {
    const ctx = getAudioContext();
    if (ctx.state === 'suspended') {
      ctx.resume().catch(() => {});
    }

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(587.33, ctx.currentTime); // Nốt D5
    osc.frequency.setValueAtTime(880, ctx.currentTime + 0.12); // Nốt A5

    gain.gain.setValueAtTime(0.25, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.42);

    showToast('🔔 Đang phát tiếng chuông kiểm tra loa máy tính...');
    if (diag) {
      diag.textContent = '🔔 Đã phát tiếng chuông! Nếu bạn nghe thấy, loa và âm lượng hoạt động tốt 100%.';
      diag.className = 'text-[11px] text-emerald-600 dark:text-emerald-400 mt-1.5 font-medium';
    }
  } catch (err) {
    showToast('Lỗi khi mở thiết bị âm thanh: ' + err.message);
    if (diag) {
      diag.textContent = '⚠️ Trình duyệt chặn thiết bị âm thanh: ' + err.message;
      diag.className = 'text-[11px] text-rose-600 dark:text-rose-400 mt-1.5 font-medium';
    }
  }
}

async function testCurrentTtsVoice() {
  const select = document.getElementById('settings-tts-voice');
  const lang = select ? select.value : (state.ttsVoice || 'en-US');
  const diag = document.getElementById('tts-diagnostic-msg');
  const phrases = {
    'en-US': 'Hello! Welcome to VocabMaster. Ready to learn English?',
    'en-GB': 'Hello! Welcome to VocabMaster. Splendid day to learn vocabulary.',
    'en-AU': 'G\'day! Welcome to VocabMaster. Let\'s practice your vocabulary.'
  };
  const phrase = phrases[lang] || 'Hello! Welcome to VocabMaster.';
  
  showToast(`🔊 Đang phát thử giọng (${lang})...`);
  if (diag) {
    diag.textContent = `🔊 Đang tải luồng âm thanh (${lang})...`;
    diag.className = 'text-[11px] text-indigo-600 dark:text-indigo-400 mt-1.5 font-medium';
  }

  try {
    await speakText(phrase, lang);
    if (diag) {
      diag.textContent = `✅ Đã phát âm thanh giọng (${lang}) thành công!`;
      diag.className = 'text-[11px] text-emerald-600 dark:text-emerald-400 mt-1.5 font-medium';
    }
  } catch (err) {
    if (diag) {
      diag.textContent = `⚠️ Lỗi phát âm: ${err.message}. Kiểm tra xem Tab có bị Mute không.`;
      diag.className = 'text-[11px] text-rose-600 dark:text-rose-400 mt-1.5 font-medium';
    }
  }
}

const GLUED_VN_MAP = {
  'vềnhà': 'về nhà', 'dựán': 'dự án', 'kếtoán': 'kế toán', 'tụtập': 'tụ tập',
  'họhàng': 'họ hàng', 'tựhào': 'tự hào', 'cờbàn': 'cờ bàn', 'thểdục': 'thể dục',
  'sởthích': 'sở thích', 'thểthao': 'thể thao', 'dựbáo': 'dự báo', 'dữliệu': 'dữ liệu',
  'kỹthuật': 'kỹ thuật', 'giữliên': 'giữ liên', 'dễsửdụng': 'dễ sử dụng', 'dịứng': 'dị ứng',
  'chếđộăn': 'chế độ ăn', 'trởnên': 'trở nên', 'nghỉngơi': 'nghỉ ngơi', 'ởnông': 'ở nông',
  'bịnội': 'bị nội', 'trảtiền': 'trả tiền', 'tủlạnh': 'tủ lạnh', 'vệmôi': 'vệ môi',
  'xảrác': 'xả rác', 'tựnhiên': 'tự nhiên', 'ởnước': 'ở nước', 'thuởxưa': 'thuở xưa',
  'đồmặc': 'đồ mặc', 'quảcam': 'quả cam', 'sựrơi': 'sự rơi', 'sựthưởng': 'sự thưởng',
  'sựviệc': 'sự việc', 'sựkiện': 'sự kiện', 'sựdựđoán': 'sự dự đoán', 'sựđi': 'sự đi',
  'giữlại': 'giữ lại', 'nhớđược': 'nhớ được', 'củamình': 'của mình', 'thểchất': 'thể chất',
  'tinhthần': 'tinh thần', 'sứckhỏe': 'sức khỏe', 'bệnhnhân': 'bệnh nhân',
  'thôngbáo': 'thông báo', 'họcsinh': 'học sinh', 'giáoviên': 'giáo viên',
  'đềán': 'đề án', 'từvựng': 'từ vựng', 'sốliệu': 'số liệu', 'thếgiới': 'thế giới',
  'thếkỷ': 'thế kỷ', 'kếhoạch': 'kế hoạch', 'hệthống': 'hệ thống', 'thịtrường': 'thị trường',
  'tổchức': 'tổ chức', 'sửdụng': 'sử dụng', 'nghềnghiệp': 'nghề nghiệp', 'kỳdiệu': 'kỳ diệu',
  'lễhội': 'lễ hội', 'đềcập': 'đề cập', 'đềnghị': 'đề nghị', 'bốtrí': 'bố trí',
  'chỉdẫn': 'chỉ dẫn', 'chỉtrích': 'chỉ trích', 'chỉđường': 'chỉ đường', 'tựtin': 'tự tin',
  'tựđộng': 'tự động', 'hồbơi': 'hồ bơi', 'bểbơi': 'bể bơi', 'cổhọng': 'cổ họng',
  'cờbạc': 'cờ bạc', 'củhành': 'củ hành', 'cửnhân': 'cử nhân', 'trởvề': 'trở về',
  'trởlại': 'trở lại', 'trợlý': 'trợ lý', 'trụsở': 'trụ sở', 'chủnhà': 'chủ nhà',
  'kẻthù': 'kẻ thù', 'nữhoàng': 'nữ hoàng', 'nghệsĩ': 'nghệ sĩ', 'nghỉhưu': 'nghỉ hưu',
  'nghỉviệc': 'nghỉ việc', 'nhỏgiọt': 'nhỏ giọt', 'phủđịnh': 'phủ định', 'quảhạch': 'quả hạch',
  'sởhữu': 'sở hữu', 'sợhãi': 'sợ hãi', 'thịlực': 'thị lực', 'thủđoạn': 'thủ đoạn',
  'thợchụp': 'thợ chụp', 'thểdiện': 'thể diện', 'hồsơ': 'hồ sơ'
};

function cleanVietnamese(text) {
  if (!text) return '';
  let s = (text.normalize ? text.normalize('NFC') : text).trim();
  for (const [glued, sep] of Object.entries(GLUED_VN_MAP)) {
    const reg = new RegExp(glued, 'gi');
    s = s.replace(reg, (match) => {
      if (match === match.toUpperCase()) return sep.toUpperCase();
      if (match[0] === match[0].toUpperCase()) return sep.charAt(0).toUpperCase() + sep.slice(1);
      return sep;
    });
  }
  s = s.replace(/(?<!\p{L})(sự|nghỉ|thuở|chỗ|nghề|thợ)(\p{L}{2,})(?!\p{L})/giu, '$1 $2');
  return s.replace(/\s+/g, ' ').trim();
}

function formatQuizChoice(text) {
  if (!text) return '';
  let clean = cleanVietnamese(text);
  // Loại bỏ các định nghĩa phụ dài dòng sau dấu chấm phẩy
  if (clean.includes(';')) {
    clean = clean.split(';')[0].trim();
  }
  // Rút gọn nếu có mệnh đề quá dài sau dấu phẩy
  if (clean.includes(',') && clean.length > 25) {
    clean = clean.split(',')[0].trim();
  }
  clean = clean.replace(/[,;.\s]+$/, '').trim();
  if (!clean) return text;
  // Viết hoa chữ cái đầu ĐỒNG NHẤT cho tất cả 4 phương án
  return clean.charAt(0).toUpperCase() + clean.slice(1);
}

function highlightWordInSentence(sentence, word) {
  if (!sentence || !word) return escapeHtml(sentence || '');
  const safeSentence = escapeHtml(sentence);
  const safeWord = escapeHtml(word);
  const escaped = safeWord.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const reg = new RegExp(`\\b(${escaped})\\b`, 'gi');
  return safeSentence.replace(reg, '<span class="font-bold not-italic text-indigo-600 dark:text-indigo-400 bg-indigo-100/70 dark:bg-indigo-950/60 px-1.5 py-0.5 rounded-lg border border-indigo-200/50 dark:border-indigo-800/50">$1</span>');
}

function formatCleanSentence(sentence, word) {
  if (!sentence || sentence.trim() === '') {
    return word ? `The term "${word}" is frequently used in this context.` : '';
  }
  let s = sentence.trim();
  // Nếu câu chứa dấu hiệu mục từ điển thô (phiên âm, Synonym, Antonym, (n):, ...)
  if (s.includes('Synonym:') || s.includes('Antonym:') || /\/[^\/\n]{2,}\//.test(s) || /\((n|v|adj|adv)\)/.test(s) || /:\s*[^\n]+/.test(s)) {
    const cleanMatches = s.match(/[A-Z][a-zA-Z0-9\s,\'\";:-]{15,}\./g);
    if (cleanMatches && cleanMatches.length > 0) {
      if (word) {
        const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const pat = new RegExp(`\\b${escaped}\\b`, 'i');
        for (const m of cleanMatches) {
          if (pat.test(m) && !m.includes(':') && !/\((n|v|adj|adv)\)/.test(m)) {
            return m.trim();
          }
        }
      }
      for (const m of cleanMatches) {
        if (!m.includes(':') && !/\((n|v|adj|adv)\)/.test(m)) {
          return m.trim();
        }
      }
    }
    return word ? `The word "${word}" is an important term in this study.` : '';
  }
  return s.replace(/\([^\)]*[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ][^\)]*\)/g, '').replace(/\s+/g, ' ').trim();
}

// ==========================================
// 1. FLASHCARDS 3D MODE
// ==========================================
function renderFlashcard() {
  const cardInner = document.getElementById('flashcard-inner');
  const jumpInput = document.getElementById('fc-jump-input');
  const totalCount = document.getElementById('fc-total-count');
  if (cardInner) {
    cardInner.classList.remove('is-flipped');
    state.isCardFlipped = false;
  }

  const rateButtons = document.querySelectorAll('#pane-flashcards .btn-rate-forgot, #pane-flashcards .btn-rate-vhard, #pane-flashcards .btn-rate-hard, #pane-flashcards .btn-rate-good, #pane-flashcards .btn-rate-easy');

  if (!state.words || state.words.length === 0) {
    rateButtons.forEach(b => { b.disabled = true; b.classList.add('opacity-40', 'cursor-not-allowed'); });
    const frontWord = document.getElementById('fc-front-word');
    if (frontWord) frontWord.textContent = 'Kho từ đang trống';
    const frontIpa = document.getElementById('fc-front-ipa');
    if (frontIpa) frontIpa.textContent = 'Hãy tải lên file PDF/Word hoặc thêm từ mới';
    const frontPos = document.getElementById('fc-front-pos');
    if (frontPos) frontPos.textContent = '--';
    const frontLevel = document.getElementById('fc-front-level');
    if (frontLevel) frontLevel.textContent = '--';

    const backWord = document.getElementById('fc-back-word');
    if (backWord) backWord.textContent = 'Kho từ đang trống';
    const backMeaning = document.getElementById('fc-back-meaning');
    if (backMeaning) backMeaning.textContent = 'Chưa có từ vựng nào được lưu';
    const backExample = document.getElementById('fc-back-example');
    if (backExample) backExample.textContent = 'Hãy tải lên tài liệu PDF hoặc Word để hệ thống tự động trích xuất từ vựng và câu ví dụ cho bạn.';

    if (jumpInput) { jumpInput.value = 0; jumpInput.max = 0; }
    if (totalCount) totalCount.textContent = '/ 0';
    return;
  }

  rateButtons.forEach(b => { b.disabled = false; b.classList.remove('opacity-40', 'cursor-not-allowed'); });

  if (state.currentCardIndex >= state.words.length) state.currentCardIndex = 0;
  if (state.currentCardIndex < 0) state.currentCardIndex = state.words.length - 1;

  const w = state.words[state.currentCardIndex];

  // Populate Front
  document.getElementById('fc-front-word').textContent = w.word;
  document.getElementById('fc-front-ipa').textContent = w.ipa || `/${w.word}/`;
  document.getElementById('fc-front-pos').textContent = w.pos || 'vocabulary';
  document.getElementById('fc-front-level').textContent = w.cefr_level || 'B1';
  
  if (jumpInput) {
    jumpInput.value = state.currentCardIndex + 1;
    jumpInput.max = state.words.length;
  }
  if (totalCount) {
    totalCount.textContent = `/ ${state.words.length}`;
  }

  // Populate Back
  document.getElementById('fc-back-word').textContent = w.word;
  document.getElementById('fc-back-meaning').textContent = cleanVietnamese(w.meaning_vi);
  const cleanSentence = formatCleanSentence(w.example_sentence, w.word);
  document.getElementById('fc-back-example').textContent = cleanSentence ? `"${cleanSentence}"` : `Example for ${w.word}`;
}

function jumpToCard(val) {
  if (!state.words || state.words.length === 0) return;
  const num = parseInt(val);
  if (isNaN(num)) return;
  state.currentCardIndex = Math.max(0, Math.min(num - 1, state.words.length - 1));
  renderFlashcard();
}

function flipCurrentCard() {
  if (!state.words || state.words.length === 0) {
    showToast('Kho từ đang trống. Hãy tải tài liệu hoặc thêm từ để bắt đầu học!');
    return;
  }
  const cardInner = document.getElementById('flashcard-inner');
  if (!cardInner) return;
  state.isCardFlipped = !state.isCardFlipped;
  cardInner.classList.toggle('is-flipped', state.isCardFlipped);
}

function nextCard() {
  if (!state.words || state.words.length === 0) return;
  state.currentCardIndex = (state.currentCardIndex + 1) % state.words.length;
  renderFlashcard();
}

function prevCard() {
  if (!state.words || state.words.length === 0) return;
  state.currentCardIndex = (state.currentCardIndex - 1 + state.words.length) % state.words.length;
  renderFlashcard();
}

function shuffleWords() {
  if (!state.words || state.words.length < 2) return;
  for (let i = state.words.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [state.words[i], state.words[j]] = [state.words[j], state.words[i]];
  }
  state.currentCardIndex = 0;
  renderFlashcard();
  showToast('Đã xáo trộn danh sách từ vựng ngẫu nhiên');
}

function speakWord() {
  if (!state.words || state.words.length === 0) return;
  const w = state.words[state.currentCardIndex];
  if (w) speakText(w.word);
}

async function rateCard(quality) {
  if (state.isCardSubmitting) return;
  if (!state.words || state.words.length === 0) {
    showToast('Kho từ đang trống, không có từ để đánh giá.');
    return;
  }
  const current = state.words[state.currentCardIndex];
  if (!current) return;

  state.isCardSubmitting = true;
  try {
    const res = await fetch('/api/words/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word_id: current.id, quality, mode: 'flashcard' })
    });
    if (res.ok) {
      const resData = await res.json();
      if (resData && resData.word) {
        Object.assign(current, resData.word);
      }
      loadStats();
      showToast(`Đã ghi nhận đánh giá cho "${current.word}"`);
      nextCard();
    }
  } catch (err) {
    console.error('Error submitting card review:', err);
  } finally {
    state.isCardSubmitting = false;
  }
}

// ==========================================
// 2. SPACED REPETITION (SM-2) SESSION MODE
// ==========================================
let srsAbortCtrl = null;

async function initSrsSession() {
  if (typeof srsTypingTimeout !== 'undefined' && srsTypingTimeout) {
    clearTimeout(srsTypingTimeout);
    srsTypingTimeout = null;
  }
  if (srsAbortCtrl) {
    srsAbortCtrl.abort();
  }
  srsAbortCtrl = new AbortController();
  const signal = srsAbortCtrl.signal;

  try {
    const res = await fetch('/api/words/due', { signal });
    if (!res.ok) return;
    const data = await res.json();
    state.dueWords = data.words || [];
    state.srsIndex = 0;
    state.srsReviewedCount = 0;
    renderSrsCard();
  } catch (err) {
    if (err.name === 'AbortError') return;
    console.error('Error initializing SRS session:', err);
  }
}

function renderSrsCard() {
  if (typeof srsTypingTimeout !== 'undefined' && srsTypingTimeout) {
    clearTimeout(srsTypingTimeout);
    srsTypingTimeout = null;
  }
  const emptyState = document.getElementById('srs-empty-state');
  const content = document.getElementById('srs-content');
  const queueCounter = document.getElementById('srs-queue-counter');
  const iconBox = document.getElementById('srs-empty-icon-box');
  const title = document.getElementById('srs-empty-title');
  const desc = document.getElementById('srs-empty-desc');
  const btn = document.getElementById('srs-empty-btn');

  // Trường hợp 1: Kho từ vựng hoàn toàn trống (0 từ)
  if (!state.words || state.words.length === 0) {
    emptyState.classList.remove('hidden');
    content.classList.add('hidden');
    queueCounter.textContent = '0 từ trong kho';
    if (iconBox) {
      iconBox.className = 'w-16 h-16 bg-indigo-50 text-indigo-600 rounded-full flex items-center justify-center mx-auto';
      iconBox.innerHTML = '<i data-lucide="folder-open" class="w-8 h-8"></i>';
    }
    if (title) title.textContent = 'Kho từ vựng hiện đang trống';
    if (desc) desc.textContent = 'Bạn chưa có từ vựng nào trong kho. Hãy bấm nút bên dưới để tải lên tài liệu PDF/Word nhé!';
    if (btn) {
      btn.textContent = 'Tải tài liệu lên ngay';
      btn.onclick = () => openUploadModal();
    }
    lucide.createIcons();
    return;
  }

  // Trường hợp 2: Đã có từ nhưng không còn từ nào đến hạn ôn hôm nay
  if (state.dueWords.length === 0 || state.srsIndex >= state.dueWords.length) {
    emptyState.classList.remove('hidden');
    content.classList.add('hidden');
    queueCounter.textContent = '0 từ đến hạn';
    if (iconBox) {
      iconBox.className = 'w-16 h-16 bg-emerald-50 text-emerald-600 rounded-full flex items-center justify-center mx-auto';
      iconBox.innerHTML = '<i data-lucide="check-check" class="w-8 h-8"></i>';
    }
    if (title) title.textContent = 'Hoàn thành phiên ôn tập hôm nay!';
    if (desc) desc.textContent = 'Bạn đã ôn tập xong tất cả từ vựng cần ôn theo lịch SM-2 hôm nay. Tuyệt vời!';
    if (btn) {
      btn.textContent = 'Tiếp tục học với Flashcards';
      btn.onclick = () => switchTab('flashcards');
    }
    lucide.createIcons();
    if (state.srsReviewedCount > 0) {
      triggerConfetti();
      state.srsReviewedCount = 0;
    }
    return;
  }

  emptyState.classList.add('hidden');
  content.classList.remove('hidden');
  queueCounter.textContent = `Còn ${state.dueWords.length - state.srsIndex} từ cần ôn`;

  const w = state.dueWords[state.srsIndex];
  state.isSrsRevealed = false;

  // Display repetition count and review interval telemetry
  const repEl = document.getElementById('srs-rep-count');
  const intvEl = document.getElementById('srs-cur-interval');
  const efEl = document.getElementById('srs-cur-ef');
  if (repEl) repEl.textContent = w.repetitions || 0;
  if (intvEl) intvEl.textContent = w.interval || 1;
  if (efEl) efEl.textContent = (w.ease_factor || 2.5).toFixed(2);

  // Dynamic calculated interval previews based on SM-2
  const reps = Math.max(0, parseInt(w.repetitions) || 0);
  const ef = parseFloat(w.ease_factor) || 2.5;
  const curInt = Math.max(1, parseInt(w.interval) || 1);

  function calcDynamicInterval(quality) {
    if (quality < 3) return quality === 1 ? 'Hôm nay' : '+1 ngày';
    let nextInt = 1;
    if (reps === 0) nextInt = 1;
    else if (reps === 1) nextInt = 6;
    else nextInt = Math.min(365, Math.max(1, Math.round(curInt * ef)));
    return `+${nextInt} ngày`;
  }

  const int1 = document.getElementById('srs-interval-1');
  const int2 = document.getElementById('srs-interval-2');
  const int3 = document.getElementById('srs-interval-3');
  const int4 = document.getElementById('srs-interval-4');
  const int5 = document.getElementById('srs-interval-5');
  if (int1) int1.textContent = 'Hôm nay';
  if (int2) int2.textContent = '+1 ngày';
  if (int3) int3.textContent = calcDynamicInterval(3);
  if (int4) int4.textContent = calcDynamicInterval(4);
  if (int5) int5.textContent = calcDynamicInterval(5);

  const mode = (typeof getSrsMode === 'function') ? getSrsMode() : 'flashcard';
  const wordEl = document.getElementById('srs-word');
  const ipaEl = document.getElementById('srs-ipa');
  const meaningEl = document.getElementById('srs-meaning');
  const exampleEl = document.getElementById('srs-example');
  
  const typingZone = document.getElementById('srs-typing-zone');
  const typingInput = document.getElementById('srs-typing-input');
  const answerBox = document.getElementById('srs-answer-box');
  const ratingBox = document.getElementById('srs-rating-buttons');
  const showBtnWrapper = document.getElementById('srs-show-answer-btn-wrapper');
  const clozeHint = document.getElementById('srs-cloze-hint');
  
  // Reset display states
  if (typingZone) typingZone.classList.add('hidden');
  if (answerBox) answerBox.classList.add('hidden');
  if (ratingBox) ratingBox.classList.add('hidden');
  if (showBtnWrapper) showBtnWrapper.classList.add('hidden');
  if (clozeHint) clozeHint.classList.add('hidden');
  if (typingInput) {
    typingInput.value = '';
    typingInput.classList.remove('border-rose-500', 'bg-rose-50', 'border-emerald-500', 'bg-emerald-50');
  }
  
  let srsSentence = '';
  if (mode === 'flashcard') {
    wordEl.textContent = w.word;
    ipaEl.textContent = w.ipa || `/${w.word}/`;
    meaningEl.textContent = cleanVietnamese(w.meaning_vi);
    srsSentence = formatCleanSentence(w.example_sentence, w.word);
    if(exampleEl) exampleEl.textContent = srsSentence ? `"${srsSentence}"` : '';
    showBtnWrapper.classList.remove('hidden');
  } else if (mode === 'cloze') {
    wordEl.textContent = '...';
    ipaEl.textContent = '';
    meaningEl.textContent = cleanVietnamese(w.meaning_vi);
    
    srsSentence = w.example_sentence ? w.example_sentence.replace(/\|/g, ' - ') : '';
    if (srsSentence && srsSentence !== 'Trong bài đọc') {
        const regex = new RegExp(`\\b${escapeRegex(w.word)}\\b`, 'i');
        srsSentence = srsSentence.replace(regex, '[...]');
    } else {
        srsSentence = 'Không có câu ví dụ cho từ này.';
    }
    if(exampleEl) exampleEl.textContent = srsSentence ? `"${srsSentence}"` : '';
    
    typingZone.classList.remove('hidden');
    clozeHint.textContent = 'Điền từ còn thiếu vào chỗ trống:';
    clozeHint.classList.remove('hidden');
    setTimeout(() => typingInput && typingInput.focus(), 100);
  } else if (mode === 'spelling') {
    wordEl.textContent = '?';
    ipaEl.textContent = 'Nghe và gõ lại từ';
    meaningEl.textContent = cleanVietnamese(w.meaning_vi);
    if(exampleEl) exampleEl.classList.add('hidden');
    
    typingZone.classList.remove('hidden');
    clozeHint.textContent = 'Lắng nghe phát âm và gõ từ:';
    clozeHint.classList.remove('hidden');
    
    speakText(w.word);
    setTimeout(() => typingInput && typingInput.focus(), 100);
  }
}

function revealSrsAnswer() {
  state.isSrsRevealed = true;
  const answerBox = document.getElementById('srs-answer-box');
  const ratingBox = document.getElementById('srs-rating-buttons');
  const showBtnWrapper = document.getElementById('srs-show-answer-btn-wrapper');
  if (answerBox) answerBox.classList.remove('hidden');
  if (ratingBox) ratingBox.classList.remove('hidden');
  if (showBtnWrapper) showBtnWrapper.classList.add('hidden');
  speakSrsWord();
}

function speakSrsWord() {
  const w = state.dueWords[state.srsIndex];
  if (w) speakText(w.word);
}

let isSrsSubmitting = false;

async function submitSrsReview(quality) {
  if (isSrsSubmitting) return;
  if (typeof srsTypingTimeout !== 'undefined' && srsTypingTimeout) {
    clearTimeout(srsTypingTimeout);
    srsTypingTimeout = null;
  }
  const w = (state.dueWords || [])[state.srsIndex];
  if (!w) return;

  isSrsSubmitting = true;
  try {
    const res = await fetch('/api/words/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word_id: w.id, quality, mode: 'srs' })
    });
    if (res.ok) {
      const resData = await res.json();
      if (resData && resData.word) {
        Object.assign(w, resData.word);
        const mainWord = (state.words || []).find(item => item.id === w.id);
        if (mainWord) {
          Object.assign(mainWord, resData.word);
        }
      }
      loadStats();
      state.srsReviewedCount = (state.srsReviewedCount || 0) + 1;
      state.srsIndex++;
      renderSrsCard();
    }
  } catch (err) {
    console.error('Error submitting SRS review:', err);
  } finally {
    isSrsSubmitting = false;
  }
}

// ==========================================
// 3. MULTIPLE CHOICE QUIZ MODE
// ==========================================
function initQuiz() {
  state.quizScore = 0;
  state.quizStreak = 0;
  document.getElementById('quiz-score').textContent = 'Điểm: 0';
  document.getElementById('quiz-streak').textContent = 'Chuỗi: 0 🔥';
  nextQuizQuestion();
}

function nextQuizQuestion() {
  const targetWordEl = document.getElementById('quiz-target-word');
  const contextSentenceEl = document.getElementById('quiz-context-sentence');
  const optionsContainer = document.getElementById('quiz-options');
  const feedback = document.getElementById('quiz-feedback');
  const nextBtn = document.getElementById('quiz-next-btn');

  if (feedback) feedback.classList.add('hidden');
  if (nextBtn) nextBtn.classList.add('hidden');

  if (!state.words || state.words.length < 4) {
    if (targetWordEl) targetWordEl.textContent = 'Chưa đủ từ vựng';
    if (contextSentenceEl) contextSentenceEl.textContent = 'Kho từ cần ít nhất 4 từ vựng để tạo bài trắc nghiệm 4 đáp án.';
    if (optionsContainer) {
      optionsContainer.innerHTML = `
        <div class="col-span-full py-10 text-center space-y-3">
          <p class="text-sm text-slate-400">Bạn có thể nạp ngay bộ từ có sẵn từ Thư viện để bắt đầu chơi.</p>
          <button onclick="openPremadeDecksModal()" class="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-2xl shadow-sm transition cursor-pointer">
            Mở thư viện bộ từ mẫu
          </button>
        </div>
      `;
    }
    return;
  }

  // Track session question history so words aren't repeated prematurely
  state.quizHistory = state.quizHistory || [];
  let availableWords = state.words.filter(w => !state.quizHistory.includes(w.id));
  if (availableWords.length === 0) {
    state.quizHistory = [];
    availableWords = state.words;
  }
  const target = availableWords[Math.floor(Math.random() * availableWords.length)];
  state.quizHistory.push(target.id);
  if (state.quizHistory.length > Math.min(10, state.words.length - 1)) {
    state.quizHistory.shift();
  }
  state.currentQuizWord = target;

  if (targetWordEl) targetWordEl.textContent = target.word;
  const cleanEnglishSentence = formatCleanSentence(target.example_sentence, target.word);
  if (contextSentenceEl) {
    if (cleanEnglishSentence) {
      contextSentenceEl.innerHTML = `&ldquo;${highlightWordInSentence(cleanEnglishSentence, target.word)}&rdquo;`;
    } else {
      contextSentenceEl.textContent = '';
    }
  }

  // Chuẩn hóa đáp án đúng
  const formattedCorrect = formatQuizChoice(target.meaning_vi);
  let rawChoices = [];
  if (target.distractors && target.distractors.length >= 3) {
    rawChoices = target.distractors.slice(0, 3);
  } else {
    // Lấy nghĩa ngẫu nhiên từ các từ vựng khác
    const others = state.words.filter(w => w.id !== target.id);
    const shuffledOthers = others.sort(() => 0.5 - Math.random());
    for (let i = 0; i < Math.min(6, shuffledOthers.length); i++) {
      rawChoices.push(shuffledOthers[i].meaning_vi);
    }
  }

  // Chuẩn hóa và lọc trùng 4 lựa chọn
  const choiceSet = new Set([formattedCorrect]);
  const formattedDistractors = [];

  for (const raw of rawChoices) {
    const formatted = formatQuizChoice(raw);
    if (formatted && !choiceSet.has(formatted) && formattedDistractors.length < 3) {
      choiceSet.add(formatted);
      formattedDistractors.push(formatted);
    }
  }

  // Bổ sung thêm nếu chưa đủ 3 phương án gây nhiễu
  if (formattedDistractors.length < 3) {
    const remainingOthers = state.words.filter(w => w.id !== target.id).sort(() => 0.5 - Math.random());
    for (const other of remainingOthers) {
      const fmtChoice = formatQuizChoice(other.meaning_vi);
      if (fmtChoice && !choiceSet.has(fmtChoice)) {
        choiceSet.add(fmtChoice);
        formattedDistractors.push(fmtChoice);
        if (formattedDistractors.length >= 3) break;
      }
    }
  }

  // Bổ sung từ tách nghĩa phụ của các từ khác nếu vẫn chưa đủ
  if (formattedDistractors.length < 3) {
    for (const other of state.words) {
      if (other.id === target.id) continue;
      const parts = (other.meaning_vi || '').split(/[,;\/\n]/);
      for (const part of parts) {
        const fmt = formatQuizChoice(part);
        if (fmt && fmt.length >= 3 && !choiceSet.has(fmt)) {
          choiceSet.add(fmt);
          formattedDistractors.push(fmt);
          if (formattedDistractors.length >= 3) break;
        }
      }
      if (formattedDistractors.length >= 3) break;
    }
  }

  const richDistractorFallbacks = [
    'Đạt được thành tựu xuất sắc qua nỗ lực bền bỉ',
    'Khả năng thích ứng linh hoạt với hoàn cảnh mới',
    'Tập trung nghiên cứu chuyên sâu về một lĩnh vực',
    'Duy trì sự cân bằng và phát triển ổn định lâu dài',
    'Tạo dựng mối quan hệ tin cậy và hợp tác bền vững',
    'Khám phá quy luật tự nhiên và phân tích logic'
  ];
  for (const fb of richDistractorFallbacks) {
    if (formattedDistractors.length < 3 && !choiceSet.has(fb)) {
      choiceSet.add(fb);
      formattedDistractors.push(fb);
    }
  }

  let finalChoices = [formattedCorrect, ...formattedDistractors];
  // Shuffle choices
  finalChoices = finalChoices.sort(() => 0.5 - Math.random());

  if (optionsContainer) {
    optionsContainer.innerHTML = '';
  }

  finalChoices.forEach((choice, idx) => {
    const btn = document.createElement('button');
    btn.className = 'quiz-opt-btn w-full p-4 text-left border-2 border-slate-200 dark:border-zinc-800 hover:border-indigo-500 dark:hover:border-indigo-500 rounded-2xl font-medium text-slate-800 dark:text-zinc-100 transition flex items-center justify-between group bg-white dark:bg-zinc-900 cursor-pointer';
    btn.innerHTML = `
      <span class="quiz-opt-text">${escapeHtml(choice)}</span>
      <span class="w-6 h-6 rounded-full border border-slate-300 dark:border-zinc-700 flex items-center justify-center text-xs text-slate-400 group-hover:border-indigo-500 group-hover:text-indigo-600 font-bold">${String.fromCharCode(65 + idx)}</span>
    `;
    btn.onclick = () => selectQuizAnswer(btn, choice, formattedCorrect);
    optionsContainer.appendChild(btn);
  });
}

function selectQuizAnswer(selectedBtn, chosenMeaning, correctMeaning) {
  // Disable all options
  document.querySelectorAll('.quiz-opt-btn').forEach(btn => {
    btn.disabled = true;
    const txtSpan = btn.querySelector('.quiz-opt-text');
    const txt = txtSpan ? txtSpan.textContent.trim() : btn.textContent.trim();
    if (txt === correctMeaning) {
      btn.classList.remove('border-slate-200', 'dark:border-zinc-800');
      btn.classList.add('border-emerald-500', 'bg-emerald-50', 'dark:bg-emerald-950/50', 'text-emerald-800', 'dark:text-emerald-200', 'font-bold');
    }
  });

  const feedback = document.getElementById('quiz-feedback');
  feedback.classList.remove('hidden');

  const target = state.currentQuizWord;
  const cleanExp = target ? formatCleanSentence(target.example_sentence, target.word) : '';
  const sentenceExplain = (target && cleanExp) ? `<div class="mt-2 text-xs font-normal opacity-90 p-2.5 rounded-xl bg-white/70 dark:bg-zinc-800/80 border border-slate-200 dark:border-zinc-700 text-slate-700 dark:text-zinc-200">📖 <strong>Câu ví dụ:</strong> ${escapeHtml(cleanExp)}</div>` : '';

  if (chosenMeaning === correctMeaning) {
    selectedBtn.classList.add('border-emerald-500', 'bg-emerald-50', 'dark:bg-emerald-950/50', 'text-emerald-800', 'dark:text-emerald-200');
    state.quizScore += 10;
    state.quizStreak += 1;
    feedback.className = 'p-4 rounded-xl text-sm font-semibold bg-emerald-50 dark:bg-emerald-950/50 border border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-200';
    feedback.innerHTML = `<div>🎉 Chính xác! (+10 điểm) Chuỗi đúng: ${state.quizStreak} câu liên tiếp 🔥</div>${sentenceExplain}`;
    
    // Update SRS rating in background (Good)
    if (state.currentQuizWord) {
      fetch('/api/words/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word_id: state.currentQuizWord.id, quality: 4, mode: 'quiz' })
      }).catch(err => console.warn('Background quiz review update failed:', err));
    }
  } else {
    selectedBtn.classList.add('border-rose-500', 'bg-rose-50', 'dark:bg-rose-950/50', 'text-rose-800', 'dark:text-rose-200');
    state.quizStreak = 0;
    feedback.className = 'p-4 rounded-xl text-sm font-semibold bg-rose-50 dark:bg-rose-950/50 border border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-200';
    feedback.innerHTML = `<div>❌ Chưa chính xác! Đáp án đúng: <strong>"${escapeHtml(correctMeaning)}"</strong></div>${sentenceExplain}`;
  }

  document.getElementById('quiz-score').textContent = `Điểm: ${state.quizScore}`;
  document.getElementById('quiz-streak').textContent = `Chuỗi: ${state.quizStreak} 🔥`;
  document.getElementById('quiz-next-btn').classList.remove('hidden');
}

function speakQuizWord() {
  if (state.currentQuizWord) speakText(state.currentQuizWord.word);
}

// ==========================================
// 4. FILL IN THE BLANKS MODE
// ==========================================
function initFillBlank() {
  state.fillIndex = 0;
  nextFillQuestion();
}

function nextFillQuestion() {
  if (!state.words || state.words.length === 0) {
    const meaningEl = document.getElementById('fill-meaning');
    const sentenceEl = document.getElementById('fill-sentence');
    const inputEl = document.getElementById('fill-input');
    const hintEl = document.getElementById('fill-hint-text');
    const resultBox = document.getElementById('fill-result-box');
    const nextBtn = document.getElementById('fill-next-btn');
    if (meaningEl) meaningEl.textContent = 'Kho từ đang trống';
    if (sentenceEl) sentenceEl.innerHTML = '<span class="text-slate-400 italic">Vui lòng tải lên tài liệu hoặc thêm từ vựng để bắt đầu luyện tập điền từ.</span>';
    if (inputEl) { inputEl.value = ''; inputEl.disabled = true; }
    if (hintEl) hintEl.textContent = '';
    if (resultBox) resultBox.classList.add('hidden');
    if (nextBtn) nextBtn.classList.add('hidden');
    return;
  }
  const filtered = state.words.filter(w => {
    const s = formatCleanSentence(w.example_sentence, w.word);
    return s && s.toLowerCase().includes(w.word.toLowerCase());
  });
  const pool = filtered.length > 0 ? filtered : state.words;
  
  const target = pool[Math.floor(Math.random() * pool.length)];
  state.currentFillWord = target;

  document.getElementById('fill-meaning').textContent = cleanVietnamese(target.meaning_vi);
  document.getElementById('fill-input').value = '';
  document.getElementById('fill-input').disabled = false;
  document.getElementById('fill-input').focus();
  document.getElementById('fill-hint-text').textContent = '';
  document.getElementById('fill-result-box').classList.add('hidden');
  document.getElementById('fill-next-btn').classList.add('hidden');

  // Mask the word in sentence safely
  let rawSentence = formatCleanSentence(target.example_sentence, target.word);
  if (!rawSentence || rawSentence.trim() === '') {
    rawSentence = `In everyday communication, we use "${target.word}" to mean: ${cleanVietnamese(target.meaning_vi)}.`;
  }

  // Update dynamic question counter
  state.fillIndex = (state.fillIndex || 0) + 1;
  const fillCounterEl = document.getElementById('fill-counter');
  if (fillCounterEl) {
    fillCounterEl.textContent = `Câu ${state.fillIndex}/${pool.length}`;
  }

  let safeSentence = escapeHtml(rawSentence);
  const safeWord = escapeHtml(target.word);
  const escapedWord = safeWord.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  // Extract contextual inflected form from sentence
  let matchedForm = null;
  const rawMatch = rawSentence.match(new RegExp(`\\b${escapedWord}\\w*\\b`, 'i'));
  if (rawMatch) {
    matchedForm = rawMatch[0].toLowerCase();
  }
  state.currentFillMatchedForm = matchedForm;

  let regex = new RegExp(`\\b${escapedWord}\\w*\\b`, 'gi');
  let maskedSentence = safeSentence.replace(regex, '<span class="bg-indigo-100 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300 font-bold px-2 py-0.5 rounded">______</span>');
  if (maskedSentence === safeSentence) {
    maskedSentence = safeSentence.replace(new RegExp(escapedWord, 'gi'), '<span class="bg-indigo-100 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300 font-bold px-2 py-0.5 rounded">______</span>');
  }
  document.getElementById('fill-sentence').innerHTML = `"${maskedSentence}"`;
}

function showFillHint() {
  if (!state.currentFillWord) return;
  const w = state.currentFillWord.word;
  const hint = `Bắt đầu bằng: "${w.slice(0, 2).toUpperCase()}..." (${w.length} ký tự, loại từ: ${state.currentFillWord.pos || 'word'})`;
  document.getElementById('fill-hint-text').textContent = hint;
}

function checkFillAnswer() {
  if (!state.currentFillWord) return;
  const input = document.getElementById('fill-input').value.trim().toLowerCase();
  const target = state.currentFillWord.word.trim().toLowerCase();
  const matchedForm = (state.currentFillMatchedForm || '').trim().toLowerCase();
  const box = document.getElementById('fill-result-box');
  box.classList.remove('hidden');

  // Fair answer evaluation: support base form, sentence matched inflection, or regular inflections
  const isMatch = (input === target) ||
    (matchedForm && input === matchedForm) ||
    (input + 'e' === target) || (input === target + 'e') ||
    (input === target + 's') || (input === target + 'es') ||
    (input === target + 'ed') || (input === target + 'd') ||
    (input === target + 'ing') ||
    (target.endsWith('e') && input === target.slice(0, -1) + 'ing') ||
    (target.endsWith('e') && input === target.slice(0, -1) + 'ed') ||
    (target.endsWith('y') && input === target.slice(0, -1) + 'ies') ||
    (target.endsWith('y') && input === target.slice(0, -1) + 'ied');

  if (isMatch) {
    box.className = 'p-4 rounded-xl text-sm font-semibold bg-emerald-50 dark:bg-emerald-950/50 border border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-200';
    const displayedWord = (input === target) ? target : `${input} (${target})`;
    box.textContent = `🎉 Hoàn hảo! Bạn đã điền chính xác từ "${displayedWord}".`;
    speakText(target);
    document.getElementById('fill-input').disabled = true;
    const nextBtn = document.getElementById('fill-next-btn');
    if (nextBtn) {
      nextBtn.classList.remove('hidden');
      nextBtn.focus();
    }

    // Background SRS update
    fetch('/api/words/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word_id: state.currentFillWord.id, quality: 5, mode: 'fill' })
    }).catch(err => console.warn('Background fill review update failed:', err));
  } else {
    box.className = 'p-4 rounded-xl text-sm font-semibold bg-rose-50 dark:bg-rose-950/50 border border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-200';
    box.textContent = `Chưa đúng! Thử lại hoặc nhấn "Gợi ý chữ cái đầu".`;
  }
}

function clearMatchTimeouts() {
  if (state.matchTimeouts && state.matchTimeouts.length > 0) {
    state.matchTimeouts.forEach(t => clearTimeout(t));
    state.matchTimeouts = [];
  }
}

// ==========================================
// 5. WORD MATCHING GAME MODE
// ==========================================
function startMatchGame() {
  const grid = document.getElementById('match-grid');

  if (!state.words || state.words.length < 4) {
    if (grid) {
      grid.innerHTML = `
        <div class="col-span-full py-16 text-center space-y-3">
          <p class="text-sm font-semibold text-slate-400">Trò chơi cần ít nhất 4 từ vựng trong kho để ghép cặp thẻ.</p>
          <button onclick="openPremadeDecksModal()" class="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl shadow-sm transition cursor-pointer">
            Nạp bộ từ mẫu từ Thư viện
          </button>
        </div>
      `;
    }
    showToast('Cần ít nhất 4 từ vựng để chơi trò chơi ghép thẻ.');
    return;
  }

  // Stop previous timer & timeouts
  if (state.matchTimerInterval) clearInterval(state.matchTimerInterval);
  clearMatchTimeouts();
  state.matchSeconds = 0;
  state.matchErrors = 0;
  state.matchStreak = 0;
  state.matchMaxStreak = 0;
  state.matchLiveScore = 0;
  document.getElementById('match-timer').textContent = '00:00';
  const liveScoreEl = document.getElementById('match-live-score');
  if (liveScoreEl) liveScoreEl.textContent = '0';
  document.getElementById('match-win-banner').classList.add('hidden');

  // Select up to 6 words with distinct definitions to eliminate card matching collisions
  const sampleCount = Math.min(6, state.words.length);
  const shuffledCandidates = [...state.words].sort(() => 0.5 - Math.random());
  const selectedWords = [];
  const usedMeanings = new Set();

  for (const w of shuffledCandidates) {
    const fmt = formatQuizChoice(w.meaning_vi);
    if (fmt && !usedMeanings.has(fmt)) {
      usedMeanings.add(fmt);
      selectedWords.push(w);
      if (selectedWords.length >= sampleCount) break;
    }
  }
  if (selectedWords.length < sampleCount) {
    for (const w of shuffledCandidates) {
      if (!selectedWords.includes(w)) {
        selectedWords.push(w);
        if (selectedWords.length >= sampleCount) break;
      }
    }
  }
  state.currentMatchWords = selectedWords;

  // Generate 12 cards (6 English, 6 Vietnamese)
  let cards = [];
  selectedWords.forEach(w => {
    const fmtDef = formatQuizChoice(w.meaning_vi);
    cards.push({ id: w.id, text: w.word, type: 'en', definition: fmtDef });
    cards.push({ id: w.id, text: fmtDef, type: 'vi', definition: fmtDef });
  });

  // Shuffle cards
  cards = cards.sort(() => 0.5 - Math.random());
  state.matchCards = cards;
  state.selectedMatchCards = [];
  state.matchedPairsCount = 0;

  if (grid) grid.innerHTML = '';

  cards.forEach((card, idx) => {
    const el = document.createElement('div');
    el.className = 'match-card p-3 sm:p-5 rounded-2xl border-2 border-slate-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-900 flex items-center justify-center text-center font-bold text-xs sm:text-sm md:text-base text-slate-800 dark:text-zinc-100 min-h-[90px] sm:min-h-[110px] shadow-sm cursor-pointer select-none transition-all duration-200 break-words';
    el.textContent = card.text;
    el.onclick = () => handleMatchCardClick(el, card);
    grid.appendChild(el);
  });

  // Start timer
  state.matchTimerInterval = setInterval(() => {
    state.matchSeconds++;
    const mins = String(Math.floor(state.matchSeconds / 60)).padStart(2, '0');
    const secs = String(state.matchSeconds % 60).padStart(2, '0');
    document.getElementById('match-timer').textContent = `${mins}:${secs}`;
  }, 1000);
}

function handleMatchCardClick(el, card) {
  if (el.classList.contains('matched') || el.classList.contains('selected')) return;
  if (state.selectedMatchCards.length >= 2) return;

  el.classList.add('selected');
  state.selectedMatchCards.push({ el, card });

  if (state.selectedMatchCards.length === 2) {
    const [first, second] = state.selectedMatchCards;
    
    // Check match condition: same word id or matching definition across en & vi
    const isMatch = (first.card.id === second.card.id && first.card.type !== second.card.type) ||
      (first.card.type !== second.card.type && first.card.definition && first.card.definition === second.card.definition);

    if (isMatch) {
      state.matchStreak = (state.matchStreak || 0) + 1;
      state.matchMaxStreak = Math.max(state.matchMaxStreak || 0, state.matchStreak);
      const currentScore = Math.max(100, Math.round(1000 + Math.max(0, 500 - state.matchSeconds * 10) + ((state.matchMaxStreak || 0) * 50) - ((state.matchErrors || 0) * 40)));
      state.matchLiveScore = currentScore;
      const liveScoreEl = document.getElementById('match-live-score');
      if (liveScoreEl) liveScoreEl.textContent = currentScore;

      const t = setTimeout(() => {
        first.el.classList.remove('selected');
        second.el.classList.remove('selected');
        first.el.classList.add('matched');
        second.el.classList.add('matched');
        state.selectedMatchCards = [];
        state.matchedPairsCount++;

        // Pronounce if it's english
        const enCard = first.card.type === 'en' ? first.card : second.card;
        speakText(enCard.text);

        if (state.matchedPairsCount === state.matchCards.length / 2) {
          // Game won!
          clearInterval(state.matchTimerInterval);
          state.matchTimerInterval = null;
          const finalScore = Math.max(100, Math.round(1000 + Math.max(0, 500 - state.matchSeconds * 10) + ((state.matchMaxStreak || 0) * 50) - ((state.matchErrors || 0) * 40)));
          const banner = document.getElementById('match-win-banner');
          if (banner) banner.classList.remove('hidden');
          const finalScoreEl = document.getElementById('match-final-score');
          if (finalScoreEl) finalScoreEl.textContent = `Điểm số: ${finalScore} điểm ⭐ (Chuỗi tối đa: ${state.matchMaxStreak || 0} 🔥, Lỗi: ${state.matchErrors || 0})`;
          const finalTimeEl = document.getElementById('match-final-time');
          if (finalTimeEl) finalTimeEl.textContent = `Thời gian hoàn thành: ${state.matchSeconds} giây!`;
          triggerConfetti();

          // Background SRS update for all matched words
          if (state.currentMatchWords && Array.isArray(state.currentMatchWords)) {
            state.currentMatchWords.forEach(w => {
              if (w && w.id) {
                fetch('/api/words/review', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ word_id: w.id, quality: 4, mode: 'match' })
                }).catch(err => console.warn('Match game background review update failed:', err));
              }
            });
          }
        }
      }, 300);
      state.matchTimeouts.push(t);
    } else {
      // Wrong match
      state.matchStreak = 0;
      state.matchErrors = (state.matchErrors || 0) + 1;
      const currentScore = Math.max(100, Math.round(1000 + Math.max(0, 500 - state.matchSeconds * 10) + ((state.matchMaxStreak || 0) * 50) - ((state.matchErrors || 0) * 40)));
      state.matchLiveScore = currentScore;
      const liveScoreEl = document.getElementById('match-live-score');
      if (liveScoreEl) liveScoreEl.textContent = currentScore;

      const t1 = setTimeout(() => {
        first.el.classList.add('wrong');
        second.el.classList.add('wrong');
      }, 200);

      const t2 = setTimeout(() => {
        first.el.classList.remove('selected', 'wrong');
        second.el.classList.remove('selected', 'wrong');
        state.selectedMatchCards = [];
      }, 700);
      state.matchTimeouts.push(t1, t2);
    }
  }
}

// ==========================================
// 6. SMART THESAURUS & DICTIONARY MODE
// ==========================================
let thesaurusAbortCtrl = null;
let isThesaurusDelegated = false;

function setupThesaurusDelegation() {
  if (isThesaurusDelegated) return;
  isThesaurusDelegated = true;

  ['thesaurus-synonyms-grid', 'thesaurus-antonyms-grid'].forEach(containerId => {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.addEventListener('click', (e) => {
      const card = e.target.closest('.thesaurus-item-card');
      if (!card) return;

      const rawWord = card.dataset.word ? decodeURIComponent(card.dataset.word) : '';
      const rawMeaning = card.dataset.meaning ? decodeURIComponent(card.dataset.meaning) : '';
      if (!rawWord) return;

      const speakBtn = e.target.closest('.thesaurus-btn-speak');
      if (speakBtn) {
        speakText(rawWord);
        return;
      }

      const saveBtn = e.target.closest('.thesaurus-btn-save');
      if (saveBtn) {
        saveQuickThesaurusWord(saveBtn, rawWord, rawMeaning, card);
        return;
      }

      const trigger = e.target.closest('.thesaurus-lookup-trigger');
      if (trigger) {
        quickThesaurusLookup(rawWord);
        return;
      }
    });
  });
}

function initThesaurus() {
  setupThesaurusDelegation();
  const input = document.getElementById('thesaurus-input');
  if (input) {
    input.focus();
  }
}

function executeThesaurusSearch() {
  const input = document.getElementById('thesaurus-input');
  const word = (input?.value || '').trim();
  if (!word) {
    showToast('Vui lòng nhập từ tiếng Anh cần tra cứu.');
    if (input) input.focus();
    return;
  }
  searchThesaurus(word);
}

function quickThesaurusLookup(word, context) {
  const input = document.getElementById('thesaurus-input');
  if (input) {
    input.value = word;
    toggleThesaurusClearBtn(word);
  }
  searchThesaurus(word, context);
}

function toggleThesaurusClearBtn(val) {
  const clearBtn = document.getElementById('thesaurus-btn-clear');
  if (!clearBtn) return;
  if (val && val.trim().length > 0) {
    clearBtn.classList.remove('hidden');
  } else {
    clearBtn.classList.add('hidden');
  }
}

function clearThesaurusInput() {
  const input = document.getElementById('thesaurus-input');
  if (input) {
    input.value = '';
    input.focus();
  }
  toggleThesaurusClearBtn('');
}

function openThesaurusFromReader() {
  const word = state.activeLookupWord;
  if (!word) {
    showToast('Chưa chọn từ vựng nào trong bài đọc.');
    return;
  }
  // Truyền ngữ cảnh câu từ bài đọc sang Thesaurus để tra cứu chính xác từ đa nghĩa
  const context = state.activeLookupContext || '';
  switchTab('thesaurus');
  quickThesaurusLookup(word, context);
}

function speakThesaurusHeadword() {
  if (state.thesaurusData && state.thesaurusData.word) {
    speakText(state.thesaurusData.word);
  }
}

// BỘ TĂNG TỐC: Tách riêng logic render để dùng chung cho cache hit và API response
function renderThesaurusResult(data) {
  const emptyEl = document.getElementById('thesaurus-empty-state');
  const loadingEl = document.getElementById('thesaurus-loading-state');
  const resultEl = document.getElementById('thesaurus-result-container');

  state.thesaurusData = data;

  // Render Headword Overview
  document.getElementById('thesaurus-headword').textContent = data.word;
  document.getElementById('thesaurus-ipa').textContent = data.ipa || `/${data.word}/`;
  document.getElementById('thesaurus-pos').textContent = data.pos || 'vocabulary';
  const headwordMeaning = (data.meaning_vi && data.meaning_vi.toLowerCase().trim() !== (data.word || '').toLowerCase().trim())
    ? cleanVietnamese(data.meaning_vi)
    : 'Đang cập nhật';
  document.getElementById('thesaurus-meaning-vi').textContent = headwordMeaning;

  // Example Box
  const exEn = document.getElementById('thesaurus-example-en');
  const exVi = document.getElementById('thesaurus-example-vi');
  if (data.example_sentence) {
    exEn.innerHTML = `&ldquo;${highlightWordInSentence(data.example_sentence, data.word)}&rdquo;`;
    exVi.textContent = data.example_vi ? `Dịch: "${data.example_vi}"` : '';
    document.getElementById('thesaurus-example-box').classList.remove('hidden');
  } else {
    document.getElementById('thesaurus-example-box').classList.add('hidden');
  }

  // Saved Status Badge & Add button
  updateThesaurusSavedState(data.is_saved);

  // Cập nhật CEFR level badge
  const levelEl = document.getElementById('thesaurus-level');
  if (levelEl) {
    levelEl.textContent = data.cefr_level || '';
    levelEl.classList.toggle('hidden', !data.cefr_level);
  }

  // Render Synonyms
  renderThesaurusList('thesaurus-synonyms-grid', 'thesaurus-syn-count', data.synonyms || [], 'synonym');

  // Render Antonyms
  renderThesaurusList('thesaurus-antonyms-grid', 'thesaurus-ant-count', data.antonyms || [], 'antonym');

  if (loadingEl) loadingEl.classList.add('hidden');
  if (emptyEl) emptyEl.classList.add('hidden');
  if (resultEl) resultEl.classList.remove('hidden');
  lucide.createIcons();
}

function renderThesaurusInitialFast(data) {
  const emptyEl = document.getElementById('thesaurus-empty-state');
  const loadingEl = document.getElementById('thesaurus-loading-state');
  const resultEl = document.getElementById('thesaurus-result-container');

  if (emptyEl) emptyEl.classList.add('hidden');
  if (loadingEl) loadingEl.classList.add('hidden');
  if (resultEl) resultEl.classList.remove('hidden');

  // Render ngay lập tức dữ liệu từ vựng ngoại tuyến O(1)
  document.getElementById('thesaurus-headword').textContent = data.word;
  document.getElementById('thesaurus-ipa').textContent = data.ipa || `/${data.word}/`;
  document.getElementById('thesaurus-pos').textContent = data.pos || 'vocabulary';
  const meaning = cleanVietnamese(data.meaning_vi || data.word);
  document.getElementById('thesaurus-meaning-vi').textContent = meaning;

  const levelEl = document.getElementById('thesaurus-level');
  if (levelEl) {
    levelEl.textContent = data.cefr_level || 'B1';
    levelEl.classList.remove('hidden');
  }

  // Cập nhật trạng thái đã lưu
  const isSaved = (state.words || []).some(w => w.word.toLowerCase() === data.word.toLowerCase());
  updateThesaurusSavedState(isSaved);

  // Hiển thị khung chờ nạp từ đồng nghĩa / trái nghĩa thanh lịch mà không che mất định nghĩa từ vựng
  const synGrid = document.getElementById('thesaurus-synonyms-grid');
  const antGrid = document.getElementById('thesaurus-antonyms-grid');
  const synCount = document.getElementById('thesaurus-syn-count');
  const antCount = document.getElementById('thesaurus-ant-count');
  if (synCount) synCount.textContent = '...';
  if (antCount) antCount.textContent = '...';

  if (synGrid) {
    synGrid.innerHTML = `
      <div class="col-span-full p-4 rounded-2xl bg-indigo-50/60 dark:bg-indigo-950/20 text-indigo-600 dark:text-indigo-400 text-xs font-semibold flex items-center space-x-2 animate-pulse">
        <i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i>
        <span>Đang nạp mạng lưới từ đồng nghĩa...</span>
      </div>`;
  }
  if (antGrid) {
    antGrid.innerHTML = `
      <div class="col-span-full p-4 rounded-2xl bg-slate-50 dark:bg-zinc-900/40 text-slate-500 dark:text-zinc-400 text-xs font-semibold flex items-center space-x-2 animate-pulse">
        <i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i>
        <span>Đang nạp từ trái nghĩa...</span>
      </div>`;
  }
  if (window.lucide) lucide.createIcons();
}

async function searchThesaurus(rawWord, rawContext) {
  const word = (rawWord || '').trim().toLowerCase();
  const context = (rawContext || '').trim();
  if (!word) return;

  state.activeThesaurusWord = word;

  // BỘ TĂNG TỐC 1: Kiểm tra cache client trước — hiển thị tức thời 0ms nếu đã tra
  const cacheKey = `${word}:${context}`;
  if (thesaurusCache.has(cacheKey)) {
    const cachedData = thesaurusCache.get(cacheKey);
    setupThesaurusDelegation();
    renderThesaurusResult(cachedData);
    return;
  }

  setupThesaurusDelegation();

  // BỘ TĂNG TỐC 2: Hiển thị TỨC THÌ (< 2ms) từ điển 108K ngoại tuyến!
  // Người dùng nhìn thấy ngay Nghĩa tiếng Việt, Phiên âm IPA, Loại từ mà không cần chờ mạng!
  const quickCached = translateCache.get(word);
  if (quickCached) {
    renderThesaurusInitialFast(quickCached);
  } else {
    fetch(`/api/translate?word=${encodeURIComponent(word)}`)
      .then(res => res.ok ? res.json() : null)
      .then(tData => {
        if (tData && state.activeThesaurusWord === word && !thesaurusCache.has(cacheKey)) {
          translateCache.set(word, tData);
          renderThesaurusInitialFast(tData);
        }
      })
      .catch(() => {});
  }

  // Hủy request trước nếu đang chạy dở để chống race condition
  if (thesaurusAbortCtrl) {
    thesaurusAbortCtrl.abort();
  }
  thesaurusAbortCtrl = new AbortController();
  const signal = thesaurusAbortCtrl.signal;

  try {
    let url = `/api/thesaurus?word=${encodeURIComponent(word)}`;
    if (context) {
      url += `&context=${encodeURIComponent(context)}`;
    }
    if (state.geminiModel) {
      url += `&gemini_model=${encodeURIComponent(state.geminiModel)}`;
    }

    const fetchHeaders = {};
    if (state.geminiApiKey) {
      fetchHeaders['X-Gemini-API-Key'] = state.geminiApiKey;
    }

    const res = await fetch(url, { signal, headers: fetchHeaders });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();

    // Lưu cache client (LRU - xóa entry cũ nhất khi đầy)
    if (thesaurusCache.size >= THESAURUS_CACHE_MAX) {
      const firstKey = thesaurusCache.keys().next().value;
      thesaurusCache.delete(firstKey);
    }
    thesaurusCache.set(cacheKey, data);

    if (state.activeThesaurusWord === word) {
      renderThesaurusResult(data);
    }

  } catch (err) {
    if (err.name === 'AbortError') {
      return;
    }
    console.error('Error searching thesaurus:', err);
    // Nếu đã có dữ liệu nhanh từ điển 108K thì không hiện màn hình lỗi toàn trang
    const resultEl = document.getElementById('thesaurus-result-container');
    if (resultEl && !resultEl.classList.contains('hidden')) {
      const synGrid = document.getElementById('thesaurus-synonyms-grid');
      if (synGrid) synGrid.innerHTML = `<p class="col-span-full text-xs text-slate-400 py-2">Không tìm thấy thêm từ đồng nghĩa.</p>`;
      const antGrid = document.getElementById('thesaurus-antonyms-grid');
      if (antGrid) antGrid.innerHTML = `<p class="col-span-full text-xs text-slate-400 py-2">Không tìm thấy từ trái nghĩa.</p>`;
      return;
    }
    showToast('Lỗi khi tra cứu từ vựng. Vui lòng thử lại.');
    const loadingEl = document.getElementById('thesaurus-loading-state');
    const emptyEl = document.getElementById('thesaurus-empty-state');
    if (loadingEl) loadingEl.classList.add('hidden');
    if (emptyEl) emptyEl.classList.add('hidden');
    if (resultEl) {
      resultEl.innerHTML = `
        <div class="p-8 text-center bg-rose-50/60 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-900/40 rounded-3xl space-y-3 max-w-lg mx-auto">
          <div class="w-12 h-12 rounded-2xl bg-rose-100 dark:bg-rose-900/40 text-rose-600 dark:text-rose-400 flex items-center justify-center mx-auto">
            <i data-lucide="alert-circle" class="w-6 h-6"></i>
          </div>
          <h4 class="font-bold text-slate-800 dark:text-zinc-100">Không thể tra cứu từ "${escapeHtml(word)}"</h4>
          <p class="text-xs text-slate-500 dark:text-zinc-400">Đã xảy ra lỗi kết nối với máy chủ hoặc dịch vụ từ điển.</p>
          <button onclick="searchThesaurus('${escapeHtml(word)}', '${escapeHtml(context)}')" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-xl shadow-sm transition cursor-pointer">Thử lại</button>
        </div>
      `;
      resultEl.classList.remove('hidden');
      if (window.lucide) lucide.createIcons();
    }
  }
}

function updateThesaurusSavedState(isSaved) {
  const badge = document.getElementById('thesaurus-saved-badge');
  const btn = document.getElementById('thesaurus-btn-add-deck');
  if (!badge || !btn) return;

  if (isSaved) {
    badge.className = 'px-3 py-1 text-xs font-bold rounded-full bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800 flex items-center space-x-1';
    badge.innerHTML = '<i data-lucide="check" class="w-3.5 h-3.5"></i><span>Đã lưu trong kho</span>';
    btn.disabled = true;
    btn.className = 'w-full py-3.5 px-5 bg-emerald-600/90 text-white font-bold text-sm rounded-2xl cursor-default flex items-center justify-center space-x-2';
    btn.innerHTML = '<i data-lucide="check" class="w-4 h-4"></i><span>Đã có trong Flashcards</span>';
  } else {
    badge.className = 'px-3 py-1 text-xs font-bold rounded-full bg-slate-100 dark:bg-white/10 text-slate-600 dark:text-slate-300 border border-transparent dark:border-white/10';
    badge.textContent = 'Chưa lưu';
    btn.disabled = false;
    btn.className = 'w-full py-3.5 px-5 bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-sm rounded-2xl transition shadow-md shadow-indigo-200 dark:shadow-none flex items-center justify-center space-x-2 cursor-pointer';
    btn.innerHTML = '<i data-lucide="bookmark-plus" class="w-4 h-4"></i><span>+ Thêm vào Flashcards</span>';
  }
}

function renderThesaurusList(containerId, countId, items, type) {
  const container = document.getElementById(containerId);
  const countEl = document.getElementById(countId);
  if (!container) return;

  if (countEl) countEl.textContent = `${items.length} từ`;

  if (!items || items.length === 0) {
    container.innerHTML = `
      <div class="col-span-full py-8 text-center text-xs text-slate-400 dark:text-zinc-500 italic">
        ${type === 'synonym' ? 'Chưa tìm thấy từ đồng nghĩa phù hợp.' : 'Không có từ trái nghĩa trực tiếp cho từ này.'}
      </div>
    `;
    return;
  }

  container.innerHTML = items.map(it => {
    const wordEncoded = encodeURIComponent(it.word || '');
    let meaningClean = cleanVietnamese(it.meaning_vi || '');
    if (!meaningClean || meaningClean.toLowerCase().trim() === (it.word || '').toLowerCase().trim()) {
      meaningClean = 'Đang cập nhật';
    }
    const meaningEncoded = encodeURIComponent(meaningClean);
    const isSaved = it.is_saved;

    return `
      <div class="thesaurus-item-card p-3 sm:p-3.5 rounded-2xl border border-slate-200/80 dark:border-zinc-800/80 bg-slate-50/80 dark:bg-zinc-900/70 hover:border-indigo-400/80 dark:hover:border-indigo-500/80 transition-all duration-150 group flex items-center justify-between space-x-2.5" data-word="${wordEncoded}" data-meaning="${meaningEncoded}">
        <div class="thesaurus-lookup-trigger flex-1 min-w-0 cursor-pointer" title="Nhấp để tra cứu tiếp từ '${escapeHtml(it.word)}'">
          <div class="flex items-center space-x-1.5">
            <span class="thesaurus-word-text font-bold text-sm sm:text-base text-slate-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition truncate tracking-tight">${escapeHtml(it.word)}</span>
            ${it.pos ? `<span class="text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-slate-100 dark:bg-zinc-700 text-slate-500 dark:text-zinc-400 ml-1.5">${escapeHtml(it.pos)}</span>` : ''}
            <span class="thesaurus-saved-indicator ${isSaved ? '' : 'hidden'} w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0" title="Đã có trong kho"></span>
          </div>
          <p class="thesaurus-meaning-text text-xs text-slate-600 dark:text-zinc-300 line-clamp-2 mt-0.5 leading-snug font-normal">${escapeHtml(meaningClean)}</p>
        </div>
        <div class="flex items-center space-x-1 flex-shrink-0">
          <button type="button" class="thesaurus-btn-speak p-1.5 text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-300 rounded-lg hover:bg-white dark:hover:bg-zinc-800 transition cursor-pointer" title="Phát âm">
            <i data-lucide="volume-2" class="w-3.5 h-3.5"></i>
          </button>
          <button type="button" class="thesaurus-btn-save p-1.5 ${isSaved ? 'text-emerald-500 cursor-default' : 'text-slate-400 hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer'} rounded-lg hover:bg-white dark:hover:bg-zinc-800 transition" title="${isSaved ? 'Đã lưu trong kho' : 'Thêm vào kho từ'}" ${isSaved ? 'disabled' : ''}>
            <i data-lucide="${isSaved ? 'check' : 'plus'}" class="w-3.5 h-3.5"></i>
          </button>
        </div>
      </div>
    `;
  }).join('');
}

async function saveCurrentThesaurusToDeck() {
  if (!state.thesaurusData) return;
  const d = state.thesaurusData;
  const btn = document.getElementById('thesaurus-btn-add-deck');
  if (btn && btn.disabled) return;

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Đang lưu...</span>';
    lucide.createIcons();
  }

  try {
    const res = await fetch('/api/words/manual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        word: d.word,
        meaning_vi: d.meaning_vi,
        ipa: d.ipa || `/${d.word}/`,
        pos: d.pos || 'vocabulary',
        example_sentence: d.example_sentence || `Example sentence for ${d.word}.`,
        cefr_level: d.cefr_level || 'B1'
      })
    });

    if (res.ok) {
      d.is_saved = true;
      updateThesaurusSavedState(true);
      await loadWords();
      await loadStats();
      if (typeof triggerConfetti === 'function') triggerConfetti();
      showToast(`Đã thêm "${d.word}" vào kho từ vựng và Flashcards!`);
      lucide.createIcons();
    } else {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<i data-lucide="bookmark-plus" class="w-4 h-4"></i><span>+ Thêm vào Flashcards</span>';
        lucide.createIcons();
      }
      showToast('Không thể lưu từ vựng');
    }
  } catch (err) {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<i data-lucide="bookmark-plus" class="w-4 h-4"></i><span>+ Thêm vào Flashcards</span>';
      lucide.createIcons();
    }
    console.error('Error saving thesaurus word:', err);
    showToast('Lỗi kết nối khi lưu từ vựng');
  }
}

async function saveQuickThesaurusWord(btnEl, word, meaningVi, cardEl) {
  if (!word || (btnEl && btnEl.disabled)) return;

  if (btnEl) {
    btnEl.disabled = true;
    btnEl.innerHTML = '<i data-lucide="loader-2" class="w-3.5 h-3.5 animate-spin"></i>';
    lucide.createIcons();
  }

  try {
    const res = await fetch('/api/words/manual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        word: word,
        meaning_vi: meaningVi || '',
        ipa: '',
        pos: '',
        example_sentence: '',
        cefr_level: ''
      })
    });

    if (res.ok) {
      const resData = await res.json();
      const savedWord = (resData && resData.word && resData.word.id) ? resData.word : {
        id: (resData && (resData.word_id || resData.id)) || Date.now(),
        word: word,
        meaning_vi: meaningVi || '',
        status: 'new'
      };
      if (btnEl) {
        btnEl.className = 'thesaurus-btn-save p-1.5 text-emerald-500 rounded-lg hover:bg-white dark:hover:bg-zinc-800 transition cursor-default';
        btnEl.title = 'Đã có trong kho';
        btnEl.innerHTML = '<i data-lucide="check" class="w-3.5 h-3.5"></i>';
        btnEl.disabled = true;
      }
      if (cardEl) {
        const dot = cardEl.querySelector('.thesaurus-saved-indicator');
        if (dot) dot.classList.remove('hidden');
      }
      // Đồng bộ vào state.thesaurusData
      if (state.thesaurusData) {
        const wl = word.toLowerCase();
        if (state.thesaurusData.synonyms) {
          state.thesaurusData.synonyms.forEach(s => {
            if (s.word.toLowerCase() === wl) s.is_saved = true;
          });
        }
        if (state.thesaurusData.antonyms) {
          state.thesaurusData.antonyms.forEach(a => {
            if (a.word.toLowerCase() === wl) a.is_saved = true;
          });
        }
      }
      lucide.createIcons();
      // Cập nhật state cục bộ có ID hợp lệ để tránh HTTP 422 khi xóa/đánh giá
      if (state.words && Array.isArray(state.words)) {
        state.words.push(savedWord);
      }
      loadStats();
      showToast(`Đã lưu "${word}" vào kho từ vựng!`);
    } else {
      if (btnEl) {
        btnEl.disabled = false;
        btnEl.innerHTML = '<i data-lucide="plus" class="w-3.5 h-3.5"></i>';
        lucide.createIcons();
      }
      showToast(`Không thể lưu từ "${word}"`);
    }
  } catch (err) {
    if (btnEl) {
      btnEl.disabled = false;
      btnEl.innerHTML = '<i data-lucide="plus" class="w-3.5 h-3.5"></i>';
      lucide.createIcons();
    }
    console.error('Error saving quick thesaurus word:', err);
    showToast('Lỗi kết nối khi lưu từ');
  }
}

// ==========================================
// 7. INTERACTIVE DOCUMENT READER MODE
// ==========================================
let isReaderDelegated = false;
function setupReaderDelegation() {
  if (isReaderDelegated) return;
  const container = document.getElementById('reader-text-container');
  if (!container) return;

  container.addEventListener('click', (e) => {
    const speakBtn = e.target.closest('.reader-speak-btn');
    if (speakBtn) {
      const word = speakBtn.dataset.word;
      if (word) speakText(word);
      return;
    }

    const wordEl = e.target.closest('.reader-word');
    if (wordEl) {
      const word = wordEl.getAttribute('data-word');
      const sentence = wordEl.getAttribute('data-sentence') || '';
      if (word) lookupWord(word, sentence);
    }
  });

  isReaderDelegated = true;
}

let readerAbortCtrl = null;

async function loadReaderDoc(docId) {
  const deleteBtn = document.getElementById('btn-delete-doc');
  if (!docId) {
    if (readerAbortCtrl) {
      readerAbortCtrl.abort();
      readerAbortCtrl = null;
    }
    if (deleteBtn) deleteBtn.disabled = true;
    state.currentDoc = null;
    document.getElementById('reader-text-container').innerHTML = '<p class="text-slate-400 dark:text-zinc-500 italic text-center py-20">Vui lòng chọn hoặc tải lên một tài liệu PDF/Word để bắt đầu đọc tương tác.</p>';
    return;
  }
  if (deleteBtn) deleteBtn.disabled = false;
  const container = document.getElementById('reader-text-container');
  container.innerHTML = `
    <div class="flex flex-col items-center justify-center py-20 space-y-3">
      <div class="w-8 h-8 border-4 border-indigo-200 dark:border-zinc-700 border-t-indigo-600 rounded-full animate-spin"></div>
      <p class="text-sm font-medium text-slate-500 dark:text-zinc-400">Đang tải và chuẩn bị nội dung văn bản...</p>
    </div>
  `;

  setupReaderDelegation();

  if (readerAbortCtrl) {
    readerAbortCtrl.abort();
  }
  readerAbortCtrl = new AbortController();
  const signal = readerAbortCtrl.signal;

  try {
    const res = await fetch(`/api/documents/${docId}`, { signal });
    if (!res.ok) throw new Error('Không thể tải tài liệu');
    const doc = await res.json();
    state.currentDoc = doc;

    const savedWordSet = new Set((state.words || []).map(sw => sw.word.toLowerCase()));
    const rawLines = (doc.text_content || '').split('\n').map(l => l.trim()).filter(l => l.length > 0);

    const VN_DIACRITICS = /[àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]/i;
    const IPA_LINE_REGEX = /^([a-zA-Z\s\-\(\)\'\.]+?)\s+(adj|v|n|adv|prep|conj|pron|phr|phrase|verb phrase|noun phrase|adjective phrase|adverb phrase)?\s*(\/[^\/\n]+\/)\s*(.*)$/i;

    let html = '';
    let tableRows = [];

    const flushTable = () => {
      if (tableRows.length === 0) return '';
      let tbl = `
        <div class="overflow-x-auto rounded-2xl border border-slate-200 dark:border-zinc-800 shadow-sm mb-6 bg-white dark:bg-zinc-950/60">
          <table class="w-full text-left border-collapse text-sm">
            <thead class="bg-slate-100/90 dark:bg-zinc-900 border-b border-slate-200 dark:border-zinc-800 text-[11px] font-bold text-slate-500 dark:text-zinc-400 uppercase tracking-wider">
              <tr>
                <th class="py-2.5 px-3.5">Từ vựng</th>
                <th class="py-2.5 px-2">Loại từ</th>
                <th class="py-2.5 px-3 font-mono">Phiên âm</th>
                <th class="py-2.5 px-3.5">Dịch nghĩa</th>
                <th class="py-2.5 px-2 text-center w-12">Phát âm</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100 dark:divide-zinc-800/60">
      `;
      tableRows.forEach(row => {
        const cleanW = row.word.toLowerCase();
        const isSaved = savedWordSet.has(cleanW);
        const savedClass = isSaved ? 'is-saved' : '';
        const displaySentence = row.sentence || `${row.word}: ${row.meaning}`;
        tbl += `
          <tr class="hover:bg-slate-50/80 dark:hover:bg-zinc-800/60 transition">
            <td class="py-2 px-3.5 font-semibold text-slate-800 dark:text-zinc-100">
              <span class="reader-word ${savedClass} cursor-pointer" data-word="${escapeHtml(row.word)}" data-sentence="${escapeHtml(displaySentence)}">${escapeHtml(row.word)}</span>
            </td>
            <td class="py-2 px-2">
              <span class="px-1.5 py-0.5 text-[11px] rounded bg-slate-100 dark:bg-zinc-800 text-slate-500 dark:text-zinc-400 font-mono">${escapeHtml(row.pos || '-')}</span>
            </td>
            <td class="py-2 px-3 font-mono text-xs text-slate-500 dark:text-zinc-400">${escapeHtml(row.ipa || '')}</td>
            <td class="py-2 px-3.5 text-slate-700 dark:text-zinc-300 font-normal">${escapeHtml(row.meaning || '')}</td>
            <td class="py-2 px-2 text-center">
              <button type="button" class="reader-speak-btn p-1 text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition cursor-pointer" data-word="${escapeHtml(row.word)}" title="Nghe phát âm">
                <i data-lucide="volume-2" class="w-4 h-4 inline"></i>
              </button>
            </td>
          </tr>
        `;
      });
      tbl += `
            </tbody>
          </table>
        </div>
      `;
      tableRows = [];
      return tbl;
    };

    for (let i = 0; i < rawLines.length; i++) {
      const line = rawLines[i];

      // Format 1: Bảng phân cách bằng dấu pipe | (như Doc 11 trích xuất chuẩn)
      if (line.includes('|')) {
        const parts = line.split('|').map(p => p.trim());
        const headerTest = parts[0].toLowerCase();
        if (headerTest.includes('từ vựng') || headerTest.includes('từvựng') || headerTest === 'word') {
          continue; // Bỏ qua tiêu đề cột bảng gốc
        }
        if (parts.length >= 2) {
          tableRows.push({
            word: parts[0],
            pos: parts[1] || '',
            ipa: parts[2] || '',
            meaning: parts[3] || '',
            sentence: parts[4] || ''
          });
          continue;
        }
      }

      // Format 2: Danh sách từ có kèm phiên âm /.../
      const ipaMatch = line.match(IPA_LINE_REGEX);
      if (ipaMatch) {
        tableRows.push({
          word: ipaMatch[1].trim(),
          pos: ipaMatch[2] ? ipaMatch[2].trim() : '',
          ipa: ipaMatch[3].trim(),
          meaning: ipaMatch[4] ? ipaMatch[4].trim() : ''
        });
        continue;
      }

      // Đẩy các hàng bảng tích lũy ra trước khi vẽ đoạn văn/tiêu đề tiếp theo
      if (tableRows.length > 0) {
        html += flushTable();
      }

      // Format 3: Tiêu đề mục hoặc chủ đề (viết hoa hoặc bắt đầu bằng Chủ đề/Topic/Unit)
      if (line.length < 60 && (line.toUpperCase() === line || line.startsWith('Chủ đề') || line.startsWith('Topic') || line.startsWith('Unit') || line.startsWith('Phần'))) {
        html += `<h4 class="font-bold text-slate-800 dark:text-zinc-100 mt-5 mb-2.5 text-xs sm:text-sm tracking-wide uppercase border-b border-slate-200 dark:border-zinc-800 pb-1">${escapeHtml(line)}</h4>`;
        continue;
      }

      // Bỏ qua dòng tiêu đề bảng độc lập thừa
      if (/^(từ\s*vựng|loại\s*từ|phiên\s*âm|dịch\s*nghĩa)/i.test(line)) {
        continue;
      }

      // Format 4: Đoạn văn bản bình thường (tách IPA và bọc từ tiếng Anh chuẩn xác)
      const parts = line.split(/(\/[^\/\n]+\/)/g);
      let lineHtml = '';

      for (let pIdx = 0; pIdx < parts.length; pIdx++) {
        const part = parts[pIdx];
        if (pIdx % 2 === 1) {
          // Phiên âm IPA: giữ nguyên không bọc lookup
          lineHtml += `<span class="font-mono text-xs text-slate-400 dark:text-zinc-500">${escapeHtml(part)}</span>`;
        } else {
          // Tách từ tiếng Anh
          const tokens = part.split(/(\b[a-zA-Z]+(?:[-'][a-zA-Z]+)*\b)/g);
          tokens.forEach(tok => {
            if (/^[a-zA-Z\']+$/.test(tok) && tok.length >= 2 && !VN_DIACRITICS.test(tok)) {
              const clean = tok.toLowerCase();
              const isSaved = savedWordSet.has(clean);
              const savedClass = isSaved ? 'is-saved' : '';
              lineHtml += `<span class="reader-word ${savedClass} cursor-pointer" data-word="${escapeHtml(tok)}" data-sentence="${escapeHtml(line)}">${escapeHtml(tok)}</span>`;
            } else {
              lineHtml += escapeHtml(tok);
            }
          });
        }
      }

      html += `<p class="mb-3 leading-relaxed text-slate-700 dark:text-zinc-300">${lineHtml}</p>`;
    }

    if (tableRows.length > 0) {
      html += flushTable();
    }

    container.innerHTML = html || '<p class="text-slate-400 dark:text-zinc-500 italic text-center py-20">Tài liệu không có nội dung hiển thị.</p>';
    if (window.lucide) lucide.createIcons();
  } catch (err) {
    if (err.name === 'AbortError') return;
    container.innerHTML = `
      <div class="p-6 rounded-2xl bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900/50 text-center my-8 max-w-md mx-auto space-y-2">
        <p class="text-rose-600 dark:text-rose-400 font-semibold">Lỗi: ${escapeHtml(err.message || 'Không thể tải tài liệu')}</p>
        <button onclick="loadReaderDoc(${escapeHtml(String(docId))})" class="px-4 py-1.5 bg-rose-600 hover:bg-rose-700 text-white rounded-xl text-xs font-bold transition cursor-pointer">Thử lại</button>
      </div>
    `;
  }
}

async function deleteCurrentDocument() {
  if (!state.currentDoc) {
    showToast('Vui lòng chọn tài liệu cần xóa.');
    return;
  }
  const filename = state.currentDoc.filename;
  const confirmed = confirm(`Bạn có chắc chắn muốn xóa tài liệu "${filename}" khỏi hệ thống?\n\nLưu ý: Các từ vựng trích xuất từ tài liệu này cũng sẽ được dọn dẹp.`);
  if (!confirmed) return;

  try {
    const res = await fetch(`/api/documents/${state.currentDoc.id}`, { method: 'DELETE' });
    if (res.ok) {
      showToast(`Đã xóa tài liệu "${filename}"`);
      state.currentDoc = null;
      const deleteBtn = document.getElementById('btn-delete-doc');
      if (deleteBtn) deleteBtn.disabled = true;
      document.getElementById('reader-text-container').innerHTML = '<p class="text-slate-400 dark:text-zinc-500 italic text-center py-20">Tài liệu đã được xóa. Vui lòng chọn tài liệu khác hoặc tải file mới.</p>';
      
      // Reset lookup panel
      document.getElementById('lookup-word').textContent = '...';
      document.getElementById('lookup-ipa').textContent = '/.../';
      document.getElementById('lookup-meaning').textContent = 'Click vào một từ trong văn bản để xem nghĩa';
      document.getElementById('lookup-sentence').textContent = '...';
      document.getElementById('lookup-badge').textContent = 'Sẵn sàng';
      document.getElementById('lookup-badge').className = 'px-2 py-0.5 text-[10px] font-bold rounded-full bg-slate-200 dark:bg-zinc-800 text-slate-600 dark:text-zinc-300';
      document.getElementById('lookup-save-btn').disabled = true;

      await loadDocuments();
      await loadWords();
      await loadStats();
    } else {
      showToast('Lỗi khi xóa tài liệu');
    }
  } catch (err) {
    console.error(err);
    showToast('Lỗi kết nối máy chủ');
  }
}

async function clearAllDocuments() {
  const select = document.getElementById('reader-doc-select');
  const count = select ? select.options.length - 1 : 0;
  if (count <= 0) {
    showToast('Không có tài liệu nào để xóa.');
    return;
  }
  const confirmed = confirm(`⚠️ CẢNH BÁO: Bạn có chắc chắn muốn xóa TOÀN BỘ ${count} tài liệu đã tải lên trong hệ thống?\n\nThao tác này không thể hoàn tác!`);
  if (!confirmed) return;

  try {
    const res = await fetch('/api/documents', { method: 'DELETE' });
    if (res.ok) {
      state.currentDoc = null;
      const deleteBtn = document.getElementById('btn-delete-doc');
      if (deleteBtn) deleteBtn.disabled = true;
      document.getElementById('reader-text-container').innerHTML = '<p class="text-slate-400 dark:text-zinc-500 italic text-center py-20">Đã xóa sạch tất cả tài liệu. Hãy tải tài liệu mới để bắt đầu học.</p>';
      await loadDocuments();
      await loadStats();
      showToast('🗑️ Đã xóa sạch toàn bộ tài liệu đã tải lên!');
    } else {
      showToast('Lỗi khi xóa tài liệu');
    }
  } catch (err) {
    console.error(err);
    showToast('Lỗi kết nối máy chủ');
  }
}

async function lookupWord(word, contextSentence = '') {
  const cleanWord = (word || '').trim();
  if (!cleanWord) return;
  state.activeLookupWord = cleanWord;

  // Phát âm ngay lập tức trong sự kiện click của người dùng để bảo toàn User Gesture token
  speakText(cleanWord);

  document.getElementById('lookup-word').textContent = cleanWord;
  document.getElementById('lookup-badge').textContent = 'Đang tra cứu...';
  document.getElementById('lookup-badge').className = 'px-2 py-0.5 text-[10px] font-bold rounded-full bg-slate-200 dark:bg-zinc-800 text-slate-600 dark:text-zinc-300';

  // Xác định câu ngữ cảnh
  let ctx = (contextSentence || '').trim();
  if (!ctx && state.currentDoc && state.currentDoc.text_content) {
    const lines = state.currentDoc.text_content.split('\n');
    const matchedLine = lines.find(l => new RegExp(`\\b${escapeRegex(cleanWord)}\\b`, 'i').test(l));
    if (matchedLine) {
      ctx = matchedLine.replace(/\|/g, ' - ').trim();
    }
  }
  if (!ctx) ctx = `The word "${cleanWord}" was found in the text.`;
  document.getElementById('lookup-sentence').textContent = ctx;
  state.activeLookupContext = ctx;

  // Kiểm tra xem từ đã lưu trong kho hay chưa
  const saved = (state.words || []).find(w => w.word.toLowerCase() === cleanWord.toLowerCase());
  if (saved) {
    document.getElementById('lookup-ipa').textContent = saved.ipa || `/${cleanWord}/`;
    document.getElementById('lookup-meaning').textContent = saved.meaning_vi;
    if (saved.example_sentence && saved.example_sentence !== 'Trong bài đọc') {
      document.getElementById('lookup-sentence').textContent = saved.example_sentence;
    }
    document.getElementById('lookup-badge').textContent = 'Đã lưu';
    document.getElementById('lookup-badge').className = 'px-2 py-0.5 text-[10px] font-bold rounded-full bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300';
    document.getElementById('lookup-save-btn').disabled = true;
    document.getElementById('lookup-save-btn').innerHTML = '<i data-lucide="check" class="w-4 h-4 mr-1"></i><span>Đã có trong kho</span>';
  } else {
    // Chưa lưu: tra nghĩa động
    document.getElementById('lookup-ipa').textContent = `/${cleanWord}/`;
    document.getElementById('lookup-badge').textContent = 'Chưa lưu';
    document.getElementById('lookup-badge').className = 'px-2 py-0.5 text-[10px] font-bold rounded-full bg-amber-100 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300';
    
    const saveBtn = document.getElementById('lookup-save-btn');
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<i data-lucide="bookmark-plus" class="w-4 h-4 mr-1"></i><span>+ Thêm vào danh sách học</span>';

    // BỘ TĂNG TỐC: Kiểm tra cache client translateCache trước
    const cacheKey = cleanWord.toLowerCase();
    if (translateCache.has(cacheKey)) {
      const cached = translateCache.get(cacheKey);
      document.getElementById('lookup-meaning').textContent = cached.meaning_vi || cleanWord;
      if (cached.ipa) document.getElementById('lookup-ipa').textContent = cached.ipa;
      saveBtn.disabled = false;
      if (window.lucide) lucide.createIcons();
      return;
    }

    document.getElementById('lookup-meaning').textContent = 'Đang tra từ điển...';

    try {
      const transRes = await fetch(`/api/translate?word=${encodeURIComponent(cleanWord)}`);
      if (transRes.ok) {
        const transData = await transRes.json();
        // Lưu vào translateCache (LRU - giới hạn 1000 từ)
        if (translateCache.size >= 1000) {
          const oldestKey = translateCache.keys().next().value;
          translateCache.delete(oldestKey);
        }
        translateCache.set(cacheKey, transData);

        if (state.activeLookupWord === cleanWord) {
          document.getElementById('lookup-meaning').textContent = transData.meaning_vi || cleanWord;
          if (transData.ipa) document.getElementById('lookup-ipa').textContent = transData.ipa;
          saveBtn.disabled = false;
        }
      } else {
        document.getElementById('lookup-meaning').textContent = '(Chưa có bản dịch)';
        saveBtn.disabled = false;
      }
    } catch (e) {
      document.getElementById('lookup-meaning').textContent = '(Lỗi kết nối tra từ)';
      saveBtn.disabled = false;
    }
  }

  if (window.lucide) lucide.createIcons();
}

function speakLookupWord() {
  if (state.activeLookupWord) {
    speakText(state.activeLookupWord);
  }
}

function speakLookupSentence() {
  const sentenceEl = document.getElementById('lookup-sentence');
  if (sentenceEl && sentenceEl.textContent && sentenceEl.textContent !== '...') {
    // Nếu câu chứa nhiều cột phân cách, chỉ lấy phần câu/từ tiếng Anh thích hợp hoặc đọc toàn bộ
    const text = sentenceEl.textContent.trim();
    speakText(text);
  }
}

async function saveLookupWordToStudy() {
  if (!state.activeLookupWord) return;
  const word = state.activeLookupWord;
  const meaning = document.getElementById('lookup-meaning').textContent;
  const ipa = document.getElementById('lookup-ipa').textContent;
  const sentence = document.getElementById('lookup-sentence').textContent;

  try {
    const res = await fetch('/api/words/manual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        word,
        meaning_vi: meaning,
        ipa,
        pos: 'word',
        example_sentence: sentence
      })
    });
    if (res.ok) {
      showToast(`Đã thêm "${word}" vào kho từ vựng`);
      await loadWords();
      await loadStats();

      // Cập nhật ngay lập tức highlight xanh (.is-saved) trên bài đọc
      document.querySelectorAll('#reader-text-container .reader-word').forEach(el => {
        const w = el.getAttribute('data-word') || el.textContent;
        if (w && w.trim().toLowerCase() === word.trim().toLowerCase()) {
          el.classList.add('is-saved');
        }
      });

      // Xóa cache thesaurus để trạng thái is_saved được làm mới
      thesaurusCache.clear();
      lookupWord(word, sentence); // refresh card state
    }
  } catch (err) {
    showToast('Lỗi khi lưu từ');
  }
}

// ==========================================
// 7. VOCABULARY MANAGEMENT TABLE (TỐI ƯU HÓA HIỆU NĂNG CHO HÀNG NGHÌN TỪ)
// ==========================================
let vocabTablePage = 1;
let vocabTablePageSize = 50;
let vocabDebounceTimer = null;

function debounceFilterVocabTable() {
  clearTimeout(vocabDebounceTimer);
  vocabDebounceTimer = setTimeout(() => {
    vocabTablePage = 1;
    renderVocabTable();
  }, 180);
}

function filterVocabTable() {
  vocabTablePage = 1;
  renderVocabTable();
}

function changeVocabPageSize(newSize) {
  vocabTablePageSize = parseInt(newSize) || 50;
  vocabTablePage = 1;
  renderVocabTable();
}

function changeVocabPage(action) {
  const search = (document.getElementById('vocab-search-input')?.value || '').toLowerCase();
  const statusFilter = document.getElementById('vocab-filter-status')?.value || 'all';

  const filtered = (state.words || []).filter(w => {
    const matchesSearch = !search || w.word.toLowerCase().includes(search) || w.meaning_vi.toLowerCase().includes(search);
    const matchesStatus = (statusFilter === 'all') || (w.status === statusFilter);
    return matchesSearch && matchesStatus;
  });

  const totalPages = Math.max(1, Math.ceil(filtered.length / vocabTablePageSize));
  if (action === 'first') vocabTablePage = 1;
  else if (action === 'prev') vocabTablePage = Math.max(1, vocabTablePage - 1);
  else if (action === 'next') vocabTablePage = Math.min(totalPages, vocabTablePage + 1);
  else if (action === 'last') vocabTablePage = totalPages;

  renderVocabTable();
}

function renderVocabTable() {
  const tbody = document.getElementById('vocab-table-body');
  if (!tbody) return;

  const search = (document.getElementById('vocab-search-input')?.value || '').toLowerCase();
  const statusFilter = document.getElementById('vocab-filter-status')?.value || 'all';

  // Lọc từ vựng trong bộ nhớ cực nhanh
  const filtered = (state.words || []).filter(w => {
    const matchesSearch = !search || w.word.toLowerCase().includes(search) || w.meaning_vi.toLowerCase().includes(search);
    const matchesStatus = (statusFilter === 'all') || (w.status === statusFilter);
    return matchesSearch && matchesStatus;
  });

  const totalItems = filtered.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / vocabTablePageSize));
  if (vocabTablePage > totalPages) vocabTablePage = totalPages;
  if (vocabTablePage < 1) vocabTablePage = 1;

  // Cập nhật thông tin phân trang
  const startIdx = (vocabTablePage - 1) * vocabTablePageSize;
  const endIdx = Math.min(startIdx + vocabTablePageSize, totalItems);

  const pageInfo = document.getElementById('vocab-page-info');
  if (pageInfo) {
    if (totalItems === 0) {
      pageInfo.textContent = '0 từ vựng';
    } else {
      pageInfo.textContent = `Đang hiển thị ${startIdx + 1} - ${endIdx} trên tổng số ${totalItems} từ`;
    }
  }

  const pageCurrent = document.getElementById('vocab-page-current');
  if (pageCurrent) {
    pageCurrent.textContent = `Trang ${vocabTablePage} / ${totalPages}`;
  }

  const btnFirst = document.getElementById('btn-page-first');
  const btnPrev = document.getElementById('btn-page-prev');
  const btnNext = document.getElementById('btn-page-next');
  const btnLast = document.getElementById('btn-page-last');

  if (btnFirst) btnFirst.disabled = (vocabTablePage <= 1);
  if (btnPrev) btnPrev.disabled = (vocabTablePage <= 1);
  if (btnNext) btnNext.disabled = (vocabTablePage >= totalPages);
  if (btnLast) btnLast.disabled = (vocabTablePage >= totalPages);

  if (totalItems === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="py-8 text-center text-slate-400">Không tìm thấy từ vựng nào</td></tr>`;
    return;
  }

  // CHỈ RENDER 50 TỪ CỦA TRANG HIỆN TẠI (Giảm từ 25.000 phần tử DOM xuống còn vài chục phần tử)
  const pageItems = filtered.slice(startIdx, endIdx);

  tbody.innerHTML = pageItems.map(w => {
    let statusBadge = '<span class="px-2 py-0.5 text-xs font-semibold rounded-full bg-slate-100 text-slate-600">Mới</span>';
    if (w.status === 'learning') statusBadge = '<span class="px-2 py-0.5 text-xs font-semibold rounded-full bg-indigo-50 text-indigo-700">Đang học</span>';
    if (w.status === 'mastered') statusBadge = '<span class="px-2 py-0.5 text-xs font-semibold rounded-full bg-emerald-50 text-emerald-700">Thành thạo</span>';

    const rawWord = w.word || '';
    return `
      <tr class="hover:bg-slate-50/80 dark:hover:bg-zinc-800/60 transition">
        <td class="py-3 px-4 font-bold text-slate-900 dark:text-zinc-100 flex items-center space-x-2">
          <button data-word="${escapeHtml(rawWord)}" onclick="speakText(this.getAttribute('data-word'))" class="text-indigo-600 hover:text-indigo-800 dark:text-indigo-400 p-1 cursor-pointer" title="Phát âm">
            <i data-lucide="volume-2" class="w-4 h-4"></i>
          </button>
          <span>${escapeHtml(rawWord)}</span>
        </td>
        <td class="py-3 px-4 text-slate-400 font-mono text-xs">${escapeHtml(w.ipa || `/${rawWord}/`)}</td>
        <td class="py-3 px-4 font-medium text-slate-800 dark:text-zinc-200">${escapeHtml(cleanVietnamese(w.meaning_vi))}</td>
        <td class="py-3 px-4"><span class="px-2 py-0.5 text-[10px] font-bold rounded bg-slate-100 dark:bg-zinc-800 text-slate-600 dark:text-zinc-300">${escapeHtml(w.cefr_level || 'B1')}</span></td>
        <td class="py-3 px-4">${statusBadge}</td>
        <td class="py-3 px-4 text-xs text-slate-500 dark:text-zinc-400">${parseInt(w.repetitions, 10) || 0} lần (${parseInt(w.interval, 10) || 1} ngày)</td>
        <td class="py-3 px-4 text-right">
          <button onclick="deleteWord(${parseInt(w.id, 10)})" class="text-rose-500 hover:text-rose-700 p-1 rounded hover:bg-rose-50 dark:hover:bg-rose-950/40 transition cursor-pointer" title="Xóa từ này">
            <i data-lucide="trash-2" class="w-4 h-4"></i>
          </button>
        </td>
      </tr>
    `;
  }).join('');

  // Chỉ scan icon bên trong bảng và thanh phân trang thay vì toàn bộ trang web
  lucide.createIcons();
}

async function deleteWord(wordId) {
  if (!confirm('Bạn có chắc chắn muốn xóa từ vựng này khỏi kho?')) return;
  try {
    const res = await fetch(`/api/words/${wordId}`, { method: 'DELETE' });
    if (res.ok) {
      showToast('Đã xóa từ vựng');
      thesaurusCache.clear();
      state.words = state.words.filter(w => w.id !== wordId);
      state.dueWords = (state.dueWords || []).filter(w => w.id !== wordId);
      if (state.currentCardIndex >= state.words.length) {
        state.currentCardIndex = Math.max(0, state.words.length - 1);
      }
      if (state.srsIndex >= (state.dueWords.length || 0)) {
        state.srsIndex = Math.max(0, (state.dueWords.length || 0) - 1);
      }
      renderVocabTable();
      renderFlashcard();
      renderSrsCard();
      loadStats();
    }
  } catch (err) {
    showToast('Lỗi khi xóa từ');
  }
}

async function clearAllWords() {
  if (!state.words || state.words.length === 0) {
    showToast('Kho từ vựng hiện đang trống.');
    return;
  }
  const count = state.words.length;
  const confirmed = confirm(`⚠️ CẢNH BÁO: Bạn có chắc chắn muốn xóa TOÀN BỘ ${count} từ vựng đã lưu trong kho?\n\nThao tác này sẽ xóa sạch danh sách từ và toàn bộ tiến độ ôn tập (SRS), không thể hoàn tác!`);
  if (!confirmed) return;

  try {
    const res = await fetch('/api/words', { method: 'DELETE' });
    if (res.ok) {
      thesaurusCache.clear();
      state.words = [];
      state.dueWords = [];
      state.currentCardIndex = 0;
      state.srsIndex = 0;
      renderVocabTable();
      renderFlashcard();
      renderSrsCard();
      loadStats();
      showToast(`🗑️ Đã xóa sạch toàn bộ ${count} từ vựng trong kho!`);
    } else {
      showToast('Có lỗi xảy ra khi xóa danh sách từ');
    }
  } catch (err) {
    console.error('Error clearing words:', err);
    showToast('Lỗi kết nối máy chủ');
  }
}

// ==========================================
// File Upload & Modal Handlers
// ==========================================
let selectedUploadFile = null;

function openUploadModal() {
  document.getElementById('modal-upload').classList.remove('hidden');
}

function closeUploadModal() {
  document.getElementById('modal-upload').classList.add('hidden');
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;
  selectedUploadFile = file;
  document.getElementById('drop-text').innerHTML = `Đã chọn: <strong class="text-indigo-600">${escapeHtml(file.name)}</strong> (${(file.size / 1024).toFixed(1)} KB)`;
}

async function startUpload() {
  if (!selectedUploadFile) {
    showToast('Vui lòng chọn một file PDF hoặc DOCX trước');
    return;
  }

  const btn = document.getElementById('btn-submit-upload');
  btn.disabled = true;
  btn.textContent = 'Đang trích xuất từ vựng...';

  const formData = new FormData();
  formData.append('file', selectedUploadFile);
  formData.append('max_words', document.getElementById('upload-max-words').value);
  if (state.geminiApiKey) {
    formData.append('gemini_api_key', state.geminiApiKey);
    formData.append('gemini_model', state.geminiModel || 'gemini-3.7-flash');
  }

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Lỗi xử lý file');

    closeUploadModal();
    showToast(`🎉 Trích xuất thành công ${data.extracted_count} từ vựng từ ${data.filename}!`);
    triggerConfetti();

    // Reload all data
    await loadWords();
    await loadStats();
    await loadDocuments();

    // Reset upload state
    selectedUploadFile = null;
    document.getElementById('drop-text').innerHTML = `Kéo thả file vào đây hoặc <span class="text-indigo-600 underline">chọn từ máy tính</span>`;
    
    // Switch to flashcards
    switchTab('flashcards');
  } catch (err) {
    alert(`Lỗi trích xuất: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Bắt đầu trích xuất';
  }
}

// ==========================================
// Theme & Appearance Handlers
// ==========================================
function initTheme() {
  const savedMode = localStorage.getItem('vocabmaster_theme_mode') || 'system';
  const savedColor = localStorage.getItem('vocabmaster_theme_color') || 'indigo';

  applyThemeColor(savedColor);
  applyThemeMode(savedMode);
  initFont();
  initFontScale();

  if (window.matchMedia && !state.isThemeListenerAttached) {
    state.isThemeListenerAttached = true;
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
      if (state.themeMode === 'system') {
        applyThemeMode('system');
      }
    });
  }
}

function applyThemeMode(mode) {
  state.themeMode = mode;
  let isDark = false;
  if (mode === 'dark') {
    isDark = true;
  } else if (mode === 'light') {
    isDark = false;
  } else {
    isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  if (isDark) {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }

  document.querySelectorAll('.theme-mode-btn').forEach(btn => {
    if (btn.dataset.mode === mode) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  applyBackgroundDOM();
}

function setThemeMode(mode) {
  localStorage.setItem('vocabmaster_theme_mode', mode);
  applyThemeMode(mode);
}

function applyThemeColor(color) {
  state.themeColor = color;
  document.documentElement.setAttribute('data-theme', color);

  document.querySelectorAll('.theme-color-btn').forEach(btn => {
    if (btn.dataset.color === color) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
}

function setThemeColor(color) {
  localStorage.setItem('vocabmaster_theme_color', color);
  applyThemeColor(color);
}

// ==========================================
// Typography & Font Handlers
// ==========================================
const APP_FONTS = {
  'be-vietnam-pro': {
    name: 'Be Vietnam Pro',
    label: 'Be Vietnam Pro ⭐ (Chuẩn tiếng Việt)',
    family: "'Be Vietnam Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    letterSpacing: '-0.012em'
  },
  'lexend': {
    name: 'Lexend',
    label: 'Lexend (Thoáng mắt, chống mỏi)',
    family: "'Lexend', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    letterSpacing: '-0.005em'
  },
  'plus-jakarta-sans': {
    name: 'Plus Jakarta Sans',
    label: 'Plus Jakarta Sans (Bo góc quốc tế)',
    family: "'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    letterSpacing: '-0.012em'
  },
  'inter': {
    name: 'Inter',
    label: 'Inter (Chuẩn công nghệ)',
    family: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    letterSpacing: '-0.011em'
  },
  'nunito': {
    name: 'Nunito',
    label: 'Nunito (Mềm mại thân thiện)',
    family: "'Nunito', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    letterSpacing: '0em'
  }
};

function initFont() {
  const savedFont = localStorage.getItem('vocabmaster_font') || 'be-vietnam-pro';
  applyFont(savedFont);
}

function applyFont(fontKey) {
  const fontObj = APP_FONTS[fontKey] || APP_FONTS['be-vietnam-pro'];
  state.currentFont = fontKey;

  document.documentElement.style.setProperty('--app-font', fontObj.family);
  document.documentElement.style.setProperty('--app-letter-spacing', fontObj.letterSpacing);

  const selectEl = document.getElementById('settings-font-family');
  if (selectEl) selectEl.value = fontKey;

  const previewLabel = document.getElementById('font-current-preview');
  if (previewLabel) previewLabel.textContent = fontObj.name;
}

function setFontFamily(fontKey) {
  localStorage.setItem('vocabmaster_font', fontKey);
  applyFont(fontKey);
  showToast(`Đã đổi phông chữ sang ${APP_FONTS[fontKey]?.name || fontKey}`);
}

function initFontScale() {
  const savedScale = localStorage.getItem('vocabmaster_font_scale') || 'normal';
  applyFontScale(savedScale);
}

function applyFontScale(scale) {
  state.fontScale = scale;
  if (scale === 'normal') {
    document.documentElement.removeAttribute('data-font-scale');
  } else {
    document.documentElement.setAttribute('data-font-scale', scale);
  }

  document.querySelectorAll('.font-scale-btn').forEach(btn => {
    if (btn.dataset.scale === scale) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
}

function setFontScale(scale) {
  localStorage.setItem('vocabmaster_font_scale', scale);
  applyFontScale(scale);
  const scaleNames = { 'normal': 'Vừa (100%)', 'medium': 'Lớn (+10%)', 'large': 'Rất lớn (+20%)' };
  showToast(`Đã đổi cỡ chữ sang ${scaleNames[scale] || scale}`);
}

// ==========================================
// Custom Background Handlers
// ==========================================
function initBackground() {
  const savedUrl = localStorage.getItem('vocabmaster_bg_url') || '';
  const savedOverlay = parseInt(localStorage.getItem('vocabmaster_bg_overlay') || '35', 10);
  const savedBlur = parseInt(localStorage.getItem('vocabmaster_bg_blur') || '0', 10);
  const savedPreset = localStorage.getItem('vocabmaster_bg_preset') || 'none';

  state.bgUrl = savedUrl;
  state.bgOverlay = savedOverlay;
  state.bgBlur = savedBlur;
  state.bgPreset = savedPreset;

  applyBackgroundDOM();
}

function applyBackgroundDOM() {
  const bgLayer = document.getElementById('app-bg-layer');
  const bgOverlay = document.getElementById('app-bg-overlay');
  const removeBtn = document.getElementById('btn-remove-bg');
  const controlsPanel = document.getElementById('bg-controls-panel');
  const overlaySlider = document.getElementById('bg-overlay-slider');
  const blurSlider = document.getElementById('bg-blur-slider');
  const overlayVal = document.getElementById('bg-overlay-val');
  const blurVal = document.getElementById('bg-blur-val');

  if (state.bgUrl) {
    document.body.classList.add('has-custom-bg');
    document.documentElement.classList.add('has-custom-bg');
    if (bgLayer) {
      bgLayer.style.backgroundImage = `url("${state.bgUrl}")`;
      bgLayer.style.filter = `blur(${state.bgBlur}px)`;
      bgLayer.style.opacity = '1';
    }
    if (bgOverlay) {
      const isDark = document.documentElement.classList.contains('dark');
      const alpha = (state.bgOverlay / 100).toFixed(2);
      bgOverlay.style.backgroundColor = isDark ? `rgba(0, 0, 0, ${alpha})` : `rgba(255, 255, 255, ${alpha})`;
      bgOverlay.style.opacity = '1';
    }
    if (removeBtn) removeBtn.classList.remove('hidden');
    if (controlsPanel) controlsPanel.classList.remove('hidden');
  } else {
    document.body.classList.remove('has-custom-bg');
    document.documentElement.classList.remove('has-custom-bg');
    if (bgLayer) {
      bgLayer.style.backgroundImage = 'none';
      bgLayer.style.opacity = '0';
    }
    if (bgOverlay) {
      bgOverlay.style.opacity = '0';
    }
    if (removeBtn) removeBtn.classList.add('hidden');
    if (controlsPanel) controlsPanel.classList.add('hidden');
  }

  if (overlaySlider) overlaySlider.value = state.bgOverlay;
  if (overlayVal) overlayVal.textContent = `${state.bgOverlay}%`;
  if (blurSlider) blurSlider.value = state.bgBlur;
  if (blurVal) blurVal.textContent = `${state.bgBlur}px`;

  document.querySelectorAll('.bg-preset-thumb').forEach(el => {
    if (el.dataset.preset === (state.bgUrl ? state.bgPreset : 'none')) {
      el.classList.add('active');
    } else {
      el.classList.remove('active');
    }
  });
}

function setBackgroundPreset(presetKey) {
  if (BG_PRESETS[presetKey]) {
    state.bgUrl = BG_PRESETS[presetKey];
    state.bgPreset = presetKey;
    localStorage.setItem('vocabmaster_bg_url', state.bgUrl);
    localStorage.setItem('vocabmaster_bg_preset', presetKey);
    applyBackgroundDOM();
    showToast('Đã áp dụng hình nền mẫu!');
  }
}

async function handleBackgroundUpload(e) {
  const file = e.target.files && e.target.files[0];
  if (!file) return;

  const statusText = document.getElementById('bg-upload-status');
  if (statusText) statusText.textContent = 'Đang tải ảnh lên...';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload-background', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.detail || data.message || 'Lỗi khi tải ảnh');
    }

    state.bgUrl = data.url;
    state.bgPreset = 'custom';
    localStorage.setItem('vocabmaster_bg_url', state.bgUrl);
    localStorage.setItem('vocabmaster_bg_preset', 'custom');
    applyBackgroundDOM();
    showToast('Tải ảnh nền lên thành công!');
  } catch (err) {
    alert(`Không thể tải ảnh nền: ${err.message}`);
  } finally {
    if (statusText) statusText.textContent = 'Tải ảnh từ máy lên làm hình nền...';
    e.target.value = '';
  }
}

function removeBackground() {
  state.bgUrl = '';
  state.bgPreset = 'none';
  localStorage.removeItem('vocabmaster_bg_url');
  localStorage.setItem('vocabmaster_bg_preset', 'none');
  applyBackgroundDOM();
  showToast('Đã khôi phục nền trơn mặc định.');
}

function updateBackgroundOverlay(val) {
  state.bgOverlay = parseInt(val, 10);
  localStorage.setItem('vocabmaster_bg_overlay', state.bgOverlay);
  applyBackgroundDOM();
}

function updateBackgroundBlur(val) {
  state.bgBlur = parseInt(val, 10);
  localStorage.setItem('vocabmaster_bg_blur', state.bgBlur);
  applyBackgroundDOM();
}

// ==========================================
// Settings Modal Handlers
// ==========================================
function openSettingsModal() {
  applyThemeMode(state.themeMode || localStorage.getItem('vocabmaster_theme_mode') || 'system');
  applyThemeColor(state.themeColor || localStorage.getItem('vocabmaster_theme_color') || 'indigo');
  applyBackgroundDOM();
  document.getElementById('modal-settings').classList.remove('hidden');
  lucide.createIcons();
}

function closeSettingsModal() {
  document.getElementById('modal-settings').classList.add('hidden');
}

async function testGeminiKey() {
  let key = document.getElementById('settings-gemini-key').value.trim();
  key = key.replace(/^["']|["']$/g, '').trim();
  document.getElementById('settings-gemini-key').value = key;

  const statusEl = document.getElementById('settings-key-status');
  if (!key) {
    statusEl.textContent = 'Vui lòng nhập API Key để kiểm tra.';
    statusEl.className = 'text-xs mt-1 text-rose-500';
    return;
  }

  statusEl.textContent = 'Đang kiểm tra kết nối Google Gemini...';
  statusEl.className = 'text-xs mt-1 text-indigo-500';

  const selectedModel = document.getElementById('settings-gemini-model') ? document.getElementById('settings-gemini-model').value : (state.geminiModel || 'gemini-3.7-flash');

  try {
    const res = await fetch('/api/test-gemini-key', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: key, model: selectedModel })
    });
    const data = await res.json();
    if (data.valid) {
      statusEl.textContent = `✓ ${data.message} (Đã tự động lưu)`;
      statusEl.className = 'text-xs mt-1 text-emerald-600 font-bold';
      state.geminiApiKey = key;
      state.geminiModel = data.model || selectedModel;
      localStorage.setItem('vocab_gemini_key', key);
      localStorage.setItem('vocab_gemini_model', state.geminiModel);
    } else {
      statusEl.textContent = `✗ ${data.message}`;
      statusEl.className = 'text-xs mt-1 text-rose-500 font-medium';
    }
  } catch (err) {
    statusEl.textContent = `Lỗi kiểm tra: ${err.message}`;
    statusEl.className = 'text-xs mt-1 text-rose-500';
  }
}

function saveSettings() {
  let key = document.getElementById('settings-gemini-key').value.trim();
  key = key.replace(/^["']|["']$/g, '').trim();
  document.getElementById('settings-gemini-key').value = key;

  const voice = document.getElementById('settings-tts-voice').value;
  const selectedModel = document.getElementById('settings-gemini-model') ? document.getElementById('settings-gemini-model').value : 'gemini-3.7-flash';
  state.geminiApiKey = key;
  state.geminiModel = selectedModel;
  state.ttsVoice = voice;
  localStorage.setItem('vocab_gemini_key', key);
  localStorage.setItem('vocab_gemini_model', selectedModel);
  localStorage.setItem('vocab_tts_voice', voice);
  closeSettingsModal();
  showToast('Đã lưu cài đặt thành công!');
}

// ==========================================
// Add Manual Word Modal Handlers
// ==========================================
function openAddWordModal() {
  document.getElementById('modal-add-word').classList.remove('hidden');
}

function closeAddWordModal() {
  document.getElementById('modal-add-word').classList.add('hidden');
}

async function submitManualWord() {
  const word = document.getElementById('manual-word').value.trim();
  const meaning = document.getElementById('manual-meaning').value.trim();
  const ipa = document.getElementById('manual-ipa').value.trim();
  const pos = document.getElementById('manual-pos').value;
  const example = document.getElementById('manual-example').value.trim();

  if (!word || !meaning) {
    showToast('Vui lòng nhập đầy đủ từ tiếng Anh và nghĩa tiếng Việt.');
    return;
  }

  try {
    const res = await fetch('/api/words/manual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word, meaning_vi: meaning, ipa, pos, example_sentence: example })
    });
    if (res.ok) {
      closeAddWordModal();
      document.getElementById('manual-word').value = '';
      document.getElementById('manual-meaning').value = '';
      document.getElementById('manual-ipa').value = '';
      document.getElementById('manual-example').value = '';
      showToast(`Đã thêm từ "${word}"`);
      thesaurusCache.clear();
      await loadWords();
      await loadStats();
    }
  } catch (err) {
    showToast('Lỗi khi thêm từ');
  }
}

// ==========================================
// Utility Functions (Toast & Confetti)
// ==========================================
function showToast(message) {
  const toast = document.getElementById('toast');
  const msg = document.getElementById('toast-message');
  if (!toast || !msg) return;

  msg.textContent = message;
  toast.classList.remove('translate-y-20', 'opacity-0');
  toast.classList.add('translate-y-0', 'opacity-100');

  setTimeout(() => {
    toast.classList.remove('translate-y-0', 'opacity-100');
    toast.classList.add('translate-y-20', 'opacity-0');
  }, 2500);
}

function triggerConfetti() {
  if (typeof confetti === 'function') {
    confetti({
      particleCount: 45,
      spread: 60,
      origin: { y: 0.6 },
      disableForReducedMotion: true
    });
  }
}


// ==========================================
// FEATURE 1: ADVANCED SRS MODES (CLOZE, SPELLING)
// ==========================================

function getSrsMode() {
  const select = document.getElementById('srs-mode-select');
  return select ? select.value : 'flashcard';
}

let srsTypingTimeout = null;

function checkSrsTyping() {
  const w = (state.dueWords || [])[state.srsIndex];
  if (!w) return;
  const input = document.getElementById('srs-typing-input');
  if (!input) return;
  const userText = input.value.trim().toLowerCase();
  const correctText = w.word.trim().toLowerCase();
  
  const wordEl = document.getElementById('srs-word');
  const ipaEl = document.getElementById('srs-ipa');
  const exampleEl = document.getElementById('srs-example');
  
  wordEl.textContent = w.word;
  wordEl.classList.remove('hidden');
  ipaEl.textContent = w.ipa || `/${w.word}/`;
  const srsSentence = formatCleanSentence(w.example_sentence, w.word);
  if (exampleEl) {
    exampleEl.textContent = srsSentence ? `"${srsSentence}"` : '';
    exampleEl.classList.remove('hidden');
  }
  
  if (srsTypingTimeout) {
    clearTimeout(srsTypingTimeout);
    srsTypingTimeout = null;
  }
  
  document.getElementById('srs-typing-zone').classList.add('hidden');
  revealSrsAnswer();
  
  if (userText === correctText) {
    showToast(`🎉 Chính xác! "${w.word}" (+4 điểm ôn tập)`);
    srsTypingTimeout = setTimeout(() => {
      submitSrsReview(4);
    }, 1800);
  } else {
    showToast(`❌ Chưa đúng! Từ vựng đúng: "${w.word}"`);
    srsTypingTimeout = setTimeout(() => {
      submitSrsReview(1);
    }, 3200);
  }
}

// ==========================================
// PREMADE DECKS & ANKI CSV EXPORT
// ==========================================

function exportWordsToCSV() {
  if (!state.words || state.words.length === 0) {
    showToast('Kho từ vựng đang trống, không có dữ liệu để xuất!');
    return;
  }
  
  let csvContent = "Word,Meaning,IPA,Part of Speech,Sentence,CEFR Level\n";
  
  state.words.forEach(w => {
    const esc = (t) => `"${(t || '').replace(/"/g, '""')}"`;
    csvContent += `${esc(w.word)},${esc(w.meaning_vi)},${esc(w.ipa)},${esc(w.pos)},${esc(w.example_sentence)},${esc(w.cefr_level || 'B1')}\n`;
  });
  
  // Thêm BOM \uFEFF để Excel / Anki nhận dạng UTF-8 tiếng Việt chuẩn xác
  const blob = new Blob(["\uFEFF" + csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `vocabmaster_export_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
  showToast(`✅ Đã xuất thành công ${state.words.length} từ vựng ra file CSV cho Anki!`);
}

function openPremadeDecksModal() {
  const modal = document.getElementById('premade-modal');
  if (modal) {
    modal.classList.remove('hidden');
    modal.classList.add('flex');
    if (window.lucide) lucide.createIcons();
    setTimeout(() => {
      modal.classList.remove('opacity-0');
      const inner = modal.querySelector('div');
      if (inner) inner.classList.remove('scale-95');
    }, 10);
  }
}

function closePremadeModal() {
  const modal = document.getElementById('premade-modal');
  if (modal) {
    modal.classList.add('opacity-0');
    const inner = modal.querySelector('div');
    if (inner) inner.classList.add('scale-95');
    setTimeout(() => {
      modal.classList.add('hidden');
      modal.classList.remove('flex');
    }, 300);
  }
}

async function loadPremadeDeck(type) {
  let deckWords = [];
  if (type === 'toeic') {
    deckWords = [
      { w: 'accommodate', m: 'đáp ứng, điều chỉnh cho phù hợp', i: '/əˈkɒm.ə.deɪt/', pos: 'v', s: 'The new hotel was designed to accommodate higher volumes of visitors.' },
      { w: 'negotiation', m: 'sự đàm phán, thương lượng', i: '/nɪˌɡəʊ.ʃiˈeɪ.ʃən/', pos: 'n', s: 'The contract terms are still under careful negotiation.' },
      { w: 'implement', m: 'triển khai thực hiện', i: '/ˈɪm.plɪ.ment/', pos: 'v', s: 'The changes to the management system will be implemented next month.' },
      { w: 'revenue', m: 'doanh thu', i: '/ˈrev.ən.juː/', pos: 'n', s: 'The company reported a substantial increase in annual revenue.' },
      { w: 'strategy', m: 'chiến lược', i: '/ˈstræt.ə.dʒi/', pos: 'n', s: 'Their marketing strategy involves expanding directly into online channels.' },
      { w: 'agenda', m: 'chương trình nghị sự, lịch trình', i: '/əˈdʒen.də/', pos: 'n', s: 'There are several important items on today\'s meeting agenda.' },
      { w: 'collaborate', m: 'hợp tác, phối hợp', i: '/kəˈlæb.ə.reɪt/', pos: 'v', s: 'Two departments agreed to collaborate on the upcoming project.' },
      { w: 'inventory', m: 'hàng tồn kho, bản kiểm kê', i: '/ˈɪn.vən.tɔːr.i/', pos: 'n', s: 'The warehouse conducts a complete inventory check each quarter.' },
      { w: 'candidate', m: 'ứng viên ứng tuyển', i: '/ˈkæn.dɪ.dət/', pos: 'n', s: 'She proved to be the most qualified candidate for the position.' },
      { w: 'productive', m: 'có năng suất cao, hiệu quả', i: '/prəˈdʌk.tɪv/', pos: 'adj', s: 'Employees are significantly more productive in an ergonomic environment.' }
    ];
  } else if (type === 'oxford') {
    deckWords = [
      { w: 'absolute', m: 'tuyệt đối, hoàn toàn', i: '/ˈæb.sə.luːt/', pos: 'adj', s: 'I have absolute confidence in her leadership abilities.' },
      { w: 'campaign', m: 'chiến dịch', i: '/kæmˈpeɪn/', pos: 'n', s: 'The organization launched a nationwide awareness campaign.' },
      { w: 'essential', m: 'thiết yếu, cốt lõi', i: '/ɪˈsen.ʃəl/', pos: 'adj', s: 'Clean water is essential for sustaining all living organisms.' },
      { w: 'fascinating', m: 'hấp dẫn, lôi cuốn', i: '/ˈfæs.ən.eɪ.tɪŋ/', pos: 'adj', s: 'The historical documentary offers a fascinating glimpse of ancient culture.' },
      { w: 'guarantee', m: 'bảo đảm, cam kết', i: '/ˌɡær.ənˈtiː/', pos: 'v', s: 'The manufacturer guarantees the equipment against any defects for two years.' },
      { w: 'frequent', m: 'thường xuyên', i: '/ˈfriː.kwənt/', pos: 'adj', s: 'Buses offer frequent service between the city center and airport.' },
      { w: 'perspective', m: 'góc nhìn, quan điểm', i: '/pəˈspek.tɪv/', pos: 'n', s: 'Travelling abroad offers you a completely fresh perspective on life.' },
      { w: 'significant', m: 'đáng kể, quan trọng', i: '/sɪɡˈnɪf.ɪ.kənt/', pos: 'adj', s: 'There has been a significant improvement in student performance.' },
      { w: 'temporary', m: 'tạm thời', i: '/ˈtem.pər.ər.i/', pos: 'adj', s: 'The road closure is temporary while repairs are being finished.' },
      { w: 'vulnerable', m: 'dễ bị tổn thương', i: '/ˈvʌl.nər.ə.bəl/', pos: 'adj', s: 'Small businesses are particularly vulnerable during financial crises.' }
    ];
  }
  
  closePremadeModal();
  showToast(`⏳ Đang nạp ${deckWords.length} từ vựng bộ ${type.toUpperCase()}...`);
  
  let addedCount = 0;
  for (let i = 0; i < deckWords.length; i++) {
    const item = deckWords[i];
    try {
      const res = await fetch('/api/words/manual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          word: item.w,
          meaning_vi: item.m,
          ipa: item.i,
          pos: item.pos || 'word',
          example_sentence: item.s
        })
      });
      if (res.ok) addedCount++;
    } catch (e) { console.error(e); }
  }
  
  await loadWords();
  await loadStats();
  showToast(`🎉 Đã nạp thành công ${addedCount} từ vựng vào kho của bạn!`);
}
