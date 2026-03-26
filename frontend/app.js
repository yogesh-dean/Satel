/* ============================================================
   app.js  –  Frontend logic for SatSeg
   Connects to FastAPI backend at http://localhost:8000
   ============================================================ */

const API_BASE = 'https://satseg-backend.onrender.com';

// ── Class config (colours must match image_processor.py BGR→RGB swap) ──────────
const CLASS_CONFIG = [
  { name: 'Background',       color: '#ffffff', icon: '🏜️' },
  { name: 'Water',            color: '#0000ff', icon: '💧' },
  { name: 'Dense Vegetation', color: '#00ff00', icon: '🌳' },
  { name: 'Light Vegetation', color: '#ffff00', icon: '🌿' },
  { name: 'Dry Land',         color: '#ff00ff', icon: '🏔️' },
  { name: 'Urban',            color: '#00ffff', icon: '🏙️' },
];

// ── State ──────────────────────────────────────────────────────────────────────
let selectedFile = null;

// ── DOM refs ───────────────────────────────────────────────────────────────────
const uploadZone    = document.getElementById('uploadZone');
const fileInput     = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImg    = document.getElementById('previewImg');
const previewFilename = document.getElementById('previewFilename');
const loaderSection = document.getElementById('loaderSection');
const loaderStep    = document.getElementById('loaderStep');
const resultsSection = document.getElementById('resultsSection');

// ── Drag & Drop ────────────────────────────────────────────────────────────────
uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelected(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFileSelected(fileInput.files[0]);
});

// ── File Selected ──────────────────────────────────────────────────────────────
function handleFileSelected(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src    = e.target.result;
    previewFilename.textContent = file.name;
    previewSection.classList.add('visible');
    resultsSection.classList.remove('visible');
    loaderSection.classList.remove('visible');

    // Smooth scroll to preview
    setTimeout(() => previewSection.scrollIntoView({ behavior: 'smooth', block: 'center' }), 100);
  };
  reader.readAsDataURL(file);
}

// ── Run Analysis ───────────────────────────────────────────────────────────────
async function runAnalysis() {
  if (!selectedFile) return;

  // Show loader
  previewSection.classList.remove('visible');
  loaderSection.classList.add('visible');
  resultsSection.classList.remove('visible');
  loaderSection.scrollIntoView({ behavior: 'smooth', block: 'center' });

  const steps = [
    'Uploading image to server…',
    'Running U-Net inference…',
    'Generating segmentation mask…',
    'Computing vulnerability metrics…',
    'Finalising results…',
  ];
  let stepIdx = 0;
  const stepInterval = setInterval(() => {
    if (stepIdx < steps.length) {
      loaderStep.textContent = steps[stepIdx++];
    }
  }, 900);

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);

    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      body: formData,
    });

    clearInterval(stepInterval);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown server error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);

  } catch (err) {
    clearInterval(stepInterval);
    loaderSection.classList.remove('visible');
    previewSection.classList.add('visible');
    showError(`Analysis failed: ${err.message}. Make sure the backend server is running at ${API_BASE}`);
  }
}

// ── Render Results ─────────────────────────────────────────────────────────────
function renderResults(data) {
  loaderSection.classList.remove('visible');

  // Original image
  document.getElementById('origImg').src = previewImg.src;

  // Segmented mask
  document.getElementById('maskImg').src = data.segmentation_mask_base64;

  // Class bars
  renderClassBars(data.class_percentages);

  // Vulnerability cards
  renderVulnCards(data.vulnerability);

  // Final banner
  renderFinalBanner(data.vulnerability.final_status);

  resultsSection.classList.add('visible');
  setTimeout(() => resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);

  // Animate bars after short delay so CSS transition works
  setTimeout(() => animateBars(data.class_percentages), 200);
}

// ── Class Bars ─────────────────────────────────────────────────────────────────
function renderClassBars(percentages) {
  const container = document.getElementById('classBars');
  container.innerHTML = '';

  CLASS_CONFIG.forEach(cls => {
    const pct = percentages[cls.name] ?? 0;
    const div = document.createElement('div');
    div.className  = 'class-bar';
    div.dataset.name = cls.name;
    div.innerHTML = `
      <div class="class-bar-header">
        <span class="class-name">
          <span class="class-dot" style="background:${cls.color};box-shadow:0 0 6px ${cls.color};"></span>
          ${cls.icon} ${cls.name}
        </span>
        <span class="class-pct">${pct.toFixed(1)}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" data-target="${pct}" style="width:0%;"></div>
      </div>`;
    container.appendChild(div);
  });
}

function animateBars(percentages) {
  document.querySelectorAll('.bar-fill').forEach(fill => {
    const target = parseFloat(fill.dataset.target);
    fill.style.width = Math.min(target, 100) + '%';
  });
}

// ── Vulnerability Cards ────────────────────────────────────────────────────────
function renderVulnCards(vuln) {
  const grid = document.getElementById('vulnGrid');
  grid.innerHTML = '';

  const cards = [
    {
      cls:    'vc-drought',
      icon:   '🌵',
      title:  'Drought Risk',
      level:  vuln.drought_risk,
      detail: vuln.drought_detail,
    },
    {
      cls:    'vc-flood',
      icon:   '🌊',
      title:  'Flood Risk',
      level:  vuln.flood_risk,
      detail: vuln.flood_detail,
    },
    {
      cls:    'vc-eco',
      icon:   '🌱',
      title:  'Ecosystem Health',
      level:  vuln.ecosystem_health,   // HEALTHY / MODERATE / POOR
      detail: vuln.ecosystem_detail,
    },
    {
      cls:    'vc-urban',
      icon:   '🏙️',
      title:  'Urban Stress',
      level:  vuln.urban_risk,
      detail: vuln.urban_detail,
    },
  ];

  cards.forEach((c, i) => {
    const div = document.createElement('div');
    div.className = `vuln-card ${c.cls}`;
    div.style.animationDelay = `${i * 0.1}s`;
    div.style.animation = 'fadeUp 0.5s ease both';
    div.innerHTML = `
      <div class="vuln-card-icon">${c.icon}</div>
      <div class="vuln-card-title">${c.title}</div>
      <span class="level-badge level-${c.level}">${c.level}</span>
      <div class="vuln-card-detail">${c.detail}</div>`;
    grid.appendChild(div);
  });

  // Key metrics row
  const km = vuln.key_metrics;
  const metricsDiv = document.createElement('div');
  metricsDiv.style.cssText = 'grid-column:1/-1;display:flex;flex-wrap:wrap;gap:12px;justify-content:center;';
  [
    { label:'Water',      val: km.water_pct,    icon:'💧' },
    { label:'Vegetation', val: km.veg_total_pct, icon:'🌿' },
    { label:'Dry Land',   val: km.dry_land_pct, icon:'🏜️' },
    { label:'Urban',      val: km.urban_pct,    icon:'🏙️' },
  ].forEach(m => {
    const span = document.createElement('div');
    span.style.cssText = 'background:var(--surface);border:1px solid var(--border);border-radius:99px;padding:6px 18px;font-size:0.85rem;font-weight:600;';
    span.textContent = `${m.icon}  ${m.label}: ${m.val.toFixed(1)}%`;
    metricsDiv.appendChild(span);
  });
  grid.appendChild(metricsDiv);
}

// ── Final Banner ───────────────────────────────────────────────────────────────
function renderFinalBanner(finalStatus) {
  const banner = document.getElementById('finalBanner');
  const statusEl = document.getElementById('bannerStatus');

  banner.className = 'final-banner';
  statusEl.textContent = finalStatus;

  if (finalStatus.includes('DROUGHT'))       banner.classList.add('banner-drought');
  else if (finalStatus.includes('FLOOD'))    banner.classList.add('banner-flood');
  else if (finalStatus.includes('STABLE') || finalStatus.includes('LOW'))
                                             banner.classList.add('banner-healthy');
  else if (finalStatus.includes('URBAN'))    banner.classList.add('banner-urban');
  else                                       banner.classList.add('banner-moderate');
}

// ── Reset ──────────────────────────────────────────────────────────────────────
function resetAll() {
  selectedFile = null;
  fileInput.value = '';
  previewSection.classList.remove('visible');
  loaderSection.classList.remove('visible');
  resultsSection.classList.remove('visible');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Error Toast ────────────────────────────────────────────────────────────────
function showError(msg) {
  const toast = document.createElement('div');
  toast.style.cssText = `
    position:fixed; bottom:28px; left:50%; transform:translateX(-50%);
    background:#1e0a0a; border:1px solid rgba(239,68,68,0.5);
    color:#fca5a5; padding:14px 28px; border-radius:12px;
    font-size:0.88rem; font-weight:500; z-index:9999;
    box-shadow:0 8px 32px rgba(0,0,0,0.5);
    animation:fadeUp 0.3s ease;
    max-width:90vw; text-align:center;
  `;
  toast.textContent = '⚠️  ' + msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 6000);
}
