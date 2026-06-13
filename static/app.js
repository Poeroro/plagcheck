/* DoubleCheck client-side JS v2
   - Theme toggle (light/dark) with localStorage persistence
   - Mobile menu toggle
   - Drag-and-drop file UX
   - Riwayat search/filter/sort
   - Toast notifications
   - Download dropdown (results page)
*/

(function () {
  'use strict';

  // -----------------------------------------------------------------------
  // Theme toggle
  // -----------------------------------------------------------------------
  const THEME_KEY = 'pc_theme';
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem(THEME_KEY, theme); } catch (e) {}
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) {
      meta.setAttribute('content', theme === 'dark' ? '#0A0A0F' : '#4F46E5');
    }
  }
  function initTheme() {
    let stored = null;
    try { stored = localStorage.getItem(THEME_KEY); } catch (e) {}
    if (stored === 'dark' || stored === 'light') {
      applyTheme(stored);
      return;
    }
    // No preference stored — use system preference
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    applyTheme(prefersDark ? 'dark' : 'light');
  }
  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    applyTheme(current === 'dark' ? 'light' : 'dark');
  }
  document.addEventListener('DOMContentLoaded', function () {
    initTheme();
    document.querySelectorAll('#theme-toggle').forEach(function (btn) {
      btn.addEventListener('click', toggleTheme);
    });
  });

  // -----------------------------------------------------------------------
  // Mobile menu
  // -----------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', function () {
    const menuBtn = document.getElementById('menu-toggle');
    const nav = document.querySelector('.nav');
    if (menuBtn && nav) {
      menuBtn.addEventListener('click', function () {
        nav.classList.toggle('nav-mobile-open');
        menuBtn.setAttribute('aria-expanded', nav.classList.contains('nav-mobile-open'));
      });
    }
  });

  // -----------------------------------------------------------------------
  // Toast
  // -----------------------------------------------------------------------
  function toast(msg, kind) {
    const el = document.getElementById('toast');
    if (!el) return;
    el.textContent = msg;
    el.className = 'toast show' + (kind ? ' ' + kind : '');
    clearTimeout(toast._t);
    toast._t = setTimeout(function () {
      el.className = 'toast';
    }, 3000);
  }
  window.pcToast = toast;

  // -----------------------------------------------------------------------
  // Download dropdown (results page)
  // -----------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', function () {
    const toggle = document.getElementById('download-toggle');
    const menu = document.getElementById('download-menu');
    if (toggle && menu) {
      toggle.addEventListener('click', function (e) {
        e.stopPropagation();
        menu.classList.toggle('open');
      });
      document.addEventListener('click', function (e) {
        if (!menu.contains(e.target) && !toggle.contains(e.target)) {
          menu.classList.remove('open');
        }
      });
      document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') menu.classList.remove('open');
      });
    }
  });

  // -----------------------------------------------------------------------
  // Drag & drop + file picker
  // -----------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', function () {
    const dz = document.getElementById('dropzone');
    const input = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const fileRemove = document.getElementById('file-remove');
    if (!dz || !input) return;

    function bytesToHuman(n) {
      if (n < 1024) return n + ' B';
      if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
      return (n / 1024 / 1024).toFixed(2) + ' MB';
    }
    function setFile(f) {
      if (!f) return;
      const maxMB = 10;
      if (f.size > maxMB * 1024 * 1024) {
        toast('File terlalu besar (maks ' + maxMB + 'MB)', 'error');
        return;
      }
      const ext = (f.name.split('.').pop() || '').toLowerCase();
      if (!['pdf', 'docx', 'txt'].includes(ext)) {
        toast('Format tidak didukung: .' + ext, 'error');
        return;
      }
      // Build a new FileList-like via DataTransfer
      try {
        const dt = new DataTransfer();
        dt.items.add(f);
        input.files = dt.files;
      } catch (e) {
        toast('Browser tidak support file upload via drag', 'error');
        return;
      }
      fileName.textContent = f.name;
      fileSize.textContent = bytesToHuman(f.size) + ' · .' + ext.toUpperCase();
      fileInfo.style.display = 'flex';
      dz.classList.add('has-file');
    }
    function clearFile() {
      input.value = '';
      fileInfo.style.display = 'none';
      dz.classList.remove('has-file');
    }

    input.addEventListener('change', function () {
      if (input.files && input.files[0]) setFile(input.files[0]);
    });
    if (fileRemove) fileRemove.addEventListener('click', function (e) { e.stopPropagation(); clearFile(); });

    // Drag and drop
    ['dragenter', 'dragover'].forEach(function (ev) {
      dz.addEventListener(ev, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dz.classList.add('drag-over');
      });
    });
    ['dragleave', 'drop'].forEach(function (ev) {
      dz.addEventListener(ev, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dz.classList.remove('drag-over');
      });
    });
    dz.addEventListener('drop', function (e) {
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) setFile(f);
    });
  });

  // -----------------------------------------------------------------------
  // Mode presets
  // -----------------------------------------------------------------------
  const MODE_PRESETS = {
    cepat:   { semantic: false, ce: false, ensemble: false, ai: false, cite: true },
    standar: { semantic: true,  ce: true,  ensemble: false, ai: true,  cite: true },
    akurat:  { semantic: true,  ce: true,  ensemble: true,  ai: true,  cite: true }
  };
  document.addEventListener('DOMContentLoaded', function () {
    const modeBtns = document.querySelectorAll('.config-mode-btn');
    const optSemantic = document.getElementById('opt-semantic');
    const optCe = document.getElementById('opt-ce');
    const optEnsemble = document.getElementById('opt-ensemble');
    const optAi = document.getElementById('opt-ai');
    const optCite = document.getElementById('opt-cite');
    if (!modeBtns.length || !optSemantic) return;

    function applyPreset(name) {
      const p = MODE_PRESETS[name] || MODE_PRESETS.standar;
      optSemantic.checked = p.semantic;
      optCe.checked = p.ce;
      if (optEnsemble) optEnsemble.checked = p.ensemble;
      optAi.checked = p.ai;
      optCite.checked = p.cite;
    }
    modeBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        modeBtns.forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');
        applyPreset(btn.dataset.mode);
      });
    });
  });

  // -----------------------------------------------------------------------
  // Submit + loading state
  // -----------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('check-form');
    if (!form) return;
    form.addEventListener('submit', async function (e) {
      e.preventDefault();
      const submitBtn = document.getElementById('submit-btn');
      const loading = document.getElementById('loading');
      const fileInput = document.getElementById('file-input');
      if (!fileInput.files || !fileInput.files[0]) {
        toast('Pilih file dulu', 'error');
        return;
      }
      submitBtn.disabled = true;
      loading.style.display = 'grid';
      const steps = document.querySelectorAll('.loading-step');
      steps.forEach(function (s) { s.classList.remove('active', 'done'); });

      // Step 0: parse
      steps[0].classList.add('active');
      await new Promise(function (r) { setTimeout(r, 200); });

      const fd = new FormData(form);
      let stepIdx = 0;
      const stepTimer = setInterval(function () {
        stepIdx++;
        if (stepIdx < steps.length) {
          steps[stepIdx - 1].classList.remove('active');
          steps[stepIdx - 1].classList.add('done');
          if (stepIdx < steps.length) steps[stepIdx].classList.add('active');
        }
      }, 4000);

      try {
        const r = await fetch('/api/check', { method: 'POST', body: fd });
        const data = await r.json();
        clearInterval(stepTimer);
        steps.forEach(function (s) { s.classList.remove('active'); s.classList.add('done'); });
        if (!r.ok) throw new Error(data.detail || 'Gagal');
        await new Promise(function (res) { setTimeout(res, 300); });
        window.location.href = '/r/' + data.id;
      } catch (err) {
        clearInterval(stepTimer);
        loading.style.display = 'none';
        submitBtn.disabled = false;
        toast('Error: ' + err.message, 'error');
      }
    });
  });

  // -----------------------------------------------------------------------
  // Riwayat search/filter/sort
  // -----------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', function () {
    const list = document.getElementById('riwayat-list');
    if (!list) return;
    const items = Array.from(list.querySelectorAll('.recent-item'));
    const search = document.getElementById('search');
    const filters = document.querySelectorAll('.filter-chip[data-filter]');
    const sortSel = document.getElementById('sort');
    const emptyEl = document.getElementById('empty-search');
    const totalCount = document.getElementById('total-count');
    let currentFilter = 'all';
    let currentSearch = '';
    let currentSort = 'newest';

    function applyFilters() {
      let visible = 0;
      items.forEach(function (item) {
        const name = item.dataset.name || '';
        const score = parseFloat(item.dataset.score || '0');
        const scoreClass = item.dataset.scoreClass || 'low';
        const ai = item.dataset.ai || '';
        let show = true;
        if (currentSearch && name.indexOf(currentSearch) === -1) show = false;
        if (show && currentFilter !== 'all') {
          if (currentFilter === 'low' && scoreClass !== 'low') show = false;
          if (currentFilter === 'med' && scoreClass !== 'med') show = false;
          if (currentFilter === 'high' && scoreClass !== 'high') show = false;
          if (currentFilter === 'ai' && !ai) show = false;
        }
        item.style.display = show ? '' : 'none';
        if (show) visible++;
      });
      if (emptyEl) emptyEl.classList.toggle('show', visible === 0);
      if (totalCount) {
        totalCount.textContent = visible + ' dari ' + items.length + ' dokumen';
      }
    }
    function applySort() {
      const sorted = items.slice().sort(function (a, b) {
        const sa = parseFloat(a.dataset.score || '0');
        const sb = parseFloat(b.dataset.score || '0');
        const ta = parseInt(a.dataset.timestamp || '0', 10);
        const tb = parseInt(b.dataset.timestamp || '0', 10);
        if (currentSort === 'highest') return sb - sa;
        if (currentSort === 'lowest') return sa - sb;
        if (currentSort === 'oldest') return ta - tb;
        return tb - ta; // newest
      });
      sorted.forEach(function (item) { list.appendChild(item); });
    }

    if (search) {
      search.addEventListener('input', function () {
        currentSearch = search.value.trim().toLowerCase();
        applyFilters();
      });
    }
    filters.forEach(function (chip) {
      chip.addEventListener('click', function () {
        filters.forEach(function (c) { c.classList.remove('active'); });
        chip.classList.add('active');
        currentFilter = chip.dataset.filter;
        applyFilters();
      });
    });
    if (sortSel) {
      sortSel.addEventListener('change', function () {
        currentSort = sortSel.value;
        applySort();
        applyFilters();
      });
    }
  });
})();
