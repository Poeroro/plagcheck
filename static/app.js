/* ==========================================================================
   PlagCheck — Client-side interactions
   ========================================================================== */

(function() {
  'use strict';

  // -------------------------------------------------------------------
  // Drag & drop + file input
  // -------------------------------------------------------------------
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const fileInfo = document.getElementById('file-info');
  const dropTitle = document.getElementById('drop-title');
  const dropSub = document.getElementById('drop-sub');

  if (dropzone && fileInput) {
    ['dragenter', 'dragover'].forEach(evt => {
      dropzone.addEventListener(evt, e => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(evt => {
      dropzone.addEventListener(evt, e => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      });
    });
    dropzone.addEventListener('drop', e => {
      const files = e.dataTransfer.files;
      if (files && files.length) {
        fileInput.files = files;
        showFileInfo(files[0]);
      }
    });
    fileInput.addEventListener('change', e => {
      if (e.target.files && e.target.files.length) {
        showFileInfo(e.target.files[0]);
      }
    });

    function showFileInfo(file) {
      const ext = (file.name.split('.').pop() || '').toLowerCase();
      const size = (file.size / 1024).toFixed(1) + ' KB';
      fileInfo.style.display = 'flex';
      fileInfo.innerHTML = `
        <div class="fi-icon ${ext}">${ext.toUpperCase()}</div>
        <div class="fi-name">${escapeHtml(file.name)}</div>
        <div class="fi-size">${size}</div>
        <button type="button" class="fi-remove" title="Hapus">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      `;
      fileInfo.querySelector('.fi-remove').addEventListener('click', () => {
        fileInput.value = '';
        fileInfo.style.display = 'none';
        dropzone.classList.remove('has-file');
        if (dropTitle) dropTitle.textContent = 'Drag & drop dokumen di sini';
        if (dropSub) dropSub.textContent = 'atau klik tombol di bawah untuk pilih file · maks 10MB';
      });
      dropzone.classList.add('has-file');
      if (dropTitle) dropTitle.textContent = 'File siap dicek';
      if (dropSub) dropSub.textContent = 'Klik tombol di bawah untuk mulai';
    }
  }

  // -------------------------------------------------------------------
  // Mode selector (Cepat / Standar / Akurat) → toggle options
  // -------------------------------------------------------------------
  const modeButtons = document.querySelectorAll('.config-mode-btn');
  const optSemantic = document.getElementById('opt-semantic');
  const optCe = document.getElementById('opt-ce');
  const optEnsemble = document.getElementById('opt-ensemble');
  const optAi = document.getElementById('opt-ai');

  if (modeButtons.length && optSemantic) {
    const presets = {
      cepat:   { semantic: true,  ce: false, ensemble: false, ai: false },
      standar: { semantic: true,  ce: true,  ensemble: false, ai: true  },
      akurat:  { semantic: true,  ce: true,  ensemble: true,  ai: true  }
    };
    modeButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        modeButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const mode = btn.dataset.mode;
        const p = presets[mode];
        if (p) {
          optSemantic.checked = p.semantic;
          optCe.checked = p.ce;
          optEnsemble.checked = p.ensemble;
          optAi.checked = p.ai;
        }
        updateSubmitHint();
      });
    });
  }

  // -------------------------------------------------------------------
  // Update submit hint based on selected options
  // -------------------------------------------------------------------
  function updateSubmitHint() {
    const hint = document.getElementById('submit-hint');
    if (!hint) return;
    let seconds = 1.0;  // base parse
    if (optSemantic && optSemantic.checked) seconds += 8.4;
    if (optCe && optCe.checked) seconds += 11.4;
    if (optEnsemble && optEnsemble.checked) seconds += 12.0;
    if (optAi && optAi.checked) seconds += 3.2;
    hint.textContent = `Estimasi: ~${seconds.toFixed(1)} detik`;
  }

  document.querySelectorAll('.config-row .toggle').forEach(t => {
    t.addEventListener('change', updateSubmitHint);
  });

  // -------------------------------------------------------------------
  // Form submission with loading state
  // -------------------------------------------------------------------
  const form = document.getElementById('check-form');
  if (form) {
    form.addEventListener('submit', async e => {
      e.preventDefault();

      if (!fileInput.files || !fileInput.files.length) {
        alert('Pilih file dulu ya');
        return;
      }

      const loading = document.getElementById('loading');
      const submitBtn = document.getElementById('submit-btn');
      if (loading) loading.style.display = 'grid';
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span>Memproses...</span>';
      }

      // Animate steps
      const steps = ['parse', 'semantic', 'ce', 'ai', 'report'];
      let stepIdx = 0;
      const advanceStep = () => {
        if (stepIdx >= steps.length) return;
        const stepEl = document.querySelector(`.loading-step[data-step="${steps[stepIdx]}"]`);
        if (stepEl) {
          if (stepIdx > 0) {
            const prevEl = document.querySelector(`.loading-step[data-step="${steps[stepIdx-1]}"]`);
            if (prevEl) {
              prevEl.classList.remove('active');
              prevEl.classList.add('done');
            }
          }
          stepEl.classList.add('active');
        }
        stepIdx++;
        if (stepIdx < steps.length) setTimeout(advanceStep, 1500);
      };
      advanceStep();

      const formData = new FormData(form);
      try {
        const response = await fetch('/api/check', {
          method: 'POST',
          body: formData
        });
        if (!response.ok) {
          const err = await response.text();
          throw new Error(err || 'Check failed');
        }
        const result = await response.json();
        // Redirect to results page
        if (result.id) {
          window.location.href = `/r/${result.id}`;
        } else {
          // Fallback: re-render with result data
          window.location.href = '/r?data=' + encodeURIComponent(JSON.stringify(result));
        }
      } catch (err) {
        alert('Error: ' + err.message);
        if (loading) loading.style.display = 'none';
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Mulai Pengecekan';
        }
      }
    });
  }

  // -------------------------------------------------------------------
  // Tab switching (results page)
  // -------------------------------------------------------------------
  document.querySelectorAll('.doc-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      document.querySelectorAll('.doc-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      document.querySelectorAll('[data-pane]').forEach(p => {
        p.style.display = p.dataset.pane === target ? 'block' : 'none';
      });
    });
  });

  // -------------------------------------------------------------------
  // Match item click → highlight paragraph
  // -------------------------------------------------------------------
  document.querySelectorAll('.match-item[data-idx]').forEach(item => {
    item.addEventListener('click', () => {
      const idx = item.dataset.idx;
      document.querySelectorAll('.match-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      const para = document.querySelector(`.match-paragraph[data-match-idx="${idx}"]`);
      if (para) {
        para.scrollIntoView({behavior: 'smooth', block: 'center'});
        para.classList.add('flash');
        setTimeout(() => para.classList.remove('flash'), 1500);
      }
    });
  });

  // -------------------------------------------------------------------
  // Helper
  // -------------------------------------------------------------------
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
})();
