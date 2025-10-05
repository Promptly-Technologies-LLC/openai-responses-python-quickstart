// Collapsible toggle for toolCall steps
// Expanded by default; toggles via header click or Enter/Space key
(function() {
  function toggleCollapsible(header) {
    try {
      var container = header.closest('.collapsible');
      if (!container) return;
      var targetId = header.getAttribute('data-target');
      var content = targetId ? document.getElementById(targetId) : container.querySelector('.collapsible__content');
      if (!content) return;

      var expanded = header.getAttribute('aria-expanded') === 'true';
      var nextExpanded = !expanded;
      header.setAttribute('aria-expanded', String(nextExpanded));
      if (nextExpanded) {
        container.classList.remove('collapsible--collapsed');
        content.removeAttribute('hidden');
      } else {
        container.classList.add('collapsible--collapsed');
        content.setAttribute('hidden', '');
      }
    } catch (e) {
      console.warn('toggleCollapsible error', e);
    }
  }

  function initHeader(header) {
    if (header.dataset.collapsibleInit === '1') return;
    header.dataset.collapsibleInit = '1';
    header.addEventListener('click', function() { toggleCollapsible(header); });
    header.addEventListener('keydown', function(ev) {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        toggleCollapsible(header);
      }
    });
  }

  function initAll() {
    var headers = document.querySelectorAll('.collapsible .collapsible__header[aria-expanded]');
    headers.forEach(initHeader);
  }

  // Initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }

  // Re-initialize after HTMX processing (handles SSE-inserted nodes)
  document.body.addEventListener('htmx:afterSwap', initAll);
  document.body.addEventListener('htmx:afterProcessNode', initAll);
})();



