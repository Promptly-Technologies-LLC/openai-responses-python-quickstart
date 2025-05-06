// A global map to hold the growing markdown string per target container node
window._streamingMarkdown = new WeakMap();

function handleSSETextDelta(evt) {
  // Check if this event is for a 'textDelta' message fired by sse.js
  const originalSSEEvent = evt.detail;
  if (!originalSSEEvent || originalSSEEvent.type !== 'textDelta') {
      return;
  }

  // Prevent the default HTMX swap for this specific message
  // sse.js triggers this *before* calling api.swap()
  evt.preventDefault();

  // The data contains the OOB swap HTML: <span hx-swap-oob="beforeend:#step-...">CHUNK</span>
  const oobHTML = originalSSEEvent.data;

  // Use DOMParser to safely extract the target selector and the markdown chunk
  const parser = new DOMParser();
  const doc = parser.parseFromString(oobHTML, 'text/html');
  const oobElement = doc.body.firstChild;

  if (!oobElement || !oobElement.getAttribute || oobElement.nodeType !== Node.ELEMENT_NODE) {
      console.error("Could not parse OOB element from SSE data:", oobHTML);
      return;
  }

  const swapOobAttr = oobElement.getAttribute('hx-swap-oob');
  const markdownChunk = oobElement.innerHTML || '';

  if (!swapOobAttr) {
       // Might be a non-OOB textDelta, handle differently or ignore?
       // For now, let's assume textDelta is always OOB for steps.
      console.warn("textDelta message did not contain hx-swap-oob:", oobHTML);
      return;
  }
   if (!markdownChunk) {
      // Empty chunk, nothing to render
      return;
  }

  // Extract the target selector (e.g., "beforeend:#step-...") -> "#step-..."
  let targetSelector = swapOobAttr;
  const colonIndex = swapOobAttr.indexOf(':');
  if (colonIndex !== -1) {
      targetSelector = swapOobAttr.substring(colonIndex + 1);
  }

  // Find the actual target element where content should be rendered
  const targetElement = document.querySelector(targetSelector);
  if (!targetElement) {
      console.warn("Target element for OOB swap not found:", targetSelector);
      return;
  }

  // Use a WeakMap keyed by the *actual target* element
  if (!window._streamingMarkdown) {
      window._streamingMarkdown = new WeakMap();
  }

  // 1) Accumulate markdown for the specific target
  const prev = window._streamingMarkdown.get(targetElement) || '';
  const updatedMarkdown = prev + markdownChunk;
  window._streamingMarkdown.set(targetElement, updatedMarkdown);

  // 2) Re-render -> sanitize -> swap into the target
  if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
      console.error("marked.js or DOMPurify not loaded.");
      // Fallback to raw text
      targetElement.textContent += markdownChunk;
      return;
  }

  try {
    // Create custom renderer with link target
    const renderer = new marked.Renderer();
    // Override link renderer: destructure the token object for href and text
    renderer.link = ({ href, title, text }) => {
      const titleAttr = title ? ` title="${title}"` : '';
      return `<a target="_blank" rel="noopener noreferrer" href="${href}"${titleAttr}>${text}</a>`;
    };

    // Use marked.parse() with custom renderer
    const rawHtml = marked.parse(updatedMarkdown, { renderer });
    // Configure DOMPurify
    const sanitizedHtml = DOMPurify.sanitize(rawHtml, {
        // Allows standard HTML elements
        USE_PROFILES: { html: true }
      });
    targetElement.innerHTML = sanitizedHtml;

    // 3) Auto-scroll the main messages container
    const messagesContainer = document.getElementById('messages');
    if (messagesContainer) {
        // Scroll only if the user isn't intentionally scrolled up
        const isScrolledToBottom = messagesContainer.scrollHeight - messagesContainer.clientHeight <= messagesContainer.scrollTop + 1;
        if(isScrolledToBottom) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }
  } catch (e) {
    console.error("Error processing markdown:", e);
    // Fallback on error: append raw chunk to existing text content
    targetElement.innerHTML = (targetElement.innerHTML || '') + markdownChunk;
  }
}