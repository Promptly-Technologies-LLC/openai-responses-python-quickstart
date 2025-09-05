// A global map to hold the growing markdown string per target container node
window._streamingMarkdown = new WeakMap();

// Main SSE event handler: called by HTMX for events listed in sse-swap if not handled by default HTMX swap
// or for all events if hx-on:htmx:sse-before-message is used.
function handleCustomSseEvents(evt) {
	const originalSSEEvent = evt.detail; // This is {data: "...", type: "...", lastEventId: "..."}
	// const sendButton = document.getElementById('sendButton'); // sendButton not directly needed here anymore for endStream

	// Explicitly check for endStream
	if (originalSSEEvent.type === 'endStream') {
        console.log("handleCustomSseEvents: Saw endStream event. Data:", originalSSEEvent.data, ". HTMX will handle sse-close.");
		// This event is primarily for sse-close. No special action (like re-enabling button) needed here.
		// HTMX will see the 'endStream' event name based on sse-close attribute and then dispatch 'htmx:sseClose'.
		return; // Return as endStream is not for typical sse-swap content handling within this function.
	}

	// For other events that are supposed to carry data for swapping:
	if (!originalSSEEvent || !originalSSEEvent.data) {
		// This condition should only be met if an event *meant* for swapping (like textDelta) is missing data.
		console.warn(`SSE event type '${originalSSEEvent.type}' (expected data for swap) without data or unexpected structure:`, originalSSEEvent);
		return;
	} else if (originalSSEEvent.type === 'textReplacement') {
		evt.preventDefault();
		processTextReplacement(originalSSEEvent);
	}

	// Prevent default HTMX swap for specific events we handle in JS
	if (originalSSEEvent.type === 'textDelta') {
		evt.preventDefault();
		processTextDelta(originalSSEEvent);
	}
	// Other event types (messageCreated, toolCallCreated, etc.) listed in sse-swap will be handled by HTMX default swap
	// if they are listed in sse-swap and not prevented here by evt.preventDefault().
}

function processTextDelta(sseEvent) {
	const oobHTML = sseEvent.data;
	const { targetElement, markdownChunk } = parseOobSwap(oobHTML, "textDelta");

	if (!targetElement || markdownChunk === null) { // markdownChunk can be empty string
		return;
	}
	if (!markdownChunk && markdownChunk !== '') { // Explicitly allow empty string chunk
		return;
	}

	if (!window._streamingMarkdown) {
		window._streamingMarkdown = new WeakMap();
	}

	const prev = window._streamingMarkdown.get(targetElement) || '';
	let updatedMarkdown = prev + markdownChunk;

	window._streamingMarkdown.set(targetElement, updatedMarkdown);
	renderMarkdown(targetElement, updatedMarkdown, markdownChunk); // Pass original chunk for fallback
}

function processTextReplacement(sseEvent) {
	const oobHTML = sseEvent.data;
	const { targetElement, payload } = parseOobSwap(oobHTML, "textReplacement");

	if (!targetElement || !payload) {
		return;
	}

	const parts = payload.split('|');
	if (parts.length !== 2) {
		console.error("Invalid payload for textReplacement:", payload);
		return;
	}
	const textToReplace = parts[0]; // This is "sandbox:/path/to/file"
	const replacementText = parts[1]; // This is the actual download URL

	if (!window._streamingMarkdown) {
		// Should have been initialized by processTextDelta
		window._streamingMarkdown = new WeakMap();
	}

	let currentMarkdown = window._streamingMarkdown.get(targetElement) || '';

	// Build regex from pattern where '*' is a wildcard within the markdown link URL
	const regex = new RegExp(`\\(\\s*${escapeRegExp(textToReplace)}\\s*\\)`, 'g');

	if (regex.test(currentMarkdown)) {
		currentMarkdown = currentMarkdown.replace(regex, `(${replacementText})`);
		console.log(`Applied replacement: ${textToReplace} -> ${replacementText}`);
		window._streamingMarkdown.set(targetElement, currentMarkdown);
		renderMarkdown(targetElement, currentMarkdown, ''); // Re-render the whole thing
	} else {
		console.warn(`Text to replace '${textToReplace}' (in markdown link format) not found in current markdown. Markdown state:`, currentMarkdown.substring(0, 300));
        // If synchronous handling is guaranteed, this path implies an issue or mismatch.
	}
}

// Helper to parse OOB swap HTML and extract target and content
function parseOobSwap(oobHTML, eventTypeForLogging) {
	const parser = new DOMParser();
	const doc = parser.parseFromString(oobHTML, 'text/html');
	const oobElement = doc.body.firstChild;

	if (!oobElement || !oobElement.getAttribute || oobElement.nodeType !== Node.ELEMENT_NODE) {
		console.error(`Could not parse OOB element from ${eventTypeForLogging} SSE data:`, oobHTML);
		return { targetElement: null, payload: null, markdownChunk: null };
	}

	const swapOobAttr = oobElement.getAttribute('hx-swap-oob');
	const content = oobElement.innerHTML; // For textDelta, this is markdownChunk

	if (!swapOobAttr) {
		console.warn(`${eventTypeForLogging} message did not contain hx-swap-oob:`, oobHTML);
		return { targetElement: null, payload: null, markdownChunk: null };
	}

	let targetSelector = swapOobAttr;
	const colonIndex = swapOobAttr.indexOf(':');
	if (colonIndex !== -1) {
		targetSelector = swapOobAttr.substring(colonIndex + 1);
	}

	const targetElement = document.querySelector(targetSelector);
	if (!targetElement) {
		console.warn(`Target element for OOB swap not found (${eventTypeForLogging}):`, targetSelector);
		return { targetElement: null, payload: content, markdownChunk: content }; // Return content for caller to check
	}
	// Depending on context, content is either markdownChunk or payload
	return { targetElement, payload: content, markdownChunk: content };
}

// Helper to escape string for use in RegExp
function escapeRegExp(string) {
	return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

// Extracted rendering logic
function renderMarkdown(targetElement, markdownToRender, fallbackChunkOnError) {
	window._streamingMarkdown.set(targetElement, markdownToRender); 

	if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
		console.error("marked.js or DOMPurify not loaded.");
		targetElement.textContent = window._streamingMarkdown.get(targetElement) || fallbackChunkOnError;
		return;
	}
	try {
		const rawHtml = marked.parse(markdownToRender);
		const sanitizedHtml = DOMPurify.sanitize(rawHtml, {
			USE_PROFILES: { html: true },
		});
		targetElement.innerHTML = sanitizedHtml;

		// Ensure links open in a new tab without altering hrefs
		try {
			const anchors = targetElement.querySelectorAll('a[href]');
			anchors.forEach((a) => {
				a.setAttribute('target', '_blank');
				a.setAttribute('rel', 'noopener noreferrer');
			});
		} catch (linkErr) {
			console.warn('Error post-processing links:', linkErr);
		}

		const messagesContainer = document.getElementById('messages');
		if (messagesContainer) {
			const isScrolledToBottom = messagesContainer.scrollHeight - messagesContainer.clientHeight <= messagesContainer.scrollTop + 10; 
			if (isScrolledToBottom) {
				messagesContainer.scrollTop = messagesContainer.scrollHeight;
			}
		}
	} catch (e) {
		console.error("Error processing markdown:", e);
		targetElement.textContent = (window._streamingMarkdown.get(targetElement) || '') + (fallbackChunkOnError || '');
	}
}

// Global functions for button state management
window.disableSendButton = function() {
    const sendButton = document.getElementById('sendButton');
    if (sendButton) {
        sendButton.disabled = true;
        sendButton.querySelector('.button__text').style.display = 'none';
        sendButton.querySelector('.button__loader').style.display = 'inline-block';
        console.log("disableSendButton: Send button disabled and loader shown.");
    }
};

window.reEnableSendButton = function() {
    const sendButton = document.getElementById('sendButton');
    if (sendButton && sendButton.disabled) {
        sendButton.disabled = false;
        sendButton.querySelector('.button__text').style.display = 'inline-block';
        sendButton.querySelector('.button__loader').style.display = 'none';
        console.log("reEnableSendButton: Send button re-enabled and text restored, loader hidden.");
    }
};

// Simplified DOMContentLoaded - no complex event listeners needed
document.addEventListener('DOMContentLoaded', () => {
    console.log("stream-md.js: DOMContentLoaded - global button functions available.");
});