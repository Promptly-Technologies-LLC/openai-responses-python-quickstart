// Audio recording and transcription
(function () {
  let mediaRecorder = null;
  let audioChunks = [];

  const MIC_SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>';
  const STOP_SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor" stroke="none"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>';

  function getMicButton() {
    return document.getElementById("micButton");
  }

  function getTextarea() {
    return document.getElementById("userInput");
  }

  async function startRecording() {
    const btn = getMicButton();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert(
        "Microphone access requires a secure context.\n" +
          "Please access this app via http://localhost:8000 instead of http://0.0.0.0:8000."
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.addEventListener("dataavailable", (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      });

      mediaRecorder.addEventListener("stop", async () => {
        // Stop all tracks so the browser releases the mic
        stream.getTracks().forEach((t) => t.stop());

        const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
        await transcribe(blob);
      });

      mediaRecorder.start();
      btn.classList.add("recording");
      btn.innerHTML = STOP_SVG;
      btn.title = "Stop recording";
    } catch (err) {
      console.error("Microphone access denied:", err);
      alert("Could not access your microphone. Please check permissions.");
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    const btn = getMicButton();
    btn.classList.remove("recording");
    btn.innerHTML = MIC_SVG;
    btn.title = "Record audio";
  }

  async function transcribe(blob) {
    const formData = new FormData();
    // Use webm extension; Whisper accepts it
    formData.append("audio", blob, "recording.webm");

    const textarea = getTextarea();
    const btn = getMicButton();
    const savedPlaceholder = textarea ? textarea.placeholder : "";

    // Show transcribing state
    if (textarea) {
      textarea.placeholder = "Transcribing...";
      textarea.disabled = true;
    }
    if (btn) btn.disabled = true;

    try {
      const res = await fetch("/audio/transcribe", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        console.error("Transcription failed:", res.status);
        alert("Transcription failed. Please try again.");
        return;
      }

      const text = await res.text();
      if (textarea) {
        // Append with a space if there's already text
        const current = textarea.value.trim();
        textarea.value = current ? current + " " + text : text;
        // Trigger resize
        textarea.style.height = "auto";
        textarea.style.height = textarea.scrollHeight + "px";
        textarea.focus();
      }
    } catch (err) {
      console.error("Transcription request error:", err);
      alert("Transcription failed. Please try again.");
    } finally {
      // Restore input state
      if (textarea) {
        textarea.placeholder = savedPlaceholder;
        textarea.disabled = false;
      }
      if (btn) btn.disabled = false;
    }
  }

  // Toggle recording on mic button click
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("#micButton");
    if (!btn) return;

    if (mediaRecorder && mediaRecorder.state === "recording") {
      stopRecording();
    } else {
      startRecording();
    }
  });
})();
