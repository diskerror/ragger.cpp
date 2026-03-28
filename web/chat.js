/**
 * Ragger Chat — Client
 *
 * Connects to the /chat endpoint via POST + SSE (EventSource pattern).
 * Variable names and endpoint paths are stable for custom UI builds.
 *
 * Key variables:
 *   CHAT_ENDPOINT  — POST target for sending messages
 *   AUTH_TOKEN      — Bearer token (read from cookie or set manually)
 *   sessionId       — Server-assigned session ID (persists across messages)
 */

(function () {
    "use strict";

    // --- Configuration ---
    const CHAT_ENDPOINT = "/chat";
    const AUTH_ENDPOINT = "/auth/login";  // future: password login
    let AUTH_TOKEN = "";  // populated from cookie or login
    let sessionId = "";

    // --- DOM ---
    const form     = document.getElementById("chat-form");
    const input    = document.getElementById("input");
    const messages = document.getElementById("messages");
    const sendBtn  = document.getElementById("send-btn");
    const status   = document.getElementById("status");

    let streaming = false;

    // --- Auth ---
    // Read token from cookie or URL parameter
    function getToken() {
        // Check URL parameter first (?token=...)
        const params = new URLSearchParams(window.location.search);
        const urlToken = params.get("token");
        if (urlToken) {
            // Store in cookie for subsequent requests, then clean URL
            document.cookie = "ragger_token=" + encodeURIComponent(urlToken)
                + ";path=/;max-age=86400;SameSite=Strict";
            window.history.replaceState({}, "", window.location.pathname);
            return urlToken;
        }
        // Fall back to cookie
        const match = document.cookie.match(/(?:^|;\s*)ragger_token=([^;]*)/);
        return match ? decodeURIComponent(match[1]) : "";
    }

    AUTH_TOKEN = getToken();

    // If no token, redirect to login
    if (!AUTH_TOKEN) {
        window.location.href = "/login.html";
        return;
    }

    // --- Helpers ---
    function setStatus(state) {
        status.className = "status " + state;
    }

    function scrollToBottom() {
        messages.scrollTop = messages.scrollHeight;
    }

    function addMessage(role, text) {
        const div = document.createElement("div");
        div.className = "message " + role;
        div.textContent = text;
        messages.appendChild(div);
        scrollToBottom();
        return div;
    }

    function setEnabled(enabled) {
        sendBtn.disabled = !enabled;
        input.disabled = !enabled;
        if (enabled) input.focus();
    }

    // Auto-resize textarea
    input.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 200) + "px";
    });

    // Shift+Enter for newline, Enter to send
    input.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            form.dispatchEvent(new Event("submit"));
        }
    });

    // --- Send message via POST, read SSE stream ---
    form.addEventListener("submit", function (e) {
        e.preventDefault();

        const text = input.value.trim();
        if (!text || streaming) return;

        // Show user message
        addMessage("user", text);
        input.value = "";
        input.style.height = "auto";

        // Prepare assistant bubble
        const assistantDiv = addMessage("assistant", "");
        streaming = true;
        setEnabled(false);
        setStatus("streaming");

        // POST the message, parse SSE from response body
        const headers = { "Content-Type": "application/json" };
        if (AUTH_TOKEN) {
            headers["Authorization"] = "Bearer " + AUTH_TOKEN;
        }

        const body = JSON.stringify({
            message: text,
            session_id: sessionId
        });

        fetch(CHAT_ENDPOINT, {
            method: "POST",
            headers: headers,
            body: body
        }).then(function (response) {
            if (response.status === 401) {
                window.location.href = "/login.html";
                return;
            }
            if (!response.ok) {
                throw new Error("Server returned " + response.status);
            }

            // Read the SSE stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let fullText = "";

            function read() {
                return reader.read().then(function (result) {
                    if (result.done) {
                        onStreamDone();
                        return;
                    }

                    buffer += decoder.decode(result.value, { stream: true });

                    // Parse SSE lines
                    const lines = buffer.split("\n");
                    buffer = lines.pop();  // keep incomplete line

                    for (let i = 0; i < lines.length; i++) {
                        const line = lines[i];
                        if (!line.startsWith("data: ")) continue;

                        try {
                            const data = JSON.parse(line.slice(6));

                            if (data.token) {
                                fullText += data.token;
                                assistantDiv.textContent = fullText;
                                scrollToBottom();
                            }

                            if (data.done) {
                                if (data.session_id) {
                                    sessionId = data.session_id;
                                }
                            }

                            if (data.error) {
                                addMessage("error", data.error);
                            }
                        } catch (parseErr) {
                            // skip malformed lines
                        }
                    }

                    return read();
                });
            }

            return read();

        }).catch(function (err) {
            addMessage("error", "Error: " + err.message);
            if (assistantDiv.textContent === "") {
                assistantDiv.remove();
            }
            onStreamDone();
        });

        function onStreamDone() {
            streaming = false;
            setEnabled(true);
            setStatus("connected");
        }
    });

    // --- Logout ---
    var logoutBtn = document.getElementById("logout-btn");
    if (logoutBtn) {
        logoutBtn.addEventListener("click", function () {
            document.cookie = "ragger_token=;path=/;max-age=0";
            window.location.href = "/login.html";
        });
    }

    // --- Init ---
    setStatus("connected");
    input.focus();

})();
