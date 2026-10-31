import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL =
  process.env.REACT_APP_API_URL || "REACT_APP_API_URL=http://127.0.0.1:5001";

function App() {
  // camera references
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // prediction references
  const predictionInProgressRef = useRef(false);
  const lastAcceptedLetterRef = useRef("");
  const candidateLetterRef = useRef("");
  const candidateCountRef = useRef(0);

  // speech-recognition reference
  const recognitionRef = useRef(null);

  // prediction configuration
  const REQUIRED_STABLE_PREDICTIONS = 3;

  // connection and camera state
  const [modelConnected, setModelConnected] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [status, setStatus] = useState("Waiting to begin");

  // translation and conversation state
  const [detectedLetter, setDetectedLetter] = useState("");
  const [currentWord, setCurrentWord] = useState("");
  const [sentMessages, setSentMessages] = useState([]);

  // confidence state
  const [confidence, setConfidence] = useState(0);

  // welcome-page state
  const [showWelcome, setShowWelcome] = useState(true);

  // speech-recognition state
  const [isListening, setIsListening] = useState(false);
  const [speechPreview, setSpeechPreview] = useState(
    "Press Start Listening and begin speaking."
  );

  async function startCamera() {
    console.log("Start Camera was pressed");
    
    try {
      setStatus("Requesting camera access...");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      streamRef.current = stream;
      videoRef.current.srcObject = stream;

      setCameraOn(true);
      setStatus("Camera active");
    } catch (error) {
      console.error("Camera error:", error);
      setStatus(`Camera error: ${error.name}`);
    }
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraOn(false);
    setStatus("Camera stopped");
    setDetectedLetter("");
    setConfidence(0);

    lastAcceptedLetterRef.current = "";
    candidateLetterRef.current = "";
    candidateCountRef.current = 0;
  }

  function toggleCamera() {
    if (cameraOn) {
      stopCamera();
    } else {
      startCamera();
    }
  }

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });
      }

      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }

      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, []);

  useEffect(() => {
    if (!cameraOn) {
      return;
    }

    async function requestPrediction() {
      const video = videoRef.current;

      if (
        !video ||
        video.readyState < 2 ||
        predictionInProgressRef.current
      ) {
        return;
      }

      predictionInProgressRef.current = true;

      try {
        const canvas = document.createElement("canvas");

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const context = canvas.getContext("2d");
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        const imageBlob = await new Promise((resolve) => {
          canvas.toBlob(resolve, "image/jpeg", 0.8);
        });

        if (!imageBlob) {
          throw new Error("Unable to capture camera frame");
        }

        const formData = new FormData();
        formData.append("image", imageBlob, "camera-frame.jpg");

        const response = await fetch(`${API_URL}/predict`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Prediction request failed: ${response.status}`);
        }

        const data = await response.json();

        setModelConnected(true);
        setConfidence(data.confidence || 0);

        if (!data.handDetected) {
          setDetectedLetter("");
          lastAcceptedLetterRef.current = "";
          setStatus("Position your hand in view");

          candidateLetterRef.current = "";
          candidateCountRef.current = 0;

          return;
        }

        if (!data.recognized || !data.letter) {
          setDetectedLetter("");
          setStatus("Hold the sign steady");

          candidateLetterRef.current = "";
          candidateCountRef.current = 0;

          return;
        }

        setDetectedLetter(data.letter);

        if (data.letter === candidateLetterRef.current) {
          candidateCountRef.current += 1;
        } else {
          candidateLetterRef.current = data.letter;
          candidateCountRef.current = 1;
        }

        if (candidateCountRef.current < REQUIRED_STABLE_PREDICTIONS) {
          setStatus(
            `Hold sign steady (${candidateCountRef.current}/${REQUIRED_STABLE_PREDICTIONS})`
          );
          return;
        }

        setStatus("Sign recognized");

        if (data.letter !== lastAcceptedLetterRef.current) {
          setCurrentWord((previousWord) => previousWord + data.letter);
          lastAcceptedLetterRef.current = data.letter;
        }
      } catch (error) {
        console.error("Prediction error:", error);
        setModelConnected(false);
        setStatus("Recognition service unavailable");
      } finally {
        predictionInProgressRef.current = false;
      }
    }

    const predictionInterval = setInterval(requestPrediction, 500);

    return () => {
      clearInterval(predictionInterval);
    };
  }, [cameraOn]);

  function addSpace() {
    setCurrentWord((previous) => {
      if (!previous || previous.endsWith(" ")) {
        return previous;
      }

      return `${previous} `;
    });

    setDetectedLetter("");
  }

  useEffect(() => {
    function handleKeyDown(event) {
      const activeElement = document.activeElement;
      const shouldIgnoreShortcut =
        activeElement?.tagName === "INPUT" ||
        activeElement?.tagName === "TEXTAREA" ||
        activeElement?.tagName === "BUTTON" ||
        activeElement?.isContentEditable;

      if (shouldIgnoreShortcut) {
        return;
      }

      if (event.code === "Space") {
        event.preventDefault();

        setCurrentWord((previous) => {
          if (!previous || previous.endsWith(" ")) {
            return previous;
          }

          return `${previous} `;
        });

        setDetectedLetter("");
      }

      if (event.key === "Backspace") {
        event.preventDefault();
        setCurrentWord((previous) => previous.slice(0, -1));
        setDetectedLetter("");
      }

      if (event.key === "Escape") {
        setCurrentWord("");
        setDetectedLetter("");
        setConfidence(0);

        lastAcceptedLetterRef.current = "";
        candidateLetterRef.current = "";
        candidateCountRef.current = 0;
      }
    }

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  function backspace() {
    setCurrentWord((previous) => previous.slice(0, -1));
    setDetectedLetter("");
  }

  function clearWord() {
    setDetectedLetter("");
    setCurrentWord("");
    setConfidence(0);

    lastAcceptedLetterRef.current = "";
    candidateLetterRef.current = "";
    candidateCountRef.current = 0;
  }

  if (showWelcome) {
    return (
      <div className="welcome-page">
        <div className="welcome-content">
          <p className="eyebrow asl-text">ASL BRIDGE</p>

          <h1>Communication without barriers</h1>

          <p className="welcome-description">
            Translate supported American Sign Language letters into readable
            English text using your camera.
          </p>

          <button
            className="welcome-start-button"
            onClick={() => setShowWelcome(false)}
          >
            Start Translating
          </button>

          <div className="welcome-steps">
            <div>
              <span>01</span>
              <p>Allow camera access</p>
            </div>

            <div>
              <span>02</span>
              <p>Position your hand in view</p>
            </div>

            <div>
              <span>03</span>
              <p>Hold a sign to translate it</p>
            </div>
          </div>

          <p className="welcome-note">
            Camera access begins only after you choose Start Camera.
          </p>
        </div>
      </div>
    );
  }

  function speakText(text) {
    if (
      !("speechSynthesis" in window) ||
      !("SpeechSynthesisUtterance" in window)
    ) {
      setStatus("Text-to-speech is not supported by this browser");
      return;
    }

    window.speechSynthesis.cancel();

    const speech = new SpeechSynthesisUtterance(text);

    speech.lang = "en-US";
    speech.rate = 0.95;
    speech.pitch = 1;
    speech.volume = 1;

    speech.onerror = (event) => {
      if (
        event.error === "canceled" ||
        event.error === "interrupted"
      ) {
        return;
      }
      
      console.error("Speech error:", event.error);
      setStatus("Unable to speak message");
    };

    window.speechSynthesis.speak(speech);
  }
  
  function sendMessage() {
    const message = currentWord.trim();

    if (!message) {
      return;
    }

    setSentMessages((previousMessages) => [
      ...previousMessages,
      {
        sender: "asl",
        text: message,
      }
    ]);

    speakText(message);

    setCurrentWord("");
    setDetectedLetter("");
    setConfidence(0);
    setStatus("Message sent");

    candidateLetterRef.current = "";
    candidateCountRef.current = 0;
  }

  function toggleSpeechRecognition() {
    if (isListening) {
      recognitionRef.current?.stop();
      return;
    }

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setSpeechPreview(
        "Speech recognition is not supported by this browser."
      );
      return;
    }

    const recognition = new SpeechRecognition();

    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsListening(true);
      setSpeechPreview("Listening...");
    };

    recognition.onresult = (event) => {
      let interimText = "";
      let finalText = "";

      for (
        let index = event.resultIndex;
        index < event.results.length;
        index += 1
      ) {
        const transcript = event.results[index][0].transcript;

        if (event.results[index].isFinal) {
          finalText += transcript;
        } else {
          interimText += transcript;
        }
      }

      if (finalText.trim()) {
        const completedTranscript = finalText.trim();

        setSentMessages((previousMessages) => [
          ...previousMessages,
          {
            sender: "hearing",
            text: completedTranscript,
          },
        ]);

        setSpeechPreview(`Heard: ${completedTranscript}`);
      } else {
        setSpeechPreview(interimText || "Listening...");
      }
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);

      if (event.error === "not-allowed") {
        setSpeechPreview("Microphone permission was denied.");
      } else if (event.error === "no-speech") {
        setSpeechPreview("No speech was detected. Please try again.");
      } else {
        setSpeechPreview(`Speech recognition error: ${event.error}`);
      }
    };

    recognition.onend = () => {
      setIsListening(false);
      recognitionRef.current = null;
    };

    recognitionRef.current = recognition;
    recognition.start();
  }

  return (
    <div className="app-shell">
      <aside className="sidebar camera-sidebar">
        <div className="brand">
          <p className="eyebrow asl-text">ASL BRIDGE</p>
          <h1>Sign Language Interpreter</h1>

          <div className="user-key">
            <span><i className="dot asl-dot" /> ASL USER</span>
            <span><i className="dot hearing-dot" /> HEARING USER</span>
          </div>
        </div>

        <div className="camera-container">
          <video
            ref={videoRef}
            className={cameraOn ? "camera-feed" : "camera-feed hidden"}
            autoPlay
            playsInline
            muted
          />

          {!cameraOn && (
            <div className="camera-placeholder">
              <p>Camera feed will appear here</p>
            </div>
          )}
        </div>

        <div className="camera-status">
          <div className="letter-preview">
            {detectedLetter || "—"}
          </div>

          <div className="confidence">
            <p>{status}</p>
            <div className="confidence-track">
              <div
                className="confidence-fill"
                style={{ width: `${Math.round(confidence * 100)}%` }}
              />
            </div>
            <span>{Math.round(confidence * 100)}% confidence</span>
          </div>
        </div>

        <div className="composer">
          <p className="eyebrow asl-text">ASL — COMPOSING</p>
          <p className="current-word">
            {currentWord || "Sign letters to build a message..."}
          </p>

          <div className="button-row">
            <button
              className="edit-button"
              onClick={addSpace}
              disabled={!currentWord || currentWord.endsWith(" ")}
            >
              Space
            </button>

            <button
              className="edit-button"
              onClick={backspace}
              disabled={!currentWord}
            >
              Backspace
            </button>

            <button
              className="clear-button"
              onClick={clearWord}
              disabled={!currentWord}
            >
              Clear
            </button>
          </div>

          <button
            className="primary-button"
            onClick={sendMessage}
            disabled={!currentWord.trim()}
          >
            Send + Speak Aloud
          </button>
        </div>

        <button className="camera-button" onClick={toggleCamera}>
          {cameraOn ? "Stop Camera" : "Start Camera"}
        </button>
      </aside>

      <main className="conversation-panel">
        <div className="conversation-header">
          <span className="dot hearing-dot" />
          <span>TRANSLATION OUTPUT</span>
        </div>

        <div className="conversation-area">
          <div className="welcome-message">
            <p className="message-author">SYSTEM</p>
            <p>
              Start the camera and sign your message. Recognized letters and
              words will appear here.
            </p>
          </div>

          {sentMessages.map((message, index) => (
            <div
              className={
                message.sender === "hearing"
                  ? "sent-message hearing-message"
                  : "sent-message"
              }
              key={`${message.sender}-${message.text}-${index}`}
            >
              <p className="message-author">
                {message.sender === "hearing" ? "HEARING USER" : "ASL USER"}
              </p>

              <p>{message.text}</p>
            </div>
          ))}

        <div className="output-area">
          <p className="eyebrow hearing-text">CURRENT TRANSLATION</p>

          <div className="translation-output">
            {currentWord || "Waiting for a translated message..."}
          </div>

          <div className="speech-controls">
            <p className="eyebrow hearing-text">
              HEARING USER — SPEECH TO TEXT
            </p>

            <button
              className={
                isListening
                  ? "microphone-button listening"
                  : "microphone-button"
              }
              onClick={toggleSpeechRecognition}
              aria-pressed={isListening}
            >
              {isListening ? "Stop Listening" : "Start Listening"}
            </button>

            <p className="speech-preview" aria-live="polite">
              {speechPreview}
            </p>
          </div>
        </div>
      </div>
    </main>

      <aside className="sidebar guide-sidebar">
        <section>
          <p className="eyebrow asl-text">ASL USER</p>

          <ol className="instructions">
            <li>Position your hand inside the camera view.</li>
            <li>Hold each sign until it is recognized.</li>
            <li>Move your hand away before repeating a letter.</li>
            <li>Review the translated word in the center panel.</li>
          </ol>
        </section>

        <section className="guide-section">
          <p className="eyebrow hearing-text">SYSTEM STATUS</p>

          <div className="status-box">
            <span className={cameraOn ? "dot asl-dot" : "dot offline-dot"} />
            {cameraOn ? "Camera active" : "Camera offline"}
          </div>

          <div className="status-box">
            <span className={modelConnected ? "dot asl-dot" : "dot offline-dot"} />
            {modelConnected
              ? "Recognition model connected"
              : "Recognition model not connected"}
          </div>
        </section>

        <section className="guide-section">
          <p className="eyebrow">SUPPORTED INPUT</p>
          
          <p className="guide-copy">
            Static alphabet signs are being developed first. Motion-based
            letters J and Z will be added afterward.
          </p>
        </section>
      </aside>
    </div>
  );
}

export default App;