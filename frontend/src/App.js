import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [status, setStatus] = useState("Waiting to begin");

  const [detectedLetter, setDetectedLetter] = useState("");
  const [currentWord, setCurrentWord] = useState("");
  const [confidence, setConfidence] = useState(0);

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
  }

  function toggleCamera() {
    if (cameraOn) {
      stopCamera();
    } else {
      startCamera();
    }
  }

  useEffect(() => {
    if (!cameraOn) {
      return;
    }

    let requestInProgress = false;

    async function sendFrameForPrediction() {
      const video = videoRef.current;

      if (
        requestInProgress ||
        !video ||
        video.readyState < 2 ||
        video.videoWidth === 0
      ) {
        return;
      }

      requestInProgress = true;

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
          return;
        }

        const formData = new FormData();
        formData.append("image", imageBlob, "camera-frame.jpg");

        const response = await fetch("http://127.0.0.1:5000/predict", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Prediction failed: ${response.status}`);
        }

        const data = await response.json();

        setConfidence(data.confidence || 0);

        if (!data.handDetected) {
          setDetectedLetter("");
          setStatus("Position your hand in the camera view");
        } else if (data.recognized) {
          setDetectedLetter(data.letter);
          setStatus("Sign recognized");
        } else {
          setDetectedLetter("");
          setStatus("Hold the sign steady");
        }
      } catch (error) {
        console.error("Prediction error:", error);
        setStatus("Recognition service unavailable");
      } finally {
        requestInProgress = false;
      }
    }

    sendFrameForPrediction();

    const predictionInterval = setInterval(
      sendFrameForPrediction,
      500
    );

    return () => {
      clearInterval(predictionInterval);
    };
  }, [cameraOn]);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });
      }
    };
  }, []);

  function clearWord() {
    setDetectedLetter("");
    setCurrentWord("");
    setConfidence(0);
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
            <button className="primary-button" disabled>
              Send Message
            </button>

            <button className="clear-button" onClick={clearWord}>
              Clear
            </button>
          </div>
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
        </div>

        <div className="output-area">
          <p className="eyebrow hearing-text">CURRENT TRANSLATION</p>
          <div className="translation-output">
            {currentWord || "Waiting for a translated message..."}
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
            <span className="dot offline-dot" />
            Recognition model not connected
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