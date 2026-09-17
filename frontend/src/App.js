import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [status, setStatus] = useState("Waiting to begin");

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
      setStatus('Camera error: ${error.name}');
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
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });
      }
    };
  }, []);

  return (
    <div className="app">
      <header>
        <h1>ASL Recognition System</h1>
        <p>Translate ASL signs into English in real time</p>
      </header>

      <main>
        <section className="camera-section">
          <div className="camera-container">
            <video
              ref={videoRef}
              className={cameraOn ? "camera-feed" : "camera-feed hidden"}
              autoPlay
              playsInline
              muted
            />

            {!cameraOn && <p>Camera feed will appear here</p>}
          </div>

          <button onClick={toggleCamera}>
            {cameraOn ? "Stop Camera" : "Start Camera"}
          </button>
        </section>

        <section className="results-section">
          <p className="status">Status: {status}</p>

          <div className="result-card">
            <h2>Detected Letter</h2>
            <p className="detected-letter">—</p>
          </div>

          <div className="result-card">
            <h2>Current Word</h2>
            <p className="current-word">—</p>
          </div>

          <button className="clear-button">Clear Word</button>
        </section>
      </main>
    </div>
  );
}

export default App;