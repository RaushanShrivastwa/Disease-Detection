import React, { useState, useRef } from 'react';

// --- UI Components ---
const CameraView = ({ onCapture, onClose }) => {
    const videoRef = useRef(null);
    const [stream, setStream] = useState(null);

    React.useEffect(() => {
        const openCamera = async () => {
            try {
                const mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
                setStream(mediaStream);
                if (videoRef.current) {
                    videoRef.current.srcObject = mediaStream;
                }
            } catch (err) {
                console.error("Error accessing camera: ", err);
                alert("Could not access the camera. Please ensure you have given permission.");
                onClose();
            }
        };

        openCamera();
        
        return () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        };
    }, []); // eslint-disable-line

    const handleCapture = () => {
        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        canvas.getContext('2d').drawImage(videoRef.current, 0, 0);
        onCapture(canvas.toDataURL('image/jpeg'));
        onClose();
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-80 flex flex-col items-center justify-center z-50 p-4">
            <video ref={videoRef} autoPlay playsInline className="w-full max-w-lg h-auto rounded-lg shadow-2xl"></video>
            <div className="flex gap-4 mt-6">
                 <button onClick={handleCapture} className="px-6 py-3 bg-cyan-500 text-white font-bold rounded-lg hover:bg-cyan-600 transition-all transform hover:scale-105">
                    Capture Photo
                </button>
                 <button onClick={onClose} className="px-6 py-3 bg-slate-700 text-slate-200 font-bold rounded-lg hover:bg-slate-600 transition-all">
                    Close Camera
                </button>
            </div>
        </div>
    );
};

const PredictionResult = ({ result, imageSrc }) => {
  if (!result) return null;

  return (
    <div className="mt-8 p-6 bg-slate-900/50 rounded-lg border border-cyan-500/30 animate-fade-in">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="md:col-span-1">
                <h3 className="text-lg font-bold text-cyan-300 mb-2">Image Analyzed</h3>
                 <img src={imageSrc} alt="User upload for analysis" className="rounded-lg object-cover w-full h-auto aspect-square shadow-lg" />
            </div>
            <div className="md:col-span-2">
                 <div className="flex justify-between items-baseline mb-3">
                    <p className="text-4xl font-bold text-white">{result.prediction}</p>
                    <span className="text-lg font-semibold px-3 py-1 rounded-full bg-cyan-500/20 text-cyan-300">
                      {result.confidence}% Confidence
                    </span>
                  </div>
                  <p className="text-slate-300 mb-6">{result.description}</p>
                  
                  <h4 className="font-semibold text-cyan-400 mb-2">Recommended Precautions:</h4>
                  <ul className="list-disc list-inside text-slate-300 space-y-1">
                    {result.precautions.map((precaution, index) => (
                      <li key={index}>{precaution}</li>
                    ))}
                  </ul>
                  
                  <div className="mt-6 p-4 bg-slate-800/50 rounded-lg">
                    <p className="text-slate-400 text-sm">
                      <strong>Disclaimer:</strong> This is an AI-based prediction and should not be considered a medical diagnosis. 
                      Please consult a healthcare professional for accurate diagnosis and treatment.
                    </p>
                  </div>
            </div>
        </div>
    </div>
  );
};

// --- Main App Component ---
export default function DiseasePrediction() {
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [imageSrc, setImageSrc] = useState(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  
  const handleImageChange = (e) => {
      if (e.target.files && e.target.files[0]) {
          const reader = new FileReader();
          reader.onload = (event) => {
              setImageSrc(event.target.result);
              setPredictionResult(null);
              setError(null);
          };
          reader.readAsDataURL(e.target.files[0]);
      }
  };

  const handleCameraCapture = (imageData) => {
    setImageSrc(imageData);
    setPredictionResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!imageSrc) return;
    setIsLoading(true);
    setPredictionResult(null);
    setError(null);

    try {
      // Convert data URL to blob
      const response = await fetch(imageSrc);
      const blob = await response.blob();
      
      // Create form data
      const formData = new FormData();
      formData.append("file", blob, "skin_image.jpg");
      
      // Send to FastAPI backend
      const apiResponse = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!apiResponse.ok) {
        throw new Error(`Server returned ${apiResponse.status}: ${apiResponse.statusText}`);
      }

      const data = await apiResponse.json();
      setPredictionResult(data);
      
    } catch (err) {
      console.error("Prediction error:", err);
      setError("Error predicting disease. Please try again. Make sure the backend server is running.");
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleClear = () => {
      setImageSrc(null);
      setPredictionResult(null);
      setError(null);
      if(fileInputRef.current) {
          fileInputRef.current.value = "";
      }
  };

  return (
    <div className="bg-slate-900 min-h-screen flex items-center justify-center p-4 font-sans text-white">
      <style>{`
        @keyframes fade-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .animate-fade-in { animation: fade-in 0.5s ease-out forwards; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        .animate-pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
      `}</style>
      
      {isCameraOpen && <CameraView onCapture={handleCameraCapture} onClose={() => setIsCameraOpen(false)} />}
      
      <div className="w-full max-w-4xl mx-auto">
        <div className="bg-slate-800 shadow-2xl shadow-cyan-500/10 rounded-xl overflow-hidden">
            <div className="p-8">
                <header className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-cyan-400 tracking-wide">AI Skin Disease Predictor</h1>
                    <p className="text-slate-400 mt-2">Upload an image of a skin condition for AI analysis</p>
                </header>
                
                <main>
                    <div className="p-6 bg-slate-900/50 rounded-lg text-center">
                        <h2 className="text-xl font-semibold text-cyan-300 mb-4">Provide an Image</h2>
                         {imageSrc ? (
                            <div className="mb-4 relative group">
                                <img src={imageSrc} alt="Uploaded preview" className="max-w-xs mx-auto rounded-lg shadow-lg"/>
                                <button onClick={handleClear} className="absolute top-2 right-2 bg-black/50 text-white rounded-full p-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" /></svg>
                                </button>
                            </div>
                         ) : (
                            <div className="w-full h-48 border-2 border-dashed border-slate-600 rounded-lg flex items-center justify-center mb-4">
                                <p className="text-slate-500">No image selected</p>
                            </div>
                         )}

                        <div className="flex flex-col sm:flex-row gap-4 justify-center">
                           <button onClick={() => fileInputRef.current.click()} className="flex-1 bg-slate-700 text-slate-200 font-bold py-3 px-6 rounded-lg hover:bg-slate-600 transition-all flex items-center justify-center gap-2">
                             <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                               <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                             </svg>
                             Upload Image
                           </button>
                           <input type="file" ref={fileInputRef} onChange={handleImageChange} className="hidden" accept="image/*"/>
                           <button onClick={() => setIsCameraOpen(true)} className="flex-1 bg-slate-700 text-slate-200 font-bold py-3 px-6 rounded-lg hover:bg-slate-600 transition-all flex items-center justify-center gap-2">
                             <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                               <path fillRule="evenodd" d="M4 5a2 2 0 00-2 2v8a2 2 0 002 2h12a2 2 0 002-2V7a2 2 0 00-2-2h-1.586a1 1 0 01-.707-.293l-1.121-1.121A2 2 0 0011.172 3H8.828a2 2 0 00-1.414.586L6.293 4.707A1 1 0 015.586 5H4zm6 9a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                             </svg>
                             Use Camera
                           </button>
                        </div>
                    </div>
                    
                    <div className="flex justify-center mt-8">
                        <button
                            onClick={handlePredict}
                            disabled={isLoading || !imageSrc}
                            className="w-full sm:w-1/2 bg-cyan-500 text-white font-bold py-3 px-8 rounded-lg hover:bg-cyan-600 transition-all duration-300 disabled:bg-slate-600 disabled:cursor-not-allowed transform hover:scale-105 focus:outline-none focus:ring-4 focus:ring-cyan-500/50 flex items-center justify-center"
                        >
                            {isLoading ? (
                                <>
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Analyzing Image...
                                </>
                            ) : "Predict Condition"}
                        </button>
                    </div>
                    
                    {error && (
                      <div className="mt-6 p-4 bg-red-900/50 border border-red-500 rounded-lg text-center">
                        <p className="text-red-200">{error}</p>
                        <p className="text-sm text-red-300 mt-2">Make sure your FastAPI server is running on http://localhost:8000</p>
                      </div>
                    )}
                    
                    {predictionResult && <PredictionResult result={predictionResult} imageSrc={imageSrc} />}
                </main>
                
            </div>
        </div>
      </div>
    </div>
  );
}