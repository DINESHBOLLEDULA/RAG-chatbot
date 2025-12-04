
import { useEffect, useState, useRef } from "react";

// Toast Component
function Toast({ message, onClose, isError = false }) {
  useEffect(() => {
    const timer = setTimeout(onClose, 3000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className={`bg-gradient-to-r ${isError ? 'from-red-800 to-red-900' : 'from-gray-800 to-gray-900'} text-white px-6 py-4 rounded-2xl shadow-2xl flex items-center gap-4 animate-slideIn min-w-[320px] mb-3`}>
      <div className={`w-8 h-8 ${isError ? 'bg-red-500' : 'bg-emerald-500'} rounded-full flex items-center justify-center flex-shrink-0`}>
        {isError ? (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        )}
      </div>
      <span className="flex-1 font-medium">{message}</span>
      <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}

function App() {
  
  const [msg, setMsg] = useState(false);
  const [pdfStatus, setPdfStatus] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [toasts, setToasts] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const abortUploadRef = useRef(false);

  const showToast = (message, isError = false) => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, isError }]);
  };

  const removeToast = (id) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  };

  const checkBackend = async () => {
    try {
      const res = await fetch("https://rag-chatbot-1-urn7.onrender.com/health");
      const data = await res.json();
      console.log("Response:", data);
      setMsg(data.status === "backend connected");
    } catch (error) {
      console.error("Fetch error:", error);
      setMsg(false);
    }
  };

  useEffect(() => {
    checkBackend();
    const interval = setInterval(() => {
    checkBackend();
  }, 5000);

  return () => clearInterval(interval);
  }, []);

  const uploadPdf = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Reset state
    setIsUploading(true);
    setPdfStatus(null);
    setToasts([]);
    abortUploadRef.current = false;
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("https://rag-chatbot-1-urn7.onrender.com/upload-pdf-stream", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // If aborted, stop processing
        if (abortUploadRef.current) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6);
            try {
              const data = JSON.parse(jsonStr);
              
              // Handle error - show error toast and abort
              if (data.step === 'error') {
                showToast(`❌ ${data.message}: ${data.error}`, true);
                abortUploadRef.current = true;
                setIsUploading(false);
                break;
              }
              
              // Show toast only after step completes successfully
              showToast(data.message, false);
              
              // If complete, save the PDF status
              if (data.step === 'complete') {
                setPdfStatus(data.data);
                setIsUploading(false);
              }
              
            } catch (e) {
              console.error('Error parsing SSE data:', e);
              showToast("❌ Error processing server response", true);
              abortUploadRef.current = true;
              setIsUploading(false);
              break;
            }
          }
        }
        
        // Break outer loop if aborted
        if (abortUploadRef.current) break;
      }
    } catch (error) {
      console.error('Upload error:', error);
      showToast(`❌ Upload failed: ${error.message}`, true);
      setIsUploading(false);
    }
    
    // Reset file input
    e.target.value = '';
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const userMsg = { role: "user", content: input };
    setMessages([...messages, userMsg]);
    setInput("");
    
    try {
      const res = await fetch("https://rag-chatbot-1-urn7.onrender.com/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: input }),
      });
      const data = await res.json();
      setMessages((msgs) => [...msgs, { role: "ai", content: data }]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((msgs) => [...msgs, { role: "ai", content: "Sorry, an error occurred." }]);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-6">
      {/* Toast Container - Fixed position, stacked vertically */}
      <div className="fixed top-6 right-6 z-50 flex flex-col pointer-events-none">
        {toasts.map((toast) => (
          <div key={toast.id} style={{ pointerEvents: 'auto' }}>
            <Toast
              message={toast.message}
              isError={toast.isError}
              onClose={() => removeToast(toast.id)}
            />
          </div>
        ))}
      </div>

      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="h-16 bg-white/10 backdrop-blur-xl rounded-xl flex items-center px-6 shadow-xl">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
            📚 RAG-DEMO
          </h1>

          <label
            htmlFor="pdf-upload"
            style={{ caretColor: "transparent"}}
            className={`ml-auto px-4 py-2 rounded-xl font-medium transition-all cursor-pointer ${
              isUploading
                ? "bg-gray-500 cursor-not-allowed"
                : "bg-emerald-500 hover:bg-emerald-600"
            }`}
          >
            {isUploading ? "⏳ Uploading..." : "📄 Upload PDF"}
          </label>
          <input
            id="pdf-upload"
            type="file"
            onChange={uploadPdf}
            accept=".pdf"
            disabled={isUploading}
            style={{ display: "none",cursor:"pointer" }}
            
          />
        </div>

        {/* Main Content */}
        <div className="flex gap-6 mt-6 h-[calc(100vh-10rem)]">
          {/* Chat Area */}
          <div className="flex-1 bg-white/5 backdrop-blur-xl rounded-2xl p-6 flex flex-col shadow-xl">
            <h2 className="text-xl font-semibold mb-4 text-gray-200">Chat Area</h2>
            
            <div className="flex-1 overflow-y-auto space-y-4 mb-4">
              {messages.length === 0 ? (
                <div className="text-center text-gray-400 mt-8">
                  <p>Upload a PDF and start asking questions!</p>
                </div>
              ) : (
                messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex ${
                      msg.role === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    <div
                      className={`max-w-2xl p-4 rounded-3xl ${
                        msg.role === "user"
                          ? "bg-gradient-to-r from-indigo-500 to-purple-500 text-white"
                          : "bg-white/10 backdrop-blur-xl border border-white/20"
                      }`}
                    >
                      {msg.content}
                    </div>
                  </div>
                ))
              )}
            </div>

            <div className="h-14 bg-white/5 rounded-xl flex items-center px-4 gap-3 border border-white/10">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && sendMessage()}
                placeholder="Ask about your PDF..."
                className="flex-1 bg-transparent outline-none text-white placeholder-gray-400"
                disabled={!pdfStatus}
              />
              <button
                onClick={sendMessage}
                disabled={!pdfStatus || !input.trim()}
                className={`px-4 py-2 rounded-lg transition-colors font-medium ${
                  !pdfStatus || !input.trim()
                    ? "bg-gray-500 cursor-not-allowed"
                    : "bg-indigo-500 hover:bg-indigo-600"
                }`}
              >
                Send
              </button>
            </div>
          </div>

          {/* PDF Panel */}
          <div className="w-96 bg-white/5 backdrop-blur-xl rounded-2xl p-6 shadow-xl">
            <h2 className="text-xl font-semibold mb-4 text-gray-200">PDF Status</h2>
            
            {pdfStatus ? (
              <div className="space-y-3 text-sm">
                <div className="p-3 bg-emerald-500/20 border border-emerald-500/50 rounded-lg">
                  <div className="flex items-center gap-2 text-emerald-400 mb-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <span className="font-semibold">Ready to Chat</span>
                  </div>
                </div>
                <div className="p-3 bg-white/10 rounded-lg">
                  <span className="text-gray-400">Filename:</span>
                  <p className="font-medium mt-1 break-words">{pdfStatus.filename}</p>
                </div>
                <div className="p-3 bg-white/10 rounded-lg">
                  <span className="text-gray-400">Pages:</span>
                  <p className="font-medium mt-1">{pdfStatus.pages}</p>
                </div>
                <div className="p-3 bg-white/10 rounded-lg">
                  <span className="text-gray-400">Chunks:</span>
                  <p className="font-medium mt-1">{pdfStatus.chunks}</p>
                </div>
                <div className="p-3 bg-white/10 rounded-lg">
                  <span className="text-gray-400">Vectors:</span>
                  <p className="font-medium mt-1">{pdfStatus.vectors_inserted}</p>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-400 mt-8">
                <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p>No PDF uploaded yet</p>
                <p className="text-sm mt-2">Upload a PDF to get started</p>
              </div>
            )}
          </div>
        </div>

        {/* Connection Status */}
        <div className="mt-4 text-center text-sm flex justify-center items-center gap-2">
  {/* Status Dot */}
  <span
    className={`h-3 w-3 rounded-full animate-pulse ${
      msg ? "bg-green-500" : "bg-red-500"
    }`}
  ></span>

  {/* Status Text */}
  <span className={msg ? "text-green-600" : "text-red-500"}>
    {msg ? "Backend is Live" : "Backend is Not Live"}
  </span>
</div>


      </div>

      <style>{`
        @keyframes slideIn {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
        .animate-slideIn {
          animation: slideIn 0.3s ease-out;
        }
      `}</style>
    </div>
  );
}

export default App;