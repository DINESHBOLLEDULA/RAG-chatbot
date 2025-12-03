import { useEffect, useState } from "react";
import axios from "axios";

function App() {
  const [msg, setMsg] = useState("");
  const [UploadMsg,setuploadMsg]=useState('')
  const checkBackend = async () => {
    try {
      const res = await axios.get("https://rag-chatbot-1-urn7.onrender.com/health");
      // const res = await axios.get("http://localhost:8000/health");
      console.log("Response:", res.data);
      setMsg(JSON.stringify(res.data.status));
    } catch (error) {
      console.error("Axios error:", error);
    }
  };

  useEffect(() => {
    checkBackend();
  }, []);


  const handleUpload= async ()=>{
      try{
        const res = await axios.get("https://rag-chatbot-1-urn7.onrender.com/upload-pdf");
        // const res = await axios.get("http://localhost:8000/upload-pdf");
        console.log("Response:", res.data);
        setuploadMsg(JSON.stringify(res.data.status));
      }
      catch (error) {
      console.error("Axios error:", error);
    }
  }
  return (
    <>
      
      <div className="h-16 bg-white/10 backdrop-blur-xl rounded-xl flex items-center px-6">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
          📚 RAG-DEMO
        </h1>
        <button className="ml-auto px-4 py-2 bg-emerald-500 text-white rounded-xl" onClick={handleUpload}>
          📁 Upload PDF
        </button>
      </div>
      <h1>{msg}</h1>
      <h1>{UploadMsg}</h1>
    </>
  );
}

export default App;
