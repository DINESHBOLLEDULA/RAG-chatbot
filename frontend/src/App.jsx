import { useEffect,useState } from 'react'
import axios from 'axios'

function App() {
  const [msg,setMsg]=useState('')
  const checkBackend = async () => {
  try {
    const res = await axios.get("https://rag-chatbot-1-urn7.onrender.com/health");
    console.log("Response:", res.data);
    setMsg(JSON.stringify(res.data.status))
  } catch (error) {
    console.error("Axios error:", error);
  }
};

useEffect(() => {
  checkBackend()
}, [])
  return (
    <>
      <h1>{msg}</h1>
    </>
  )
}

export default App
