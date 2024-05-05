import { useState } from "react";
import ImageInput from "./ImageInput";
import OutputViewer from "./OutputViewer";
import { uploadImage } from "./api";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [outputType, setOutputType] = useState(null);
  const [output, setOutput] = useState(null);
  const handleSubmit = (e, isSummarize) => {
    console.log("Submit clicked");

    if (isSummarize) {
      setOutputType("Summarization");
    } else {
      setOutputType("Recognition");
    }

    const endpoint = isSummarize ? "summarize" : "recognize";
    uploadImage(endpoint, image)
      .then((response) => {
        console.log(response);
        setOutput(response);
      })
      .catch((error) => {
        console.error(error);
      });
  };
  return (
    <div className='app'>
      <h1 className='title'>Handwritten text summarizer</h1>
      <div className='app-container'>
        <div className='container'>
          <h1>Please upload an image</h1>
          <div className='image-container'>
            {image ? (
              <img src={image} alt='Uploaded' />
            ) : (
              <p className='img-placeholder'>No image uploaded</p>
            )}
          </div>
          <div className='button-container'>
            <div className='buttons'>
              <button className='clear-button' onClick={() => setImage(null)}>
                Clear
              </button>
              <button onClick={(e) => handleSubmit(e, true)}>Summarize</button>
              <button onClick={(e) => handleSubmit(e, false)}>Read</button>
              <ImageInput image={image} setImage={setImage} />
            </div>
          </div>
        </div>
        <OutputViewer output={output} outputType={outputType} />
      </div>
    </div>
  );
}

export default App;
