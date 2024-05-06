import React, { useState } from "react";
import { PlusOutlined } from "@ant-design/icons";
import { Image, Upload, Card, Typography, Button, InputNumber } from "antd";
import testImage from "./assets/test_2.jpg";
const getBase64 = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = (error) => reject(error);
  });
const App = () => {
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewImage, setPreviewImage] = useState("");
  const [minLength, setMinLength] = useState();
  const [maxLength, setMaxLength] = useState();
  const [task, setTask] = useState("summarize"); // ["summarize", "recognize"]
  const [fileList, setFileList] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const handlePreview = async (file) => {
    if (!file.url && !file.preview) {
      file.preview = await getBase64(file.originFileObj);
    }
    setPreviewImage(file.url || file.preview);
    setPreviewOpen(true);
  };
  const handleChange = ({ fileList: newFileList }) => {
    setFileList(newFileList);
    console.log("File list: ", newFileList);
  };
  const handleSummarize = () => {
    console.log("Summarize button clicked");
    if (task === "summarize") {
      const img = selectedFile.originFileObj;
      const imgBase64 = getBase64(img);
      console.log("Selected file: ", imgBase64);
      // API call to summarize the image
    }
    setTask("summarize");
  };

  const handleRecognize = async () => {
    console.log("Recognize button clicked");
    const img = selectedFile.originFileObj;
    const imgBase64 = await getBase64(img);
    console.log("Selected fiasfsle: ", imgBase64);
    // API call to recognize the image
    setTask("recognize");
  };

  const handleClear = () => {
    console.log("Clear button clicked");
    setFileList([]); // Assuming fileList is managed in the state for uploaded files
  };
  const uploadButton = (
    <button
      style={{
        border: 0,
        background: "none",
      }}
      type='button'
    >
      <PlusOutlined />
      <div
        style={{
          marginTop: 8,
        }}
      >
        Upload
      </div>
    </button>
  );
  const { Title } = Typography;

  const handleSelect = (file) => {
    setSelectedFile(file);
    console.log("Selected file: ", file);
  };

  const itemRender = (originNode, file, currFileList) => {
    return (
      <div
        style={{
          cursor: "pointer",
        }}
        onClick={(e) => {
          handleSelect(file);
        }}
      >
        {originNode}
      </div>
    );
  };
  return (
    <>
      <Title
        level={1}
        style={{
          textAlign: "center",
          color: "transparent",
          backgroundImage: "linear-gradient(45deg, #6a11cb, #2575fc)",
          backgroundClip: "text",
          WebkitBackgroundClip: "text",
          fontSize: "36px",
          fontWeight: "bold",
          margin: "40px 0",
        }}
      >
        Handwritten Documents Summarization and Recognition
      </Title>
      <Card
        title={<h2 style={{ fontSize: "24px", marginBottom: "0" }}>Output</h2>}
        style={{ width: "100%" }}
      >
        <p>Please choose an image to summarize or recognize</p>
      </Card>

      <div style={{ display: "flex", justifyContent: "center", marginTop: 20 }}>
        <Upload
          action='https://660d2bd96ddfa2943b33731c.mockapi.io/api/upload'
          listType='picture-circle'
          fileList={fileList}
          onPreview={handlePreview}
          onChange={handleChange}
          itemRender={itemRender}
          showUploadList={{
            showPreviewIcon: true,
            showRemoveIcon: true,
            showDownloadIcon: false,
          }}
        >
          {fileList.length >= 8 ? null : uploadButton}
        </Upload>
        {previewImage && (
          <Image
            wrapperStyle={{
              display: "none",
            }}
            preview={{
              visible: previewOpen,
              onVisibleChange: (visible) => setPreviewOpen(visible),
              afterOpenChange: (visible) => !visible && setPreviewImage(""),
            }}
            src={previewImage}
          />
        )}
      </div>
      {task === "summarize" && (
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            marginTop: 20,
            gap: "10px",
          }}
        >
          <div>
            <label>Min Length:</label>
            <InputNumber min={0} value={minLength} onChange={setMinLength} />
          </div>
          <div>
            <label>Max Length:</label>
            <InputNumber min={0} value={maxLength} onChange={setMaxLength} />
          </div>
        </div>
      )}
      <div style={{ display: "flex", justifyContent: "center", marginTop: 20 }}>
        <Button
          onClick={handleSummarize}
          type='primary'
          style={{ marginRight: "10px" }}
        >
          Summarize
        </Button>
        <Button
          onClick={handleRecognize}
          type='primary'
          style={{ margin: "0 10px" }}
        >
          Recognize
        </Button>
        <Button onClick={handleClear} style={{ marginLeft: "10px" }} danger>
          Clear
        </Button>
      </div>
    </>
  );
};
export default App;
