import React, { useState, useEffect } from "react";
import { PlusOutlined } from "@ant-design/icons";
import {
  Image,
  Upload,
  Card,
  Typography,
  Button,
  InputNumber,
  Modal,
} from "antd";
import testImage from "./assets/test_2.jpg";
import { uploadImage } from "./api";

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
  const [task, setTask] = useState("recognize"); // ["summarize", "recognize"]
  const [fileList, setFileList] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [output, setOutput] = useState(
    "Please choose an image to summarize or recognize"
  );
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    setSelectedFile(null);
  }, [fileList]);

  const handlePreview = async (file) => {
    if (!file.url && !file.preview) {
      file.preview = await getBase64(file.originFileObj);
    }
    setPreviewImage(file.url || file.preview);
    setPreviewOpen(true);
  };
  const handleChange = ({ fileList: newFileList }) => {
    setFileList(newFileList);
  };
  const handleSummarize = async () => {
    if (task === "summarize") {
      if (selectedFile === null) {
        alert("Please select a file");
        return;
      }
      setIsLoading(true);
      const img = selectedFile.originFileObj;
      const result = await uploadImage(img, minLength, maxLength, "summarize");
      setOutput(result);
    }
    setIsLoading(false);
    setTask("summarize");
  };

  const handleRecognize = async () => {
    if (selectedFile === null) {
      alert("Please select a file");
      return;
    }
    setIsLoading(true);
    const img = selectedFile.originFileObj;
    const result = await uploadImage(img, 0, 0, "recognize");
    setOutput(result);
    setIsLoading(false);
    setTask("recognize");
  };

  const handleClear = () => {
    setSelectedFile(null);
    setOutput("Please choose an image to summarize or recognize");
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
        <p>{output}</p>
      </Card>
      <div style={{ display: "flex", justifyContent: "center", marginTop: 20 }}>
        <Upload
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
          customRequest={async ({ file, onSuccess }) => {
            setTimeout(() => {
              onSuccess("ok");
            }, 0);
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
      <Modal
        open={isLoading}
        title='Processing'
        footer={null}
        closable={false}
        maskClosable={false}
      >
        <p>Please wait while we are processing your image...</p>
      </Modal>
    </>
  );
};
export default App;
