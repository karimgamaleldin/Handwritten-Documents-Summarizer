import axios from "axios";

async function uploadImage(endpoint, image) {
  const url = `http://127.0.0.1:8000/api/summarize/`;
  const formData = new FormData();
  formData.append("image", image);
  console.log(formData);

  const config = {
    headers: {
      // "content-type": "multipart/form-data",
    },
  };

  try {
    const response = await axios.post(url, formData, config);
    return response.data;
  } catch (error) {
    console.error(error);
    throw error;
  }
}

export { uploadImage };
