import axios from "axios";

async function uploadImage(
  imageFile,
  minLength = 100,
  maxLength = 150,
  task = "summarize"
) {
  const endpoint = `http://localhost:5000/${task}`;
  let formData = new FormData();

  formData.append("image", imageFile);

  if (task === "summarize") {
    formData.append("min_length", minLength);
    formData.append("max_length", maxLength);
  }
  try {
    const response = await axios({
      method: "post",
      url: endpoint,
      data: formData,
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data.prediction;
  } catch (error) {
    throw error; // Rethrow to ensure the calling function can catch
  }
}

export { uploadImage };
