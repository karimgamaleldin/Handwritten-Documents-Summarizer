const ImageInput = ({ image, setImage }) => {
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onloadend = () => {
      setImage(reader.result);
    };
    reader.readAsDataURL(file);
  };

  return <input type='file' accept='image/*' onChange={handleImageChange} />;
};

export default ImageInput;
