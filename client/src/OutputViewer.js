import LoadingIndicator from "./LoadingIndicator";

const OutputViewer = ({ output, outputType }) => {
  return (
    <div className='container'>
      <div className='output-title-container'>
        <h2>{outputType}</h2>
      </div>
      <div className='output-container'>
        {false ? output : <LoadingIndicator />}
      </div>
    </div>
  );
};

export default OutputViewer;
