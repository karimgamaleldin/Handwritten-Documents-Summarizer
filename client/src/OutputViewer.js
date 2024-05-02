const OutputViewer = ({ output, outputType }) => {
  return (
    <div className='container'>
      <div className='output-title-container'>
        <h2>{outputType} output</h2>
      </div>
      <div className='output-container'>{output}</div>
    </div>
  );
};

export default OutputViewer;
