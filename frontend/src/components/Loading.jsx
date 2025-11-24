const Loading = ({ message = 'Loading...' }) => {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="loader"></div>
      <p className="mt-4 text-gray-600 text-sm">{message}</p>
    </div>
  );
};

export default Loading;
