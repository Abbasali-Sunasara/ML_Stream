import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, FileText, Loader2, AlertCircle } from 'lucide-react';

export default function UploadDataset({ onUploadComplete }) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      // CHANGED PORT TO 5000 (Flask)
      const response = await fetch('http://127.0.0.1:5000/upload', { 
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Upload failed");
      
      onUploadComplete(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }, [onUploadComplete]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false
  });

  return (
    <div className="w-full max-w-xl mx-auto">
       <div {...getRootProps()} className={`
          relative group cursor-pointer overflow-hidden rounded-2xl border-2 border-dashed transition-all duration-300
          ${isDragActive ? 'border-brand-cyan bg-brand-cyan/10' : 'border-white/10 hover:border-brand-indigo/50 hover:bg-white/5'}
          ${error ? 'border-red-500/50 bg-red-500/5' : ''}
       `}>
          <input {...getInputProps()} />
          <div className="absolute inset-0 bg-gradient-to-br from-brand-indigo/0 via-transparent to-brand-purple/0 opacity-0 group-hover:opacity-10 transition-opacity" />
          
          <div className="p-10 flex flex-col items-center justify-center text-center space-y-4 relative z-10">
             {uploading ? (
                <>
                  <Loader2 className="w-12 h-12 text-brand-cyan animate-spin" />
                  <p className="text-slate-300 font-mono animate-pulse">Uploading Pipeline...</p>
                </>
             ) : (
                <>
                  <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-2 shadow-xl transition-transform group-hover:scale-110 ${error ? 'bg-red-500/20 text-red-400' : 'bg-brand-indigo/20 text-brand-indigo'}`}>
                    {error ? <AlertCircle className="w-8 h-8" /> : <UploadCloud className="w-8 h-8" />}
                  </div>
                  <h3 className="text-xl font-bold text-white">
                    {error ? <span className="text-red-400">Upload Failed</span> : "Upload Dataset"}
                  </h3>
                  <p className="text-sm text-slate-400 max-w-xs mx-auto leading-relaxed">
                    {error ? error : "Drag & drop your CSV file here, or click to browse system."}
                  </p>
                </>
             )}
          </div>
       </div>
    </div>
  );
}