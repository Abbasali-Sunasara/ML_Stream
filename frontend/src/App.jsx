import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LayoutDashboard, Table as TableIcon, Activity, Layers, Cpu, Database, CheckCircle, AlertCircle, BarChart2, Info } from 'lucide-react';
import UploadDataset from './components/UploadDataset';
import ExperimentConfig from './components/ExperimentConfig';

// --- COMPONENT: DNA Background ---
const DNABackground = () => {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let particles = [];
    const init = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      const particleCount = Math.floor(window.innerWidth / 15); 
      particles = [];
      for (let i = 0; i < particleCount; i++) {
        particles.push({
          x: Math.random() * canvas.width, y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * 0.5, vy: (Math.random() - 0.5) * 0.5,
          size: Math.random() * 1.5 + 1, color: i % 2 === 0 ? '#6366f1' : '#8b5cf6' 
        });
      }
    };
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach(p => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = canvas.width; if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height; if (p.y > canvas.height) p.y = 0;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2); ctx.fillStyle = p.color; ctx.fill();
      });
      ctx.lineWidth = 0.5;
      for (let i = 0; i < particles.length - 1; i++) {
        const p1 = particles[i]; const p2 = particles[i + 1];
        const dx = p1.x - p2.x; const dy = p1.y - p2.y;
        if (dx * dx + dy * dy < 20000) { 
          ctx.beginPath(); ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)'; ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        }
      }
      animationFrameId = requestAnimationFrame(draw);
    };
    init(); draw();
    window.addEventListener('resize', init);
    return () => { window.removeEventListener('resize', init); cancelAnimationFrame(animationFrameId); };
  }, []);
  return (
    <div className="absolute inset-0 pointer-events-none bg-dark-900">
       <canvas ref={canvasRef} className="absolute inset-0 z-0 opacity-100" />
       <div className="absolute inset-0 bg-radial-gradient from-transparent to-dark-900/80" />
    </div>
  );
};

// --- COMPONENT: Typewriter Text ---
const TypewriterText = ({ text }) => {
  const [displayedText, setDisplayedText] = useState("");
  useEffect(() => {
    setDisplayedText(""); let index = 0;
    const intervalId = setInterval(() => {
      setDisplayedText((prev) => prev + text.charAt(index)); index++;
      if (index === text.length) clearInterval(intervalId);
    }, 50); 
    return () => clearInterval(intervalId);
  }, [text]);
  return <span className="font-mono text-brand-cyan">{displayedText}<span className="animate-pulse">_</span></span>;
};

// --- COMPONENT: Navbar ---
const Navbar = ({ hasDataset, onReset }) => {
  return (
    <nav className="relative z-50 border-b border-white/5 bg-dark-900/50 backdrop-blur-md shrink-0">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-indigo to-brand-purple flex items-center justify-center shadow-lg shadow-brand-indigo/20">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div className="text-left"><span className="block text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-brand-indigo via-brand-purple to-brand-cyan">ML Studio</span></div>
        </div>
        {hasDataset && (
           <button onClick={onReset} className="text-xs font-bold text-red-400 hover:text-red-300 transition px-4 py-2 hover:bg-red-500/10 rounded-lg border border-red-500/20">EXIT STUDIO</button>
        )}
      </div>
    </nav>
  );
};

function App() {
  const [dataset, setDataset] = useState(null);
  const [activeTab, setActiveTab] = useState('config'); // 'config' | 'data' | 'analysis'
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const trainModel = async (config) => {
    setIsLoading(true); setError(null); setResults(null);
    try {
      const response = await fetch('http://127.0.0.1:5000/train', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Training failed");
      setResults(data);
    } catch (err) { setError(err.message); } finally { setIsLoading(false); }
  };

  return (
    <div className="h-screen w-full relative overflow-hidden bg-dark-900 text-slate-200 font-sans flex flex-col">
      <DNABackground />
      <Navbar hasDataset={!!dataset} onReset={() => { setDataset(null); setResults(null); setActiveTab('config'); }} />

      <AnimatePresence>
        {isLoading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 z-[100] bg-dark-900/80 backdrop-blur-md flex flex-col items-center justify-center">
             <div className="relative w-24 h-24 mb-8">
                <div className="absolute inset-0 border-4 border-brand-indigo/30 rounded-full"></div>
                <div className="absolute inset-0 border-4 border-brand-indigo rounded-full border-t-transparent animate-spin"></div>
                <Cpu className="absolute inset-0 m-auto w-10 h-10 text-brand-purple animate-pulse" />
             </div>
             <h2 className="text-2xl font-bold text-white mb-2">Training Neural Network</h2>
             <p className="text-slate-400 font-mono">Optimizing weights & biases...</p>
          </motion.div>
        )}
      </AnimatePresence>

      <main className="relative z-10 flex-1 flex flex-col items-center justify-center p-6 overflow-hidden">
        <AnimatePresence mode="wait">
          {!dataset ? (
            <motion.div key="landing" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} className="w-full max-w-6xl flex flex-col md:flex-row items-center gap-16">
              <div className="flex-1 space-y-8 text-center md:text-left z-20">
                <div className="space-y-2">
                  <h1 className="text-6xl md:text-8xl font-bold tracking-tighter text-white leading-[1] drop-shadow-2xl">ML <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-indigo to-brand-purple">Studio</span></h1>
                  <div className="text-xl md:text-2xl text-slate-400 h-8"><TypewriterText text="> System Online. Awaiting Data..." /></div>
                </div>
                <p className="text-lg text-slate-500 max-w-xl mx-auto md:mx-0 leading-relaxed">The comprehensive platform for machine learning development. Import datasets, optimize pipelines, and train professional models.</p>
                <div className="flex flex-wrap justify-center md:justify-start gap-4">
                  {[{ icon: Layers, text: "Auto-Pipeline", color: "text-brand-cyan" }, { icon: Cpu, text: "Scikit Engine", color: "text-brand-purple" }, { icon: Database, text: "Data Analytics", color: "text-brand-indigo" }].map((feat, i) => (
                    <div key={i} className="flex items-center gap-2 px-5 py-3 rounded-full bg-white/5 border border-white/10 backdrop-blur-md cursor-default transition-colors hover:bg-white/10">
                      <feat.icon className={`w-4 h-4 ${feat.color}`} /> <span className="text-sm font-semibold text-slate-300">{feat.text}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="flex-1 w-full max-w-md relative">
                 <div className="absolute inset-0 bg-brand-indigo/10 blur-[100px] rounded-full pointer-events-none transform-gpu" />
                 <div className="relative rounded-2xl border border-white/10 bg-white/[0.02] backdrop-blur-md shadow-2xl p-1"><UploadDataset onUploadComplete={(data) => setDataset(data)} /></div>
              </div>
            </motion.div>
          ) : (
            <motion.div key="dashboard" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="w-full max-w-6xl h-full flex flex-col">
              <div className="flex items-center justify-between px-6 py-4 rounded-t-2xl bg-dark-800/80 backdrop-blur-xl border border-white/10 border-b-0">
                 <div className="flex items-center gap-6">
                   <div className="flex items-center gap-3">
                     <div className="w-10 h-10 rounded-lg bg-brand-indigo/20 flex items-center justify-center text-brand-indigo border border-brand-indigo/30"><Database className="w-5 h-5" /></div>
                     <div><div className="text-sm font-bold text-white">{dataset.filename}</div><div className="text-xs text-slate-400 font-mono">{dataset.rows} Rows • {dataset.columns} Cols</div></div>
                   </div>
                   <div className="h-8 w-[1px] bg-white/10 mx-2" />
                   <div className="flex bg-dark-900/50 p-1 rounded-lg border border-white/5">
                      <button onClick={() => setActiveTab('config')} className={`px-4 py-1.5 rounded-md text-sm font-medium transition flex items-center gap-2 ${activeTab === 'config' ? 'bg-brand-indigo text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}><LayoutDashboard className="w-4 h-4" /> Configure</button>
                      <button onClick={() => setActiveTab('analysis')} className={`px-4 py-1.5 rounded-md text-sm font-medium transition flex items-center gap-2 ${activeTab === 'analysis' ? 'bg-brand-indigo text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}><BarChart2 className="w-4 h-4" /> Analysis</button>
                      <button onClick={() => setActiveTab('data')} className={`px-4 py-1.5 rounded-md text-sm font-medium transition flex items-center gap-2 ${activeTab === 'data' ? 'bg-brand-indigo text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}><TableIcon className="w-4 h-4" /> Data</button>
                   </div>
                 </div>
              </div>

              <div className="flex-1 bg-dark-800/50 backdrop-blur-md border border-white/10 rounded-b-2xl p-6 shadow-2xl overflow-y-auto relative z-20">
                {error && (
                  <div className="mb-6 p-4 bg-red-500/10 border border-red-500/50 rounded-xl flex items-center gap-3 text-red-200 animate-pulse">
                    <AlertCircle className="w-6 h-6 shrink-0" />
                    <div><h4 className="font-bold text-sm uppercase tracking-wider">Training Failed</h4><p className="text-sm opacity-80">{error}</p></div>
                    <button onClick={() => setError(null)} className="ml-auto hover:bg-red-500/20 p-1 rounded">✕</button>
                  </div>
                )}

                {results ? (
                  <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                     <CheckCircle className="w-16 h-16 text-green-500" />
                     <h2 className="text-3xl font-bold text-white">Training Complete!</h2>
                     <p className="text-slate-400">Model Accuracy: <span className="text-green-400 font-mono text-xl">{(results.metrics.accuracy * 100).toFixed(2)}%</span></p>
                     <button onClick={() => setResults(null)} className="mt-4 px-6 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-sm">Train Another Model</button>
                  </div>
                ) : activeTab === 'config' ? (
                  <ExperimentConfig dataset={dataset} onStartTraining={trainModel} />
                ) : activeTab === 'analysis' ? (
                  // --- NEW ANALYSIS TAB ---
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="p-4 rounded-xl bg-dark-900/50 border border-white/5">
                        <div className="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Total Rows</div>
                        <div className="text-2xl font-mono text-white">{dataset.rows}</div>
                      </div>
                      <div className="p-4 rounded-xl bg-dark-900/50 border border-white/5">
                        <div className="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Total Columns</div>
                        <div className="text-2xl font-mono text-white">{dataset.columns}</div>
                      </div>
                      <div className="p-4 rounded-xl bg-dark-900/50 border border-white/5">
                        <div className="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Missing Cells</div>
                        <div className="text-2xl font-mono text-brand-cyan">
                           {Object.values(dataset.missing_values || {}).reduce((a, b) => a + b, 0)}
                        </div>
                      </div>
                    </div>

                    <div className="border border-white/10 rounded-xl overflow-hidden">
                      <div className="px-6 py-3 bg-white/5 border-b border-white/5 flex justify-between items-center">
                         <h3 className="font-bold text-sm text-white flex items-center gap-2"><Activity className="w-4 h-4 text-brand-purple" /> Column Health Check</h3>
                         <span className="text-xs text-slate-500">Green = Good • Red = Missing</span>
                      </div>
                      <div className="bg-dark-900/30 divide-y divide-white/5 max-h-[400px] overflow-y-auto">
                        {dataset.column_names.map(col => {
                          const missingCount = dataset.missing_values?.[col] || 0;
                          const missingPct = (missingCount / dataset.rows) * 100;
                          return (
                            <div key={col} className="px-6 py-4 flex items-center gap-6 hover:bg-white/[0.02] transition">
                              <div className="w-1/3">
                                <div className="font-bold text-sm text-slate-200">{col}</div>
                                <div className="text-xs text-slate-500 font-mono">{dataset.dtypes[col]}</div>
                              </div>
                              <div className="flex-1">
                                <div className="flex justify-between text-xs mb-1.5">
                                  <span className={missingCount > 0 ? "text-red-400 font-bold" : "text-green-400 font-bold"}>
                                    {missingCount > 0 ? `${missingCount} Missing` : "Healthy"}
                                  </span>
                                  <span className="text-slate-500">{missingPct.toFixed(1)}%</span>
                                </div>
                                <div className="h-1.5 w-full bg-dark-900 rounded-full overflow-hidden flex">
                                   <div style={{ width: `${100 - missingPct}%` }} className="h-full bg-green-500/50" />
                                   <div style={{ width: `${missingPct}%` }} className="h-full bg-red-500" />
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4 h-full flex flex-col">
                    <h3 className="text-lg font-bold text-white flex items-center gap-2 shrink-0"><TableIcon className="w-5 h-5 text-brand-cyan" /> Dataset Preview</h3>
                    <div className="flex-1 overflow-auto rounded-xl border border-white/10 bg-dark-900/30">
                      <table className="w-full text-sm text-left text-slate-400">
                        <thead className="text-xs text-slate-200 uppercase bg-white/5 sticky top-0 backdrop-blur-sm">
                          <tr>{dataset.column_names.map((col) => <th key={col} className="px-6 py-3 whitespace-nowrap border-b border-white/10">{col}</th>)}</tr>
                        </thead>
                        <tbody>
                          {dataset.preview.map((row, i) => (
                            <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition group">
                              {dataset.column_names.map((col) => <td key={col} className="px-6 py-4 whitespace-nowrap font-mono text-xs group-hover:text-slate-200">{row[col] !== null ? row[col] : <span className="text-red-400 italic">NaN</span>}</td>)}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;