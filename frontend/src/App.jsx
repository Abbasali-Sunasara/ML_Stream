import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LayoutDashboard, Table as TableIcon, Activity, Layers, Cpu, Database, CheckCircle, AlertCircle, BarChart2, BrainCircuit, LineChart, AlertTriangle, Wand2 } from 'lucide-react';
import Plot from 'react-plotly.js';
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
          ctx.beginPath(); ctx.strokeStyle = 'rgba(255, 255, 255, 0.4 )'; ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
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
const Navbar = ({ hasDataset, workspaceMode, onReset, onSwitchMode }) => {
  return (
    <nav className="relative z-50 border-b border-white/5 bg-dark-900/50 backdrop-blur-md shrink-0">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-indigo to-brand-purple flex items-center justify-center shadow-lg shadow-brand-indigo/20">
              {workspaceMode === 'eda' ? <LineChart className="w-5 h-5 text-white" /> : <Activity className="w-5 h-5 text-white" />}
            </div>
            <div className="text-left">
               <span className="block text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-brand-indigo via-brand-purple to-brand-cyan">
                 {workspaceMode === 'eda' ? 'EDA Dashboard' : 'ML Studio'}
               </span>
            </div>
        </div>
        {hasDataset && (
           <div className="flex items-center gap-3">
              {workspaceMode === 'ml' ? (
                 <button onClick={() => onSwitchMode('eda')} className="text-xs font-bold text-brand-cyan hover:text-white transition px-4 py-2 bg-brand-cyan/10 hover:bg-brand-cyan/20 rounded-lg border border-brand-cyan/20 flex items-center gap-2">
                    <LineChart className="w-4 h-4" /> GO TO DASHBOARD
                 </button>
              ) : (
                 <button onClick={() => onSwitchMode('ml')} className="text-xs font-bold text-brand-purple hover:text-white transition px-4 py-2 bg-brand-purple/10 hover:bg-brand-purple/20 rounded-lg border border-brand-purple/20 flex items-center gap-2">
                    <BrainCircuit className="w-4 h-4" /> GO TO MODEL BUILDER
                 </button>
              )}
              <button onClick={onReset} className="text-xs font-bold text-red-400 hover:text-red-300 transition px-4 py-2 hover:bg-red-500/10 rounded-lg border border-red-500/20">
                 {workspaceMode === 'eda' ? 'EXIT DASHBOARD' : 'EXIT STUDIO'}
              </button>
           </div>
        )}
      </div>
    </nav>
  );
};

function App() {
  const [dataset, setDataset] = useState(null);
  const [activeTab, setActiveTab] = useState('config'); 
  const [workspaceMode, setWorkspaceMode] = useState('ml'); 
  
  // ML States
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  // --- NEW: EDA DASHBOARD STATES ---
  const [edaChartType, setEdaChartType] = useState('scatter');
  const [edaX, setEdaX] = useState('');
  const [edaY, setEdaY] = useState('');
  const [edaPlotData, setEdaPlotData] = useState(null);
  const [edaIsDrawing, setEdaIsDrawing] = useState(false);
  const [edaWarning, setEdaWarning] = useState(null);

  useEffect(() => {
    if (dataset && !edaX) {
      setEdaX(dataset.column_names[0]);
      if (dataset.column_names.length > 1) setEdaY(dataset.column_names[1]);
    }
  }, [dataset, workspaceMode]);

  // --- SMART AI BRAIN (Validation Logic) ---
  useEffect(() => {
    if (!dataset || !edaX) return;
    setEdaWarning(null); // Reset warnings

    const typeX = dataset.dtypes[edaX];
    const typeY = edaY ? dataset.dtypes[edaY] : null;
    const isNumX = typeX?.includes('int') || typeX?.includes('float');
    const isNumY = typeY?.includes('int') || typeY?.includes('float');

    if (edaChartType === 'scatter' && (!isNumX || !isNumY)) {
      setEdaWarning({
        title: "Scatter Plot Error",
        msg: "Scatter plots require continuous numbers for both X and Y. You selected text/categorical data.",
        suggestion: "bar",
        fixText: "Auto-Fix: Switch to Bar Chart"
      });
    } else if (edaChartType === 'histogram' && !isNumX) {
      setEdaWarning({
         title: "Distribution Error",
         msg: "Histograms require numeric data to calculate bins. You selected text.",
         suggestion: "bar",
         fixText: "Auto-Fix: Switch to Bar Chart"
      });
    } else if (edaChartType === 'box' && !isNumX) {
      setEdaWarning({
         title: "Box Plot Error",
         msg: "Box plots map numeric outliers. You selected text.",
         suggestion: "bar",
         fixText: "Auto-Fix: Switch to Bar Chart"
      });
    }
  }, [edaChartType, edaX, edaY, dataset]);

  // --- THE FETCH FUNCTION (Talking to Python) ---
  const generatePlot = async () => {
    if (edaWarning) return;
    setEdaIsDrawing(true);
    setError(null);
    try {
      const response = await fetch('http://127.0.0.1:5000/plot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: edaChartType, x: edaX, y: edaY })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Plot generation failed");
      setEdaPlotData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setEdaIsDrawing(false);
    }
  };

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
      <Navbar 
         hasDataset={!!dataset} 
         workspaceMode={workspaceMode}
         onReset={() => { setDataset(null); setResults(null); setActiveTab('config'); setEdaPlotData(null); }} 
         onSwitchMode={(mode) => { setWorkspaceMode(mode); setActiveTab('config'); }}
      />

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

      <main className="relative z-10 flex-1 flex flex-col items-center justify-center p-6 overflow-hidden w-full">
        <AnimatePresence mode="wait">
          {!dataset ? (
            /* =========================================
               LANDING PAGE
               ========================================= */
            <motion.div key="landing" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} className="w-full max-w-6xl flex flex-col md:flex-row items-center gap-16">
              <div className="flex-1 space-y-8 text-center md:text-left z-20">
                <div className="space-y-2">
                  <h1 className="text-6xl md:text-8xl font-bold tracking-tighter text-white leading-[1] drop-shadow-2xl">ML <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-indigo to-brand-purple">Studio</span></h1>
                  <div className="text-xl md:text-2xl text-slate-400 h-8"><TypewriterText text="> System Online. Awaiting Data..." /></div>
                </div>
                <p className="text-lg text-slate-500 max-w-xl mx-auto md:mx-0 leading-relaxed">The comprehensive platform for machine learning development. Import datasets, visually profile distributions, and train professional models.</p>
                <div className="flex flex-wrap justify-center md:justify-start gap-4">
                  {[{ icon: Layers, text: "Auto-Pipeline", color: "text-brand-cyan" }, { icon: Cpu, text: "Scikit Engine", color: "text-brand-purple" }, { icon: Database, text: "Data Analytics", color: "text-brand-indigo" }].map((feat, i) => (
                    <div key={i} className="flex items-center gap-2 px-5 py-3 rounded-full bg-white/5 border border-white/10 backdrop-blur-md cursor-default transition-colors hover:bg-white/10">
                      <feat.icon className={`w-4 h-4 ${feat.color}`} /> <span className="text-sm font-semibold text-slate-300">{feat.text}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="flex-1 w-full max-w-md relative flex flex-col items-center">
                 <div className="flex bg-dark-900/80 p-1.5 rounded-xl border border-white/10 w-fit mb-6 relative z-20 backdrop-blur-md shadow-2xl">
                    <button onClick={() => setWorkspaceMode('ml')} className={`px-6 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center gap-2 ${workspaceMode === 'ml' ? 'bg-gradient-to-r from-brand-indigo to-brand-purple text-white shadow-lg' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}>
                      <BrainCircuit className="w-4 h-4" /> ML Engine
                    </button>
                    <button onClick={() => setWorkspaceMode('eda')} className={`px-6 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center gap-2 ${workspaceMode === 'eda' ? 'bg-gradient-to-r from-brand-cyan to-brand-indigo text-white shadow-lg' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}>
                      <LineChart className="w-4 h-4" /> EDA Dashboard
                    </button>
                 </div>
                 <div className="absolute inset-0 top-16 bg-brand-indigo/10 blur-[100px] rounded-full pointer-events-none transform-gpu" />
                 <div className="relative w-full rounded-2xl border border-white/10 bg-white/[0.02] backdrop-blur-md shadow-2xl p-1">
                    <UploadDataset onUploadComplete={(data) => setDataset(data)} />
                 </div>
              </div>
            </motion.div>

          ) : workspaceMode === 'eda' ? (
            /* =========================================
               EDA DASHBOARD WORKSPACE (Tableau Style)
               ========================================= */
            <motion.div key="eda-workspace" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="w-full max-w-7xl h-full flex flex-col">
              <div className="flex items-center justify-between px-6 py-4 rounded-t-2xl bg-dark-800/80 backdrop-blur-xl border border-white/10 border-b-0">
                 <h2 className="text-xl font-bold text-white flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-brand-cyan/20 flex items-center justify-center border border-brand-cyan/30"><LineChart className="w-4 h-4 text-brand-cyan" /></div>
                    Visualization Studio
                 </h2>
                 <div className="text-right"><div className="text-sm font-bold text-white">{dataset.filename}</div></div>
              </div>

              <div className="flex-1 bg-dark-800/50 backdrop-blur-md border border-white/10 rounded-b-2xl p-6 shadow-2xl flex gap-6 overflow-hidden">
                 
                 {/* LEFT SIDEBAR: Controls */}
                 <div className="w-72 shrink-0 flex flex-col gap-6">
                    <div className="space-y-4">
                       <div>
                         <label className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block">Chart Type</label>
                         <select value={edaChartType} onChange={(e) => {setEdaChartType(e.target.value); setEdaPlotData(null);}} className="w-full bg-dark-900 border border-white/10 text-white text-sm rounded-lg px-4 py-2.5 outline-none focus:border-brand-cyan transition-colors">
                            <option value="scatter">Scatter Plot</option>
                            <option value="bar">Bar Chart</option>
                            <option value="histogram">Histogram (Distribution)</option>
                            <option value="box">Box Plot (Outliers)</option>
                            <option value="heatmap">Correlation Heatmap</option>
                         </select>
                       </div>

                       {/* DYNAMIC UI: Heatmap needs NO dropdowns. Everything else needs at least X. */}
                       {edaChartType !== 'heatmap' && (
                         <div>
                           <label className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block">
                              {/* SMART LABEL: If it's a Box or Hist, it only needs 1 column, so call it "Target Feature" */}
                              {edaChartType === 'histogram' || edaChartType === 'box' || edaChartType === 'bar' ? 'Target Feature' : 'X-Axis'}
                           </label>
                           <select value={edaX} onChange={(e) => {setEdaX(e.target.value); setEdaPlotData(null);}} className="w-full bg-dark-900 border border-white/10 text-white text-sm rounded-lg px-4 py-2.5 outline-none focus:border-brand-cyan transition-colors">
                              {dataset.column_names.map(col => <option key={col} value={col}>{col} ({dataset.dtypes[col]})</option>)}
                           </select>
                         </div>
                       )}

                       {/* DYNAMIC UI: ONLY Scatter Plot gets the Y-Axis dropdown! */}
                       {edaChartType === 'scatter' && (
                         <div>
                           <label className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block">Y-Axis</label>
                           <select value={edaY} onChange={(e) => {setEdaY(e.target.value); setEdaPlotData(null);}} className="w-full bg-dark-900 border border-white/10 text-white text-sm rounded-lg px-4 py-2.5 outline-none focus:border-brand-cyan transition-colors">
                              {dataset.column_names.map(col => <option key={col} value={col}>{col} ({dataset.dtypes[col]})</option>)}
                           </select>
                         </div>
                       )}
                    </div>

                    <button 
                       onClick={generatePlot}
                       disabled={!!edaWarning || edaIsDrawing}
                       className="w-full mt-auto bg-brand-cyan text-dark-900 font-bold py-3 rounded-lg shadow-[0_0_15px_rgba(34,211,238,0.4)] hover:shadow-[0_0_25px_rgba(34,211,238,0.6)] hover:bg-cyan-300 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                       {edaIsDrawing ? <Cpu className="w-5 h-5 animate-spin" /> : <LineChart className="w-5 h-5" />}
                       {edaIsDrawing ? 'Rendering Engine...' : 'Generate Plot'}
                    </button>
                 </div>

                 {/* RIGHT CANVAS: Plotly Engine */}
                 <div className="flex-1 bg-dark-900/50 border border-white/5 rounded-xl relative overflow-hidden flex items-center justify-center">
                    {error && (
                      <div className="absolute top-4 left-4 right-4 p-4 bg-red-500/10 border border-red-500/50 rounded-xl flex items-center gap-3 text-red-200 z-50">
                        <AlertCircle className="w-6 h-6 shrink-0" />
                        <div><h4 className="font-bold text-sm">Python Engine Error</h4><p className="text-sm opacity-80">{error}</p></div>
                        <button onClick={() => setError(null)} className="ml-auto hover:bg-red-500/20 p-1 rounded">✕</button>
                      </div>
                    )}

                    {edaWarning ? (
                      <div className="max-w-md w-full p-8 border border-yellow-500/30 bg-yellow-500/5 rounded-2xl flex flex-col items-center text-center animate-in fade-in zoom-in-95 duration-300">
                         <div className="w-16 h-16 rounded-full bg-yellow-500/10 flex items-center justify-center border border-yellow-500/20 mb-4 shadow-[0_0_30px_rgba(234,179,8,0.2)]">
                            <AlertTriangle className="w-8 h-8 text-yellow-500" />
                         </div>
                         <h3 className="text-xl font-bold text-white mb-2">{edaWarning.title}</h3>
                         <p className="text-slate-400 text-sm mb-6 leading-relaxed">{edaWarning.msg}</p>
                         <button 
                            onClick={() => { setEdaChartType(edaWarning.suggestion); setEdaWarning(null); setEdaPlotData(null); }}
                            className="w-full flex items-center justify-center gap-2 bg-yellow-500/20 hover:bg-yellow-500/30 border border-yellow-500/50 text-yellow-400 font-bold py-3 rounded-xl transition-colors"
                         >
                            <Wand2 className="w-5 h-5" /> {edaWarning.fixText}
                         </button>
                      </div>
                    ) : edaPlotData ? (
                      <Plot
                         data={edaPlotData.data}
                         layout={{ 
                            ...edaPlotData.layout, 
                            paper_bgcolor: 'transparent', 
                            plot_bgcolor: 'transparent',
                            font: { color: '#94a3b8', family: 'Inter, sans-serif' },
                            margin: { t: 60, r: 40, b: 60, l: 60 },
                            xaxis: { ...edaPlotData.layout.xaxis, gridcolor: '#1e293b', zerolinecolor: '#334155' },
                            yaxis: { ...edaPlotData.layout.yaxis, gridcolor: '#1e293b', zerolinecolor: '#334155' }
                         }}
                         useResizeHandler={true}
                         style={{ width: '100%', height: '100%' }}
                         config={{ displayModeBar: true, responsive: true }}
                      />
                    ) : (
                      <div className="text-center flex flex-col items-center opacity-50">
                         <LineChart className="w-16 h-16 text-slate-500 mb-4" />
                         <p className="text-slate-400 text-lg font-mono">Select configuration and press Generate Plot.</p>
                      </div>
                    )}
                 </div>
              </div>
            </motion.div>

          ) : (
            /* =========================================
               ML STUDIO WORKSPACE (Data Health is here!)
               ========================================= */
            <motion.div key="ml-workspace" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="w-full max-w-6xl h-full flex flex-col">
              <div className="flex items-center justify-between px-6 py-4 rounded-t-2xl bg-dark-800/80 backdrop-blur-xl border border-white/10 border-b-0">
                 <div className="flex items-center gap-6">
                   <div className="flex items-center gap-3">
                     <div className="w-10 h-10 rounded-lg bg-brand-indigo/20 flex items-center justify-center text-brand-indigo border border-brand-indigo/30"><Database className="w-5 h-5" /></div>
                     <div><div className="text-sm font-bold text-white">{dataset.filename}</div><div className="text-xs text-slate-400 font-mono">{dataset.rows} Rows • {dataset.columns} Cols</div></div>
                   </div>
                   <div className="h-8 w-[1px] bg-white/10 mx-2" />
                   {!results && (
                      <div className="flex bg-dark-900/50 p-1 rounded-lg border border-white/5">
                        <button onClick={() => setActiveTab('config')} className={`px-4 py-1.5 rounded-md text-sm font-medium transition flex items-center gap-2 ${activeTab === 'config' ? 'bg-brand-indigo text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}><LayoutDashboard className="w-4 h-4" /> Configure</button>
                        <button onClick={() => setActiveTab('health')} className={`px-4 py-1.5 rounded-md text-sm font-medium transition flex items-center gap-2 ${activeTab === 'health' ? 'bg-brand-indigo text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}><Activity className="w-4 h-4" /> Data Health</button>
                        <button onClick={() => setActiveTab('data')} className={`px-4 py-1.5 rounded-md text-sm font-medium transition flex items-center gap-2 ${activeTab === 'data' ? 'bg-brand-indigo text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}><TableIcon className="w-4 h-4" /> Data Preview</button>
                      </div>
                   )}
                 </div>
              </div>

              <div className="flex-1 bg-dark-800/50 backdrop-blur-md border border-white/10 rounded-b-2xl p-6 shadow-2xl overflow-y-auto relative z-20 flex flex-col">
                {results ? (
                  /* --- ML RESULTS VIEW --- */
                  <div className="flex flex-col w-full h-full max-w-6xl mx-auto py-2">
                     <div className="flex flex-col sm:flex-row items-center justify-between bg-dark-900/60 p-5 rounded-2xl border border-white/10 shadow-lg mb-6 w-full shrink-0">
                        <div className="flex items-center gap-4 mb-4 sm:mb-0">
                          <div className="bg-green-500/10 p-3 rounded-full border border-green-500/20"><CheckCircle className="w-7 h-7 text-green-500" /></div>
                          <div className="text-left"><h2 className="text-xl font-bold text-white leading-tight">Model Trained Successfully</h2><p className="text-slate-400 text-sm">Automated pipeline execution complete.</p></div>
                        </div>
                        <div className="flex items-center gap-4 bg-dark-800/80 px-6 py-3 rounded-xl border border-white/5">
                           {results.metrics.accuracy !== undefined ? (
                               <><span className="text-slate-400 text-sm uppercase tracking-wider font-bold">Overall Accuracy</span><span className="text-green-400 font-mono text-2xl font-bold">{(results.metrics.accuracy * 100).toFixed(1)}%</span></>
                           ) : (
                               <><span className="text-slate-400 text-sm uppercase tracking-wider font-bold">R² Score (Model Fit)</span><span className="text-brand-cyan font-mono text-2xl font-bold">{(results.metrics.r2 * 100).toFixed(1)}%</span></>
                           )}
                        </div>
                     </div>

                     <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full flex-1 min-h-0 overflow-y-auto">
                       {results.metrics.confusion_matrix ? (
                          <div className="bg-dark-900/50 p-6 rounded-xl border border-white/5 shadow-xl flex flex-col">
                             <h3 className="text-xs font-bold text-brand-purple mb-6 uppercase tracking-wider flex items-center gap-2"><Activity className="w-4 h-4" /> Confusion Matrix</h3>
                             <div className="flex flex-col items-center justify-center flex-1">
                                <div className="flex mb-2 text-[10px] font-bold text-slate-500 w-full justify-center pl-10">
                                   <span className="w-16 text-center uppercase">Pred<br/><span className="text-white text-xs">{results.metrics.classes?.[0] || '0'}</span></span>
                                   <span className="w-16 text-center uppercase">Pred<br/><span className="text-white text-xs">{results.metrics.classes?.[1] || '1'}</span></span>
                                </div>
                                <div className="flex items-center">
                                   <div className="flex flex-col mr-3 text-[10px] font-bold text-slate-500 text-right uppercase">
                                      <span className="h-16 flex flex-col justify-center">Act<br/><span className="text-white text-xs">{results.metrics.classes?.[0] || '0'}</span></span>
                                      <span className="h-16 flex flex-col justify-center">Act<br/><span className="text-white text-xs">{results.metrics.classes?.[1] || '1'}</span></span>
                                   </div>
                                   <div className="grid grid-cols-2 gap-2">
                                      {results.metrics.confusion_matrix.map((row, rowIndex) => (
                                         row.map((val, colIndex) => {
                                            const isCorrect = rowIndex === colIndex;
                                            return (
                                               <div key={`${rowIndex}-${colIndex}`} className={`w-16 h-16 flex flex-col items-center justify-center rounded-xl border transition-transform hover:scale-105 ${isCorrect ? 'bg-green-500/10 border-green-500/30 text-green-400 shadow-[0_0_10px_rgba(34,197,94,0.1)]' : 'bg-red-500/10 border-red-500/30 text-red-400 shadow-[0_0_10px_rgba(239,68,68,0.1)]'}`}>
                                                  <span className="text-2xl font-bold font-mono">{val}</span>
                                               </div>
                                            );
                                         })
                                      ))}
                                   </div>
                                </div>
                             </div>
                          </div>
                       ) : results.metrics.rmse !== undefined ? (
                          <div className="bg-dark-900/50 p-6 rounded-xl border border-white/5 shadow-xl flex flex-col justify-center items-center text-center">
                             <h3 className="text-xs font-bold text-brand-purple mb-6 uppercase tracking-wider flex items-center gap-2"><Activity className="w-4 h-4" /> Regression Error</h3>
                             <div className="flex flex-col items-center justify-center flex-1">
                                 <div className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Root Mean Square Error (RMSE)</div>
                                 <div className="text-5xl font-mono font-bold text-red-400 mt-2">{results.metrics.rmse.toFixed(2)}</div>
                                 <div className="text-[11px] text-slate-500 mt-4 px-4">This means your AI's predictions are off by an average of ± {results.metrics.rmse.toFixed(2)} units.</div>
                             </div>
                          </div>
                       ) : null}

                       {results.metrics.feature_importance && results.metrics.feature_importance.length > 0 && (
                          <div className="bg-dark-900/50 p-6 rounded-xl border border-white/5 shadow-xl flex flex-col overflow-hidden">
                             <h3 className="text-xs font-bold text-brand-indigo mb-6 uppercase tracking-wider flex items-center gap-2 shrink-0"><BarChart2 className="w-4 h-4" /> Feature Importance</h3>
                             <div className="space-y-4 flex-1 overflow-y-auto pr-2">
                                {results.metrics.feature_importance.map((feat, idx) => {
                                   const maxImp = results.metrics.feature_importance[0].importance;
                                   const pct = (feat.importance / maxImp) * 100;
                                   return (
                                      <div key={feat.feature} className="flex items-center gap-3 group">
                                         <div className="w-16 text-right text-[10px] font-bold text-slate-300 uppercase truncate group-hover:text-brand-cyan transition-colors">{feat.feature}</div>
                                         <div className="flex-1 h-2.5 bg-dark-800 rounded-full overflow-hidden border border-white/5 relative">
                                            <motion.div initial={{ width: 0 }} animate={{ width: `${pct}%` }} transition={{ duration: 1, delay: idx * 0.1, ease: "easeOut" }} className="absolute top-0 left-0 h-full bg-gradient-to-r from-brand-indigo to-brand-cyan rounded-full shadow-[0_0_10px_rgba(34,211,238,0.5)]" />
                                         </div>
                                         <div className="w-12 text-left text-[10px] font-mono text-slate-400 group-hover:text-white transition-colors">{(feat.importance * 100).toFixed(1)}%</div>
                                      </div>
                                   )
                                })}
                             </div>
                          </div>
                       )}

                       <div className="bg-dark-900/50 p-6 rounded-xl border border-white/5 shadow-xl flex flex-col justify-center">
                           <h3 className="text-xs font-bold text-brand-cyan mb-6 uppercase tracking-wider flex items-center gap-2"><Cpu className="w-4 h-4" /> Export & Deploy</h3>
                           <a href="http://127.0.0.1:5000/download" download className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-brand-cyan/20 to-brand-indigo/20 hover:from-brand-cyan/30 hover:to-brand-indigo/30 text-white border border-brand-cyan/30 rounded-xl text-sm font-bold transition-all shadow-[0_0_15px_rgba(34,211,238,0.1)] hover:-translate-y-1">
                              <Database className="w-4 h-4 text-brand-cyan" /> Download (.pkl)
                           </a>
                           <p className="text-[10px] text-slate-500 mt-3 text-center">Deploy these compiled weights into any Python web server.</p>
                           <div className="w-full h-[1px] bg-white/10 my-4"></div>
                           <button onClick={() => setResults(null)} className="w-full px-4 py-2 text-slate-400 hover:text-white border border-white/10 hover:bg-white/5 rounded-xl text-xs transition-all font-semibold">
                              ← Retrain Model
                           </button>
                       </div>
                     </div>
                  </div>

                ) : activeTab === 'config' ? (
                  <ExperimentConfig dataset={dataset} onStartTraining={trainModel} />
                  
                ) : activeTab === 'health' ? (
                  /* =========================================
                     DATA HEALTH SCREEN (Moved to ML Studio)
                     ========================================= */
                  <div className="flex flex-col h-full w-full">
                    <div className="flex-1 min-h-0 overflow-y-auto w-full pr-2">
                      <div className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div className="p-4 rounded-xl bg-dark-900/50 border border-white/5"><div className="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Total Rows</div><div className="text-2xl font-mono text-white">{dataset.rows}</div></div>
                          <div className="p-4 rounded-xl bg-dark-900/50 border border-white/5"><div className="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Total Columns</div><div className="text-2xl font-mono text-white">{dataset.columns}</div></div>
                          <div className="p-4 rounded-xl bg-dark-900/50 border border-white/5"><div className="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Missing Cells</div><div className="text-2xl font-mono text-brand-cyan">{Object.values(dataset.missing_values || {}).reduce((a, b) => a + b, 0)}</div></div>
                        </div>
                        <div className="border border-white/10 rounded-xl overflow-hidden">
                          <div className="px-6 py-4 bg-white/5 border-b border-white/5 flex justify-between items-center"><h3 className="font-bold text-sm text-white flex items-center gap-2"><Activity className="w-4 h-4 text-brand-indigo" /> Column Health Report</h3></div>
                          <div className="bg-dark-900/30 divide-y divide-white/5">
                            {dataset.column_names.map(col => {
                              const missingCount = dataset.missing_values?.[col] || 0;
                              const missingPct = (missingCount / dataset.rows) * 100;
                              return (
                                <div key={col} className="px-6 py-4 flex items-center gap-6 hover:bg-white/[0.02] transition">
                                  <div className="w-1/3"><div className="font-bold text-sm text-slate-200">{col}</div><div className="text-xs text-slate-500 font-mono">{dataset.dtypes[col]}</div></div>
                                  <div className="flex-1">
                                    <div className="flex justify-between text-xs mb-1.5"><span className={missingCount > 0 ? "text-red-400 font-bold" : "text-green-400 font-bold"}>{missingCount > 0 ? `${missingCount} Missing` : "Healthy"}</span><span className="text-slate-500">{missingPct.toFixed(1)}%</span></div>
                                    <div className="h-1.5 w-full bg-dark-900 rounded-full overflow-hidden flex"><div style={{ width: `${100 - missingPct}%` }} className="h-full bg-green-500/50" /><div style={{ width: `${missingPct}%` }} className="h-full bg-red-500" /></div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                ) : (
                  /* DATA PREVIEW SCREEN */
                  <div className="space-y-4 h-full flex flex-col w-full">
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