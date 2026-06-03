import { useState, useEffect } from 'react';
import { Activity, BarChart, Split, Play, CheckCircle2, Wand2 } from 'lucide-react';

const ALGORITHMS = {
  classification: [
    { id: 'random_forest', name: 'Random Forest Classifier' },
    { id: 'decision_tree', name: 'Decision Tree Classifier' }, // NEW
    { id: 'adaboost', name: 'AdaBoost Classifier' },           // NEW
    { id: 'xgboost', name: 'Gradient Boosting (Hist)' },
    { id: 'logistic_regression', name: 'Logistic Regression' },
    { id: 'naive_bayes', name: 'Naive Bayes' },                // NEW
    { id: 'knn', name: 'K-Nearest Neighbors (KNN)' },
    { id: 'svm', name: 'Support Vector Machine (SVM)' }
  ],
  regression: [
    { id: 'random_forest', name: 'Random Forest Regressor' },
    { id: 'decision_tree', name: 'Decision Tree Regressor' }, // NEW
    { id: 'adaboost', name: 'AdaBoost Regressor' },           // NEW
    { id: 'xgboost', name: 'Gradient Boosting (Hist)' },
    { id: 'linear_regression', name: 'Linear Regression' },
    { id: 'knn', name: 'K-Nearest Neighbors (KNN)' },
    { id: 'svr', name: 'Support Vector Regressor (SVR)' }
  ]
};

const ALGO_PARAMS = {
  logistic_regression: [
    { name: 'C', label: 'Regularization (C)', type: 'slider', min: 0.01, max: 10, step: 0.1, default: 1.0 },
    { name: 'penalty', label: 'Penalty', type: 'dropdown', options: ['l1', 'l2'], default: 'l2' }
  ],
  svm: [
    { name: 'C', label: 'Penalty (C)', type: 'slider', min: 0.1, max: 10, step: 0.1, default: 1.0 },
    { name: 'kernel', label: 'Kernel Type', type: 'dropdown', options: ['linear', 'rbf', 'poly'], default: 'rbf' }
  ],
  svr: [
    { name: 'C', label: 'Penalty (C)', type: 'slider', min: 0.1, max: 10, step: 0.1, default: 1.0 },
    { name: 'kernel', label: 'Kernel Type', type: 'dropdown', options: ['linear', 'rbf', 'poly'], default: 'rbf' }
  ],
  knn: [
    { name: 'n_neighbors', label: 'K-Neighbors', type: 'slider', min: 1, max: 30, step: 1, default: 5 },
    { name: 'weights', label: 'Weight Function', type: 'dropdown', options: ['uniform', 'distance'], default: 'uniform' }
  ],
  decision_tree: [
    { name: 'max_depth', label: 'Max Depth', type: 'slider', min: 1, max: 50, step: 1, default: 10 },
    { name: 'min_samples_split', label: 'Min Samples to Split', type: 'slider', min: 2, max: 20, step: 1, default: 2 }
  ],
  adaboost: [
    { name: 'n_estimators', label: 'Estimators', type: 'slider', min: 10, max: 200, step: 10, default: 50 },
    { name: 'learning_rate', label: 'Learning Rate', type: 'slider', min: 0.01, max: 2.0, step: 0.1, default: 1.0 }
  ],
  naive_bayes: [
    { name: 'var_smoothing', label: 'Variance Smoothing', type: 'slider', min: 0.000000001, max: 0.0000001, step: 0.000000001, default: 0.000000001 }
  ],
  xgboost: [
    { name: 'learning_rate', label: 'Learning Rate', type: 'slider', min: 0.01, max: 0.5, step: 0.01, default: 0.1 },
    { name: 'max_iter', label: 'Max Iterations', type: 'slider', min: 10, max: 300, step: 10, default: 100 }
  ],
  random_forest: [
    { name: 'n_estimators', label: 'Estimators', type: 'slider', min: 10, max: 500, step: 10, default: 100 },
    { name: 'max_depth', label: 'Max Depth', type: 'slider', min: 1, max: 50, step: 1, default: 10 },
    { name: 'min_samples_split', label: 'Min Samples to Split', type: 'slider', min: 2, max: 10, step: 1, default: 2 }
  ],
  linear_regression: [
    { name: 'fit_intercept', label: 'Fit Intercept', type: 'dropdown', options: ['True', 'False'], default: 'True' }
  ]
};

export default function ExperimentConfig({ dataset, onStartTraining }) {
  const [targetCol, setTargetCol] = useState('');
  const [taskType, setTaskType] = useState('classification');
  const [missingValue, setMissingValue] = useState('drop');
  const [scaling, setScaling] = useState('standard');
  const [splitRatio, setSplitRatio] = useState(0.8);
  const [algorithm, setAlgorithm] = useState('');
// --- PRO MODE STATES ---
  const [isProMode, setIsProMode] = useState(false);
  const [hp, setHp] = useState({
    n_estimators: 100,
    max_depth: 10,
    n_neighbors: 5,
    learning_rate: 0.1
  });

  // ----- Auto Tune States---- //
  const [isAutoTune, setIsAutoTune] = useState(false);
  const [searchIntensity, setSearchIntensity] = useState(10); // Number of iterations

  // ----- Data Healing States -----
  const [outlierStrategy, setOutlierStrategy] = useState('none');
  const [imbalanceStrategy, setImbalanceStrategy] = useState('none');

  // Helper to update specific hyperparams
  const numericHpKeys = ['n_estimators', 'max_depth', 'n_neighbors', 'min_samples_split', 'max_iter', 'learning_rate', 'C', 'gamma', 'var_smoothing', 'epsilon'];
  const updateHp = (key, val) => {
    const nextValue = numericHpKeys.includes(key) ? Number(val) : val;
    setHp(prev => ({ ...prev, [key]: nextValue }));
  };
  
  // RESTORED/FIXED: Helper to reset hyperparameters to defaults
  const resetToDefaults = (algo) => {
    const defaults = {};
    ALGO_PARAMS[algo]?.forEach(p => {
      defaults[p.name] = p.default;
    });
    setHp(defaults);
  };

  // FIXED: Mutually Exclusive Toggles
  const toggleProMode = () => {
    const nextState = !isProMode;
    setIsProMode(nextState);
    if (nextState) {
      setIsAutoTune(false);
    }
    resetToDefaults(algorithm);
  };

  const toggleAutoTune = () => {
    const nextState = !isAutoTune;
    setIsAutoTune(nextState);
    if (nextState) {
      setIsProMode(false);
    }
    resetToDefaults(algorithm);
  };
  
  
  
  // Watchdog 1: Smart Auto-detect Task Type
  useEffect(() => {
    if (!targetCol) return;
    
    // Check if column is text (object)
    const isText = dataset.dtypes[targetCol] === 'object';
    // Check if column is numeric but has very few unique values (could be a category)
    // We get this recommendation hint from the backend upload logic
    const isNumeric = dataset.dtypes[targetCol]?.includes('int') || dataset.dtypes[targetCol]?.includes('float');

    if (isText) {
      setTaskType('classification');
    } else if (isNumeric) {
      // If it's a number, usually strength/price data is Regression
      // We flip to regression if it's a standard numeric data science task
      setTaskType('regression');
    }
  }, [targetCol, dataset]);

  // Watchdog 2: Sync algorithm selection when task type changes
  useEffect(() => {
    setAlgorithm(ALGORITHMS[taskType][0].id);
  }, [taskType]);

  // Watchdog 3: Smart Scaling for Tree/Boosting Models
  useEffect(() => {
    // Added decision_tree and adaboost here
    const isTreeModel = ['random_forest', 'xgboost', 'decision_tree', 'adaboost'].includes(algorithm);
    
    if (isTreeModel) {
      setScaling('none');
    } else if (scaling === 'none') {
      setScaling('standard');
    }
  }, [algorithm]);

  // Watchdog 4: Populate default hyperparameters when algorithm changes
  useEffect(() => {
    if (!algorithm || !ALGO_PARAMS[algorithm]) return;

    const defaults = {};
    ALGO_PARAMS[algorithm].forEach(p => {
      defaults[p.name] = p.default;
    });
    
    // This ensures 'hp' always has the correct keys for the selected algo
    setHp(defaults);
  }, [algorithm]);

  const handleStart = () => {
    if (!targetCol) return alert("Please select a target column first!");
    
    onStartTraining({
      filename: dataset.filename,
      target: targetCol,
      task_type: taskType,
      algorithm: algorithm, 
      auto_tune: isAutoTune,
      search_intensity: searchIntensity,
      hyperparameters: isProMode && !isAutoTune ? hp : null,
      preprocessing: {
        missing_value_strategy: missingValue,
        scaling: scaling,
        test_size: parseFloat((1 - splitRatio).toFixed(2)),
        healing: {
          outliers: outlierStrategy,
          imbalance: imbalanceStrategy
        }
      }
    });
  };

  return (
    <div className="bg-dark-800/80 backdrop-blur-xl border border-white/10 rounded-b-xl p-6 shadow-2xl">
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 divide-y md:divide-y-0 md:divide-x divide-white/5">
        
        {/* LEFT: Model Setup */}
        <div className="space-y-5 md:pr-4">
          <div className="flex items-center gap-2 text-brand-purple mb-1">
            <Activity className="w-4 h-4" />
            <h3 className="text-sm font-bold uppercase tracking-wider">Target & Task</h3>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Target Column</label>
              <div className="relative">
                <select 
                  value={targetCol}
                  onChange={(e) => setTargetCol(e.target.value)}
                  className="w-full appearance-none bg-dark-900/50 border border-slate-700 hover:border-brand-indigo/50 rounded-lg px-4 py-2.5 text-slate-200 focus:ring-1 focus:ring-brand-indigo outline-none transition text-sm"
                >
                  <option value="" className="bg-dark-800 text-slate-200">Select a column to predict...</option>
                  {dataset.column_names.map(col => (
                    <option key={col} value={col} className="bg-dark-800 text-slate-200">
                      {col}
                    </option>
                  ))}
                </select>
                <div className="absolute right-3 top-3 pointer-events-none">
                  <div className="w-0 h-0 border-l-[4px] border-l-transparent border-r-[4px] border-r-transparent border-t-[4px] border-t-slate-500"></div>
                </div>
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Task Type</label>
              <div className="grid grid-cols-2 gap-2">
                {['classification', 'regression'].map((type) => (
                  <button
                    key={type}
                    onClick={() => setTaskType(type)}
                    className={`
                      relative py-2 px-3 rounded-lg text-sm font-medium transition-all border
                      ${taskType === type 
                        ? 'bg-brand-indigo/10 border-brand-indigo text-brand-indigo' 
                        : 'bg-dark-900/30 border-transparent text-slate-500 hover:text-slate-300 hover:bg-white/5'}
                    `}
                  >
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                    {taskType === type && <CheckCircle2 className="w-3.5 h-3.5 absolute top-1 right-1 opacity-50" />}
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-2">
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Select Algorithm</label>
              <div className="relative">
                <select 
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value)}
                  className="w-full appearance-none bg-dark-900/50 border border-slate-700 hover:border-brand-indigo/50 rounded-lg px-4 py-2.5 text-slate-200 focus:ring-1 focus:ring-brand-indigo outline-none transition text-sm"
                >
                  {ALGORITHMS[taskType].map((algo) => (
                    <option key={algo.id} value={algo.id} className="bg-dark-800 text-slate-200">
                      {algo.name}
                    </option>
                  ))}
                </select>
                <div className="absolute right-3 top-3 pointer-events-none">
                  <div className="w-0 h-0 border-l-[4px] border-l-transparent border-r-[4px] border-r-transparent border-t-[4px] border-t-slate-500"></div>
                </div>
              </div>
            </div>

          </div>
        </div>

        {/* RIGHT: Preprocessing */}
        <div className="space-y-5 md:pl-4 pt-5 md:pt-0">
          <div className="flex items-center gap-2 text-brand-cyan mb-1">
            <BarChart className="w-4 h-4" />
            <h3 className="text-sm font-bold uppercase tracking-wider">Preprocessing</h3>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Missing Values</label>
              <select 
                value={missingValue}
                onChange={(e) => setMissingValue(e.target.value)}
                className="w-full bg-dark-900/50 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:ring-1 focus:ring-brand-cyan outline-none"
              >
                <option value="drop" className="bg-dark-800 text-slate-200">Drop Rows</option>
                <option value="mean" className="bg-dark-800 text-slate-200">Mean</option>
                <option value="median" className="bg-dark-800 text-slate-200">Median</option>
                <option value="mode" className="bg-dark-800 text-slate-200">Mode</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Scaling</label>
              <select 
                value={scaling}
                onChange={(e) => setScaling(e.target.value)}
                disabled={algorithm === 'random_forest' || algorithm === 'xgboost'}
                className={`w-full bg-dark-900/50 border rounded-lg px-3 py-2 text-sm text-slate-200 outline-none transition-all ${
                  (algorithm === 'random_forest' || algorithm === 'xgboost')
                  ? 'opacity-50 cursor-not-allowed border-slate-700' 
                  : 'border-slate-700 focus:ring-1 focus:ring-brand-cyan'
                }`}
              >
                {(algorithm === 'random_forest' || algorithm === 'xgboost') ? (
                  <option value="none" className="bg-dark-800 text-slate-200">None (Tree/Boost)</option>
                ) : (
                  <>
                    <option value="standard" className="bg-dark-800 text-slate-200">Standard (Z)</option>
                    <option value="minmax" className="bg-dark-800 text-slate-200">MinMax (0-1)</option>
                    <option value="none" className="bg-dark-800 text-slate-200">None</option>
                  </>
                )}
              </select>
            </div>
          </div>

          <div className="pt-2">
            <div className="flex justify-between text-xs mb-2">
              <span className="text-slate-400 flex items-center gap-1.5"><Split className="w-3 h-3" /> Train/Test Split</span>
              <span className="text-brand-cyan font-mono">{(splitRatio * 100).toFixed(0)}% Train</span>
            </div>
            <input 
              type="range" 
              min="0.5" 
              max="0.9" 
              step="0.05"
              value={splitRatio}
              onChange={(e) => setSplitRatio(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-dark-900 rounded-full appearance-none cursor-pointer accent-brand-cyan"
            />
          </div>

          <div className="pt-4 border-t border-white/5 mt-4 space-y-4">
            <h4 className="text-[10px] font-bold text-brand-indigo uppercase tracking-widest flex items-center gap-2">
              <Wand2 className="w-3 h-3" /> Data Healing
            </h4>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-[10px] font-medium text-slate-500 mb-1.5 uppercase">Outlier Logic</label>
                <select 
                  value={outlierStrategy}
                  onChange={(e) => setOutlierStrategy(e.target.value)}
                  className="w-full bg-dark-900/50 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 outline-none focus:ring-1 focus:ring-brand-indigo"
                >
                  <option value="none">Keep All Data</option>
                  <option value="remove">Remove (IQR 1.5x)</option>
                </select>
              </div>

              {taskType === 'classification' && (
                <div>
                  <label className="block text-[10px] font-medium text-slate-500 mb-1.5 uppercase">Handle Imbalance</label>
                  <select 
                    value={imbalanceStrategy}
                    onChange={(e) => setImbalanceStrategy(e.target.value)}
                    className="w-full bg-dark-900/50 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 outline-none focus:ring-1 focus:ring-brand-indigo"
                  >
                    <option value="none">None</option>
                    <option value="smote">SMOTE (Oversample)</option>
                  </select>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* UPDATED PRO MODE TOGGLE */}
      <div className="flex items-center justify-between mb-6 p-4 bg-white/5 rounded-xl border border-white/10">
        <div className="flex items-center gap-3">
          <div 
            onClick={toggleProMode} // Use the new function
            className={`w-12 h-6 rounded-full relative transition-all cursor-pointer ${isProMode ? 'bg-brand-purple' : 'bg-slate-700'}`}
          >
            <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${isProMode ? 'left-7' : 'left-1'}`} />
          </div>
          <div>
            <h4 className="text-xs font-bold text-white uppercase tracking-widest">Enable Pro Mode</h4>
            <p className="text-[10px] text-slate-500">Manual Hyperparameter Tuning (Values reset on toggle)</p>
          </div>
        </div>
      </div>

      {/* UPDATED AUTO-TUNE TOGGLE */}
      <div className="flex items-center justify-between mb-6 p-4 bg-brand-cyan/5 rounded-xl border border-brand-cyan/20">
        <div className="flex items-center gap-3">
          <div 
            onClick={toggleAutoTune} // Use the new function
            className={`w-12 h-6 rounded-full relative transition-all cursor-pointer ${isAutoTune ? 'bg-brand-cyan' : 'bg-slate-700'}`}
          >
            <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${isAutoTune ? 'left-7' : 'left-1'}`} />
          </div>
          <div>
            <h4 className="text-xs font-bold text-white uppercase tracking-widest">Enable Auto-Tune Engine</h4>
            <p className="text-[10px] text-slate-500">AI search (Disables Manual Mode)</p>
          </div>
        </div>
        {/* ... intensity slider code ... */}
        {isAutoTune && (
          <div className="flex items-center gap-4 animate-in fade-in zoom-in duration-300">
            <span className="text-[10px] font-bold text-brand-cyan uppercase">Intensity: {searchIntensity}</span>
            <input 
              type="range" min="5" max="30" step="5" 
              value={searchIntensity} 
              onChange={(e) => setSearchIntensity(Number(e.target.value))}
              className="w-24 h-1 bg-dark-900 rounded-lg accent-brand-cyan" 
            />
          </div>
        )}
      </div>

      {isProMode && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-5 bg-brand-purple/5 border border-brand-purple/20 rounded-xl mb-6 animate-in fade-in slide-in-from-top-2">
          {ALGO_PARAMS[algorithm]?.map((param) => (
            <div key={param.name} className="space-y-2">
              <div className="flex justify-between text-[11px] font-bold uppercase text-slate-400">
                <span>{param.label}</span>
                <span className="text-brand-purple">
                  {hp[param.name] !== undefined ? hp[param.name] : param.default}
                </span>
              </div>

              {param.type === 'slider' ? (
                <input 
                  type="range" 
                  min={param.min} 
                  max={param.max} 
                  step={param.step} 
                  value={hp[param.name] !== undefined ? hp[param.name] : param.default} 
                  onChange={(e) => updateHp(param.name, e.target.value)} 
                  className="w-full h-1 bg-dark-900 rounded-lg appearance-none accent-brand-purple" 
                />
              ) : (
                <select
                  value={hp[param.name] !== undefined ? hp[param.name] : param.default}
                  onChange={(e) => updateHp(param.name, e.target.value)}
                  className="w-full bg-dark-900/50 border border-slate-700 text-slate-200 text-xs rounded-lg p-2 focus:ring-1 focus:ring-brand-purple outline-none"
                >
                  {param.options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                </select>
              )}
            </div>
          )) || (
            <p className="text-[10px] text-slate-500 italic col-span-2">No additional parameters available for this model.</p>
          )}
        </div>
      )}

      <div className="mt-8 pt-6 border-t border-white/5 flex justify-end">
        <button 
          onClick={handleStart}
          className="
            flex items-center gap-2 px-8 py-2.5 rounded-lg font-semibold text-white text-sm
            bg-gradient-to-r from-brand-indigo to-brand-purple 
            hover:shadow-[0_0_20px_rgba(79,70,229,0.4)] 
            transition-all active:scale-95
          "
        >
          <Play className="w-4 h-4 fill-current" />
          Start Training
        </button>
      </div>

    </div>
  );
}