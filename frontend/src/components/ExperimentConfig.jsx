import { useState, useEffect } from 'react';
import { Activity, BarChart, Split, Play, CheckCircle2 } from 'lucide-react';

const ALGORITHMS = {
  classification: [
    { id: 'random_forest', name: 'Random Forest Classifier' },
    { id: 'xgboost', name: 'Gradient Boosting (Hist)' },
    { id: 'logistic_regression', name: 'Logistic Regression' },
    { id: 'knn', name: 'K-Nearest Neighbors (KNN)' },
    { id: 'svm', name: 'Support Vector Machine (SVM)' }
  ],
  regression: [
    { id: 'random_forest', name: 'Random Forest Regressor' },
    { id: 'xgboost', name: 'Gradient Boosting (Hist)' },
    { id: 'linear_regression', name: 'Linear Regression' },
    { id: 'knn', name: 'K-Nearest Neighbors (KNN)' },
    { id: 'svr', name: 'Support Vector Regressor (SVR)' }
  ]
};

export default function ExperimentConfig({ dataset, onStartTraining }) {
  const [targetCol, setTargetCol] = useState('');
  const [taskType, setTaskType] = useState('classification');
  const [missingValue, setMissingValue] = useState('drop');
  const [scaling, setScaling] = useState('standard');
  const [splitRatio, setSplitRatio] = useState(0.8);
  const [algorithm, setAlgorithm] = useState('');

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
    const isTreeModel = algorithm === 'random_forest' || algorithm === 'xgboost';
    if (isTreeModel) {
      setScaling('none');
    } else if (scaling === 'none') {
      setScaling('standard');
    }
  }, [algorithm]);

  const handleStart = () => {
    if (!targetCol) return alert("Please select a target column first!");
    
    onStartTraining({
      filename: dataset.filename,
      target: targetCol,
      task_type: taskType,
      algorithm: algorithm, 
      preprocessing: {
        missing_value_strategy: missingValue,
        scaling: scaling,
        test_size: parseFloat((1 - splitRatio).toFixed(2))
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
        </div>
      </div>

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