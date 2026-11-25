import { useState } from 'react';
import {
  ABLATION_B0_VS_V2,
  DATASET_SUMMARIES,
  EVALUATION_RESULTS,
  PIPELINE_SPECS,
} from '../data/repoMetrics';

type BenchmarkTab = 'combined' | 'ablation';

export function BenchmarksSection() {
  const [activeTab, setActiveTab] = useState<BenchmarkTab>('combined');

  return (
    <section id="benchmarks" className="border-t border-slate-200 bg-white">
      <div className="mx-auto max-w-[1360px] px-6 py-14 space-y-12">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div className="max-w-2xl space-y-2">
            <h2 className="font-display text-2xl font-bold tracking-tight text-slate-900 text-balance">
              Repository Model Benchmarks & Cross-Dataset Findings
            </h2>
            <p className="text-sm leading-relaxed text-slate-600">
              Empirical evaluation metrics recorded in{' '}
              <code className="font-mono text-xs text-slate-800">
                data/processed/evaluation_results.csv
              </code>{' '}
              and{' '}
              <code className="font-mono text-xs text-slate-800">
                results_b0_vs_v2.csv
              </code>
              . Single-domain models collapse out-of-domain; joint CIFAKE + Faces training restores
              cross-domain AUC.
            </p>
          </div>

          <div
            role="tablist"
            aria-label="Benchmark views"
            className="inline-flex items-center gap-1 rounded-lg bg-slate-100 p-1 self-start"
          >
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === 'combined'}
              onClick={() => setActiveTab('combined')}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeTab === 'combined'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Combined vs Single-Domain
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === 'ablation'}
              onClick={() => setActiveTab('ablation')}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeTab === 'ablation'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              B0 vs EfficientNetV2-S
            </button>
          </div>
        </div>

        {activeTab === 'combined' && (
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-12 items-start">
            {/* AUC Comparison Chart */}
            <div className="lg:col-span-5 rounded-xl border border-slate-200 bg-slate-50 p-6 space-y-5">
              <div className="flex items-baseline justify-between gap-4">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">
                    ROC-AUC Cross-Domain Comparison
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    CIFAKE Objects vs 140k Faces test splits
                  </p>
                </div>
                <div className="flex items-center gap-4 text-xs text-slate-600 shrink-0">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-xs bg-blue-600" aria-hidden="true" />
                    CIFAKE AUC
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-xs bg-slate-700" aria-hidden="true" />
                    Faces AUC
                  </span>
                </div>
              </div>

              <div className="space-y-4">
                {[
                  { name: 'model_combined (B0)', cifake: 0.992, faces: 0.972, highlight: true },
                  { name: 'model_cifake (B0)', cifake: 0.983, faces: 0.46, highlight: false },
                  { name: 'model_faces (B0)', cifake: 0.493, faces: 0.88, highlight: false },
                  { name: 'model_cifake_v2 (V2-S)', cifake: 0.984, faces: 0.445, highlight: false },
                  { name: 'model_faces_v2 (V2-S)', cifake: 0.452, faces: 0.915, highlight: false },
                ].map((item) => (
                  <div key={item.name} className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span
                        className={`font-mono ${
                          item.highlight ? 'font-semibold text-slate-900' : 'text-slate-600'
                        }`}
                      >
                        {item.name}
                      </span>
                      <span className="font-mono tabular-nums text-slate-600">
                        {item.cifake.toFixed(3)} / {item.faces.toFixed(3)}
                      </span>
                    </div>
                    <div className="space-y-1">
                      <div className="h-2 w-full rounded-xs bg-slate-200 overflow-hidden">
                        <div
                          className="h-full bg-blue-600 transition-all"
                          style={{ width: `${Math.round(item.cifake * 100)}%` }}
                        />
                      </div>
                      <div className="h-2 w-full rounded-xs bg-slate-200 overflow-hidden">
                        <div
                          className="h-full bg-slate-700 transition-all"
                          style={{ width: `${Math.round(item.faces * 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-slate-500 pt-2 border-t border-slate-200">
                0.500 baseline indicates random chance. Joint training on 20k samples per domain
                resolves cross-domain collapse.
              </p>
            </div>

            {/* Exact evaluation_results.csv Table */}
            <div className="lg:col-span-7 overflow-x-auto rounded-xl border border-slate-200 bg-white">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-xs font-semibold text-slate-600">
                    <th className="py-3 px-4">Checkpoint</th>
                    <th className="py-3 px-3">Train Data</th>
                    <th className="py-3 px-3">Test Split</th>
                    <th className="py-3 px-3 text-right">AUC</th>
                    <th className="py-3 px-3 text-right">Accuracy</th>
                    <th className="py-3 px-3 text-right">Recall (Fake)</th>
                    <th className="py-3 px-4 text-right">FPR</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-xs">
                  {EVALUATION_RESULTS.map((row) => (
                    <tr
                      key={`${row.model}-${row.testedOn}`}
                      className={row.isCombined ? 'bg-blue-50/40' : 'hover:bg-slate-50/70'}
                    >
                      <td className="py-3 px-4 font-mono font-medium text-slate-900 whitespace-nowrap">
                        {row.model}.pth
                      </td>
                      <td className="py-3 px-3 text-slate-600 whitespace-nowrap">
                        {row.trainDomain}
                      </td>
                      <td className="py-3 px-3 text-slate-700 whitespace-nowrap">{row.testedOn}</td>
                      <td className="py-3 px-3 text-right font-mono tabular-nums font-semibold text-slate-900">
                        {row.auc.toFixed(3)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-700">
                        {(row.accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-700">
                        {(row.recallFake * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-600">
                        {(row.fpr * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'ablation' && (
          <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-xs font-semibold text-slate-600">
                  <th className="py-3 px-4">Trained On</th>
                  <th className="py-3 px-3">Backbone</th>
                  <th className="py-3 px-3">Tested On</th>
                  <th className="py-3 px-3 text-right">AUC</th>
                  <th className="py-3 px-3 text-right">Accuracy</th>
                  <th className="py-3 px-3 text-right">Recall (Fake)</th>
                  <th className="py-3 px-3 text-right">FPR</th>
                  <th className="py-3 px-4 text-right">TP / FP / TN / FN</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-xs">
                {ABLATION_B0_VS_V2.map((row) => (
                  <tr
                    key={`${row.trainedOn}-${row.model}-${row.testedOn}`}
                    className="hover:bg-slate-50/70"
                  >
                    <td className="py-3 px-4 font-medium text-slate-900 whitespace-nowrap">
                      {row.trainedOn}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-700 whitespace-nowrap">
                      EfficientNet-{row.model}
                    </td>
                    <td className="py-3 px-3 text-slate-600 whitespace-nowrap">{row.testedOn}</td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums font-semibold text-slate-900">
                      {row.auc.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-700">
                      {(row.accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-700">
                      {(row.recallFake * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-600">
                      {(row.fpr * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-500 whitespace-nowrap">
                      {row.tp} / {row.fp} / {row.tn} / {row.fn}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Training Datasets */}
        <div id="datasets" className="border-t border-slate-200 pt-10 space-y-6">
          <div>
            <h3 className="font-display text-xl font-bold text-slate-900">
              Training Dataset Specifications
            </h3>
            <p className="text-sm text-slate-600 mt-1">
              Balanced class splits recorded in{' '}
              <code className="font-mono text-xs text-slate-800">
                data/processed/data_exploration_summary.csv
              </code>{' '}
              and{' '}
              <code className="font-mono text-xs text-slate-800">class_distribution.csv</code>.
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            {DATASET_SUMMARIES.map((ds) => (
              <div
                key={ds.dataset}
                className="rounded-xl border border-slate-200 bg-slate-50 p-6 space-y-4"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h4 className="font-display text-lg font-bold text-slate-900">{ds.dataset}</h4>
                    <p className="text-xs text-slate-600 mt-0.5">{ds.content}</p>
                  </div>
                  <span className="font-mono tabular-nums text-sm font-semibold text-slate-900 shrink-0">
                    {ds.totalImages.toLocaleString()} imgs
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-4 border-t border-slate-200 pt-4 text-xs">
                  <div>
                    <div className="text-slate-500">Native Size</div>
                    <div className="font-mono tabular-nums font-semibold text-slate-900 mt-1">
                      {ds.resolution}
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-500">Train Balance</div>
                    <div className="font-mono tabular-nums font-semibold text-slate-900 mt-1">
                      {(ds.trainFake / 1000).toFixed(0)}k / {(ds.trainReal / 1000).toFixed(0)}k
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-500">Mean JPEG Size</div>
                    <div className="font-mono tabular-nums font-semibold text-slate-900 mt-1">
                      {ds.meanSizeKbFake} / {ds.meanSizeKbReal} KB
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Pipeline & Inference Architecture */}
        <div id="pipeline" className="border-t border-slate-200 pt-10 space-y-6">
          <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
            <div>
              <h3 className="font-display text-xl font-bold text-slate-900">
                Preprocessing & Inference Architecture
              </h3>
              <p className="text-sm text-slate-600 mt-1">
                Preprocessing rules defined in{' '}
                <code className="font-mono text-xs text-slate-800">
                  deepfake_detection/utils.py
                </code>{' '}
                paired with live multimodal forensic vision on this web runtime.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {PIPELINE_SPECS.map((item) => (
              <div
                key={item.step}
                className="rounded-xl border border-slate-200 bg-slate-50 p-5 space-y-2"
              >
                <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
                  <span>{item.step}</span>
                  <span>{item.source}</span>
                </div>
                <h4 className="text-sm font-semibold text-slate-900">{item.title}</h4>
                <p className="text-xs leading-relaxed text-slate-600">{item.detail}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
