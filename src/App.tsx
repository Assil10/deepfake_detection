import { useRef } from 'react';
import { BenchmarksSection } from './components/BenchmarksSection';
import { InspectorWorkspace } from './components/InspectorWorkspace';
import { Navbar } from './components/Navbar';

export function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const triggerUpload = () => {
    const detectorSection = document.getElementById('detector');
    if (detectorSection) {
      detectorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    fileInputRef.current?.click();
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      <Navbar onUploadClick={triggerUpload} />

      <main className="flex-1">
        <InspectorWorkspace fileInputRef={fileInputRef} />
        <BenchmarksSection />
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto flex max-w-[1360px] flex-col gap-2 px-6 py-6 text-xs text-slate-500 sm:flex-row sm:items-center sm:justify-between">
          <span>Veritas · Deepfake Detection Research Interface</span>
          <span>
            Datasets: CIFAKE (120k) & 140k Real and Fake Faces · Benchmarks from data/processed/
          </span>
        </div>
      </footer>
    </div>
  );
}

export default App;
