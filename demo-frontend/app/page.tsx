'use client';

import { useState } from 'react';
import { useAnalysis, ReviewData } from '@/hooks/useAnalysis';
import { useToast } from '@/hooks/useToast';
import InputForm from '@/components/InputForm';
import ResultsPanel from '@/components/ResultsPanel';
import PipelineBar from '@/components/PipelineBar';
import ToastContainer from '@/components/ToastContainer';
import ThemeToggle from '@/components/ThemeToggle';
import BackgroundBlobs from '@/components/BackgroundBlobs';

export default function Home() {
  const { analyzeReview, isLoading, error, result, reset } = useAnalysis();
  const { toasts, show: showToast } = useToast();
  const [formData, setFormData] = useState<ReviewData>({
    text: '',
    rating: 5,
    tenure_days: 0,
    review_count: 1,
    seller_concentration: 1.0,
    burst_score: 1,
  });

  const handleSubmit = (data: ReviewData) => {
    setFormData(data);
    analyzeReview(data);
  };

  return (
    <>
      <BackgroundBlobs />

      {/* Full-viewport flex column — nothing scrolls */}
      <div className="relative z-10 h-screen flex flex-col overflow-hidden">

        {/* ── Header ── */}
        <header className="px-5 py-3 flex items-center justify-between shrink-0" style={{ borderBottom: '1px solid var(--glass-border)' }}>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl flex items-center justify-center font-bold text-xs"
              style={{
                background: 'linear-gradient(135deg, var(--g7-from), var(--g7-to))',
                boxShadow: '0 4px 12px var(--g7-shadow)',
                color: 'var(--g7-text)',
              }}>
              G7
            </div>
            <div>
              <span className="font-semibold text-sm tracking-wide" style={{ color: 'var(--text)' }}>CMPE 255 · Group 7</span>
              <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Fake Review Detection Pipeline</p>
            </div>
          </div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium"
            style={{ background: 'var(--pill-bg)', border: '1px solid var(--pill-border)', color: 'var(--pill-text)' }}>
            <span>⚡</span>
            <span className="hidden sm:inline">L1 ETL → L2 FP-Growth → L3 DeBERTa → L4 Clustering → L5 Ensemble</span>
            <span className="sm:hidden">5-Layer Pipeline</span>
          </div>
        </header>

        {/* ── Compact hero ── */}
        <section className="px-5 pt-3 pb-2 text-center shrink-0">
          <h1 className="text-3xl font-black tracking-tight">
            <span className="gradient-text">Yelp Fraud</span>
            <span style={{ color: 'var(--text)' }}> Detector</span>
          </h1>
        </section>

        {/* ── Pipeline bar ── */}
        <div className="shrink-0">
          <PipelineBar isLoading={isLoading} result={result} />
        </div>

        {/* ── Main two-column content (fills remaining height) ── */}
        <main className="flex-1 px-4 pb-3 min-h-0 max-w-7xl mx-auto w-full">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
            <InputForm
              initialData={formData}
              isLoading={isLoading}
              onSubmit={handleSubmit}
              showToast={showToast}
            />
            <ResultsPanel
              isLoading={isLoading}
              error={error}
              result={result}
              onReset={reset}
            />
          </div>
        </main>

        {/* ── Footer ── */}
        <footer className="px-5 py-2 text-center text-xs shrink-0" style={{ borderTop: '1px solid var(--glass-border)', color: 'var(--text-faint)' }}>
          CMPE 255 · Group 7 · San José State University · Fake Review Detection
        </footer>
      </div>

      {/* Theme toggle — fixed just below header, top-right */}
      <div className="fixed top-[75px] right-4 z-50">
        <ThemeToggle />
      </div>

      <ToastContainer toasts={toasts} />
    </>
  );
}
