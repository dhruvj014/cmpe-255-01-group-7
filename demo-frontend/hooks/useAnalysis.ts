'use client';

import { useState } from 'react';

export interface ReviewData {
  text: string;
  rating: number;
  tenure_days: number;
  review_count: number;
  seller_concentration: number;
  burst_score: number;
}

export interface LayerScores {
  L1_rules: number;
  L2_fpgrowth: number;
  L3_deberta: number;
  L4_clustering: number;
  L5_ensemble: number;
}

export interface AnalysisResult {
  is_spam: boolean;
  spam_probability: number;
  behavioral_risk_tier: string;
  top_factors: string[];
  layer_scores: LayerScores;
}

export function useAnalysis() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const analyzeReview = async (data: ReviewData) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        if (response.status === 422) throw new Error('Validation Error: Missing or invalid fields. Check all form inputs.');
        if (response.status === 500) throw new Error('Backend Error: The ML pipeline failed to process the review.');
        throw new Error(`Network Error (${response.status}): Failed to connect to backend.`);
      }

      const json: AnalysisResult = await response.json();
      setResult(json);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'An unexpected error occurred.';
      console.error('Integration Error:', err);
      setError(message.includes('fetch') ? 'Connection refused — is the FastAPI server running on port 8000?' : message);
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setError(null);
  };

  return { analyzeReview, isLoading, error, result, reset };
}
