import type { Metadata } from 'next';
import './globals.css';
import Providers from '@/components/Providers';

export const metadata: Metadata = {
  title: 'Yelp Fraud Detector — CMPE 255 Group 7',
  description: 'Multi-layer fake review detection pipeline: ETL → FP-Growth → DeBERTa → Clustering → Ensemble',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
