import dynamic from 'next/dynamic';

const MainAnalysis = dynamic(() => import('@/components/MainAnalysis'), {
    ssr: false
});

export default function AnalysisPage() {
    return <MainAnalysis />;
}
