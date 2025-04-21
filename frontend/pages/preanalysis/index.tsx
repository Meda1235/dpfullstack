// pages/preanalysis/index.tsx
import { useEffect } from 'react';
import { useRouter } from 'next/router';

export default function PreAnalysisRedirect() {
    const router = useRouter();

    useEffect(() => {
        router.replace('/preanalysis/columns');
    }, [router]);

    return null;
}
