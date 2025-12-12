import { Component, type ReactNode } from 'react';
import { Banner } from './AnnouncementBanner';

interface ErrorBoundaryProps {
    children: ReactNode;
}

interface ErrorBoundaryState {
    hasError: boolean;
    error: Error | null;
}

/**
 * Error boundary component that catches unhandled JavaScript errors
 * in child components and displays an error banner.
 */
export default class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
    constructor(props: ErrorBoundaryProps) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error: Error): ErrorBoundaryState {
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
        // Log error to console for debugging
        console.error('ErrorBoundary caught an error:', error, errorInfo);
    }

    render(): ReactNode {
        if (this.state.hasError) {
            return (
                <div style={{ width: '100%' }}>
                    <Banner
                        message="Something went wrong. Please try reloading the page."
                        type="error"
                        showReload={true}
                    />
                </div>
            );
        }

        return this.props.children;
    }
}
