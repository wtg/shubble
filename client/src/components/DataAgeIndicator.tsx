import '../styles/DataAgeIndicator.css';

type DataAgeIndicatorProps = {
    dataAge: number | null;
};

export default function DataAgeIndicator({ dataAge }: DataAgeIndicatorProps) {
    // Don't render if no data yet
    if (dataAge === null) {
        return null;
    }

    // Determine status based on data age
    const getStatus = () => {
        if (dataAge < 10) {
            return { label: 'Live', className: 'live' };
        } else if (dataAge < 30) {
            return { label: `Delayed (${Math.round(dataAge)}s)`, className: 'delayed' };
        } else {
            return { label: `Stale (${Math.round(dataAge)}s)`, className: 'stale' };
        }
    };

    const status = getStatus();

    return (
        <div className={`data-age-indicator ${status.className}`}>
            <span className="indicator-dot"></span>
            <span className="indicator-label">{status.label}</span>
        </div>
    );
}
