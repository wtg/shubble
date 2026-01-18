import type { QueuedAction } from '../types.ts';

interface ShuttleActionProps {
    action: QueuedAction;
    selected: boolean;
    onSelect: () => void;
}

export default function ShuttleAction({ action, selected, onSelect }: ShuttleActionProps) {
    const statusLabels: Record<QueuedAction['status'], string> = {
        pending: 'Pending',
        in_progress: 'Running',
        completed: 'Done',
        failed: 'Failed'
    };

    return (
        <div
            className={`queue-item ${action.status} ${selected ? 'selected' : ''}`}
            onClick={onSelect}
        >
            <div className="queue-item-content">
                <div className="queue-item-action">
                    {action.action.replace('_', ' ')}
                </div>
                {action.route && (
                    <div className="queue-item-route">
                        Route: {action.route}
                    </div>
                )}
                {action.duration && (
                    <div className="queue-item-route">
                        Duration: {action.duration}s
                    </div>
                )}
            </div>

            <span className="queue-item-status">
                {statusLabels[action.status]}
            </span>
        </div>
    );
}
