import { useState } from 'react';
import type { Shuttle as ShuttleType, ShuttleAction as ShuttleActionType, QueuedAction } from '../types.ts';
import { ACTIONS } from '../types.ts';
import ShuttleAction from './ShuttleAction.tsx';

interface ShuttleProps {
    shuttle: ShuttleType;
    routes: string[];
    onQueueAction: (action: ShuttleActionType, route?: string, duration?: number) => void;
}

export default function Shuttle({
    shuttle,
    routes,
    onQueueAction
}: ShuttleProps) {
    const [selectedActionId, setSelectedActionId] = useState<string | null>(null);
    const [addingAction, setAddingAction] = useState(false);

    // New action form state
    const [newActionType, setNewActionType] = useState<ShuttleActionType>(ACTIONS.ENTERING);
    const [newActionRoute, setNewActionRoute] = useState<string>('');
    const [newActionDuration, setNewActionDuration] = useState<number>(60);

    const handleStartAddAction = () => {
        setAddingAction(true);
        setSelectedActionId(null);
        setNewActionType(ACTIONS.ENTERING);
        setNewActionRoute(routes[0] || '');
        setNewActionDuration(60);
    };

    const handleCancelAddAction = () => {
        setAddingAction(false);
    };

    const handleQueueNewAction = () => {
        const needsRoute = newActionType === ACTIONS.LOOPING;
        const needsDuration = newActionType === ACTIONS.ON_BREAK;

        onQueueAction(
            newActionType,
            needsRoute ? newActionRoute : undefined,
            needsDuration ? newActionDuration : undefined
        );
        setAddingAction(false);
    };

    // Get display name for the action
    const getActionDisplay = (action: ShuttleActionType | null) => {
        if (!action) return 'Unknown';
        return action.replace('_', ' ');
    };

    // Get status display
    const getStatusDisplay = (status: QueuedAction['status']) => {
        const labels: Record<QueuedAction['status'], string> = {
            pending: 'Pending',
            in_progress: 'Running',
            completed: 'Completed',
            failed: 'Failed'
        };
        return labels[status];
    };

    // Find the selected action from the queue
    const selectedAction = shuttle.queue.find(a => a.id === selectedActionId);

    // Check if new action form is valid
    const isNewActionValid = () => {
        if (newActionType === ACTIONS.LOOPING && !newActionRoute) {
            return false;
        }
        return true;
    };

    return (
        <div className="shuttle-view">
            {/* Queue Sidebar */}
            <aside className="queue-sidebar">
                <div className="queue-header">
                    <h3>Action Queue</h3>
                </div>

                <div className="queue-list">
                    {shuttle.queue.length === 0 ? (
                        <div className="queue-empty">
                            No actions queued
                        </div>
                    ) : (
                        shuttle.queue.map(action => (
                            <ShuttleAction
                                key={action.id}
                                action={action}
                                selected={action.id === selectedActionId && !addingAction}
                                onSelect={() => {
                                    setSelectedActionId(action.id);
                                    setAddingAction(false);
                                }}
                            />
                        ))
                    )}
                </div>

                <div className="queue-footer">
                    <button
                        className="add-action-button"
                        onClick={handleStartAddAction}
                    >
                        + Add Action
                    </button>
                </div>
            </aside>

            {/* Right Panel */}
            <div className="current-action-panel">
                <div className="current-action-card">
                    {addingAction ? (
                        <>
                            <button
                                className="close-button"
                                onClick={handleCancelAddAction}
                                aria-label="Cancel"
                            >
                                x
                            </button>

                            <div className="shuttle-id">Add Action</div>

                            <div className="add-action-form">
                                <div className="form-group">
                                    <label>Action Type</label>
                                    <select
                                        value={newActionType}
                                        onChange={e => setNewActionType(e.target.value as ShuttleActionType)}
                                    >
                                        {Object.values(ACTIONS).map(action => (
                                            <option key={action} value={action}>
                                                {action.replace('_', ' ')}
                                            </option>
                                        ))}
                                    </select>
                                </div>

                                {newActionType === ACTIONS.LOOPING && (
                                    <div className="form-group">
                                        <label>Route</label>
                                        <select
                                            value={newActionRoute}
                                            onChange={e => setNewActionRoute(e.target.value)}
                                        >
                                            {routes.length === 0 ? (
                                                <option value="" disabled>No routes available</option>
                                            ) : (
                                                routes.map(route => (
                                                    <option key={route} value={route}>
                                                        {route}
                                                    </option>
                                                ))
                                            )}
                                        </select>
                                    </div>
                                )}

                                {newActionType === ACTIONS.ON_BREAK && (
                                    <div className="form-group">
                                        <label>Duration (seconds)</label>
                                        <input
                                            type="number"
                                            min="1"
                                            value={newActionDuration}
                                            onChange={e => setNewActionDuration(parseInt(e.target.value) || 1)}
                                        />
                                    </div>
                                )}

                                <button
                                    className="primary queue-button"
                                    onClick={handleQueueNewAction}
                                    disabled={!isNewActionValid()}
                                >
                                    Queue
                                </button>
                            </div>
                        </>
                    ) : (
                        <>
                            <div className="shuttle-id">Shuttle {shuttle.id}</div>

                            {selectedAction ? (
                                <>
                                    <div className={`current-state ${selectedAction.status}`}>
                                        {getActionDisplay(selectedAction.action)}
                                    </div>
                                    {selectedAction.route && (
                                        <div className="current-route">
                                            Route: {selectedAction.route}
                                        </div>
                                    )}
                                    {selectedAction.duration && (
                                        <div className="current-route">
                                            Duration: {selectedAction.duration}s
                                        </div>
                                    )}
                                    <div className="current-route">
                                        Status: {getStatusDisplay(selectedAction.status)}
                                    </div>
                                </>
                            ) : (
                                <div className="current-state waiting">
                                    No action selected
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
