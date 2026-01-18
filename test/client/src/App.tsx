import { useEffect, useState, useCallback, useRef, type ChangeEvent } from "react";
import type { ShuttlesState, ShuttleAction as ShuttleActionType } from './types.ts';
import {
    fetchShuttlesFromApi,
    fetchRoutes,
    addShuttleToApi,
    addToQueueApi
} from './utils/shuttles.ts';
import { fetchEventCounts, deleteEvents } from './api/events.ts';
import { loadTestFile } from './utils/testFiles.ts';
import Shuttle from './components/Shuttle.tsx';
import "./App.css";

function App() {
    const [shuttles, setShuttles] = useState<ShuttlesState>({});
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [locationCount, setLocationCount] = useState(0);
    const [geofenceCount, setGeofenceCount] = useState(0);
    const [routes, setRoutes] = useState<string[]>([]);
    const [menuOpen, setMenuOpen] = useState(false);

    // Refs for stable references
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Fetch shuttles from API
    const updateShuttles = useCallback(async () => {
        const shuttlesData = await fetchShuttlesFromApi();
        setShuttles(shuttlesData);
    }, []);

    const updateEvents = async () => {
        const counts = await fetchEventCounts();
        setLocationCount(counts.locationCount);
        setGeofenceCount(counts.geofenceCount);
    };

    const handleClearEvents = async (keepShuttles: boolean) => {
        await deleteEvents(keepShuttles);
        if (!keepShuttles) {
            setSelectedId(null);
            setShuttles({});
        }
        setMenuOpen(false);
    };

    const handleAddShuttle = async () => {
        await addShuttleToApi();
        await updateShuttles();
    };

    const handleQueueAction = async (shuttleId: string, action: ShuttleActionType, route?: string, duration?: number) => {
        await addToQueueApi(shuttleId, [{ action, route, duration }]);
        await updateShuttles();
    };

    const handleUploadTest = async (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        try {
            await loadTestFile(file);
            await updateShuttles();
        } catch (err) {
            console.error('Failed to load test file:', err);
        }

        event.target.value = "";
        setMenuOpen(false);
    };

    const handleLoadTestClick = () => {
        fileInputRef.current?.click();
    };

    // Load routes on mount
    useEffect(() => {
        fetchRoutes().then(setRoutes);
    }, []);

    // Auto-select first shuttle when shuttles change
    useEffect(() => {
        const ids = Object.keys(shuttles);
        if (ids.length > 0 && (selectedId === null || !shuttles[selectedId])) {
            setSelectedId(ids[0]);
        }
    }, [shuttles, selectedId]);

    // Poll shuttles
    useEffect(() => {
        updateShuttles();
        const interval = setInterval(updateShuttles, 1000);
        return () => clearInterval(interval);
    }, [updateShuttles]);

    // Poll events
    useEffect(() => {
        updateEvents();
        const interval = setInterval(updateEvents, 1000);
        return () => clearInterval(interval);
    }, []);

    // Close menu when clicking outside
    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            const target = e.target as HTMLElement;
            if (!target.closest('.header-actions')) {
                setMenuOpen(false);
            }
        };

        if (menuOpen) {
            document.addEventListener('click', handleClickOutside);
            return () => document.removeEventListener('click', handleClickOutside);
        }
    }, [menuOpen]);

    const selectedShuttle = selectedId !== null ? shuttles[selectedId] : null;

    return (
        <div className="app">
            {/* Header */}
            <header className="header">
                <h1>Shuttle Test Suite</h1>
                <div className="header-actions">
                    <button
                        className="menu-button"
                        onClick={() => setMenuOpen(!menuOpen)}
                        aria-label="Menu"
                    >
                        <svg viewBox="0 0 24 24">
                            <circle cx="12" cy="5" r="2" />
                            <circle cx="12" cy="12" r="2" />
                            <circle cx="12" cy="19" r="2" />
                        </svg>
                    </button>

                    {menuOpen && (
                        <div className="dropdown-menu">
                            <button
                                className="dropdown-item"
                                onClick={handleLoadTestClick}
                            >
                                Load Test File
                            </button>
                            <div className="dropdown-divider" />
                            <button
                                className="dropdown-item"
                                onClick={() => handleClearEvents(true)}
                            >
                                Clear Events (Keep Shuttles)
                            </button>
                            <button
                                className="dropdown-item danger"
                                onClick={() => handleClearEvents(false)}
                            >
                                Clear All Data
                            </button>
                            <div className="dropdown-divider" />
                            <div className="dropdown-item" style={{ cursor: 'default', opacity: 0.7 }}>
                                {locationCount} locations, {geofenceCount} events
                            </div>
                        </div>
                    )}

                    {/* Hidden file input */}
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".json"
                        onChange={handleUploadTest}
                        style={{ display: 'none' }}
                    />
                </div>
            </header>

            {/* Tabs */}
            <div className="tabs">
                {Object.entries(shuttles).map(([id, shuttle]) => {
                    const pendingCount = shuttle.queue.filter(a => a.status !== 'completed').length;
                    return (
                        <button
                            key={id}
                            className={`tab ${id === selectedId ? 'active' : ''}`}
                            onClick={() => setSelectedId(id)}
                        >
                            Shuttle {id}
                            {pendingCount > 0 && (
                                <span style={{
                                    marginLeft: '0.5rem',
                                    padding: '0.125rem 0.375rem',
                                    borderRadius: '9999px',
                                    backgroundColor: 'var(--accent)',
                                    fontSize: '0.75rem'
                                }}>
                                    {pendingCount}
                                </span>
                            )}
                        </button>
                    );
                })}
                <button className="tab add-tab" onClick={handleAddShuttle}>
                    <span>+</span> Add Shuttle
                </button>
            </div>

            {/* Main Content */}
            <main className="main-content">
                {selectedShuttle ? (
                    <Shuttle
                        shuttle={selectedShuttle}
                        routes={routes}
                        onQueueAction={(action, route, duration) => handleQueueAction(selectedId!, action, route, duration)}
                    />
                ) : (
                    <div className="empty-state">
                        <p>No shuttles available</p>
                        <p>Click "+ Add Shuttle" to create one</p>
                    </div>
                )}
            </main>
        </div>
    );
}

export default App;
