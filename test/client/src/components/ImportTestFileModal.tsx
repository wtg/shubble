import { useState, useRef, type ChangeEvent } from 'react';
import Modal from './Modal.tsx';
import {
    parseTestFile,
    validateTestData,
    importTestData,
    type ValidationResult,
    type ParseResult
} from '../utils/testFiles.ts';

interface ImportTestFileModalProps {
    isOpen: boolean;
    onClose: () => void;
    onImportComplete: () => void;
}

type ImportState = 'select' | 'preview' | 'importing' | 'error';

export default function ImportTestFileModal({
    isOpen,
    onClose,
    onImportComplete
}: ImportTestFileModalProps) {
    const [state, setState] = useState<ImportState>('select');
    const [fileName, setFileName] = useState<string>('');
    const [parseResult, setParseResult] = useState<ParseResult | null>(null);
    const [validation, setValidation] = useState<ValidationResult | null>(null);
    const [importError, setImportError] = useState<string>('');
    const fileInputRef = useRef<HTMLInputElement>(null);

    const reset = () => {
        setState('select');
        setFileName('');
        setParseResult(null);
        setValidation(null);
        setImportError('');
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const handleClose = () => {
        reset();
        onClose();
    };

    const handleFileSelect = async (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setFileName(file.name);
        const text = await file.text();
        const parsed = parseTestFile(text);
        setParseResult(parsed);

        if (parsed.success && parsed.data) {
            const validationResult = validateTestData(parsed.data);
            setValidation(validationResult);
        } else {
            setValidation(null);
        }

        setState('preview');
    };

    const handleImport = async () => {
        if (!validation?.valid || !validation.data) return;

        setState('importing');
        setImportError('');

        try {
            await importTestData(validation.data);
            onImportComplete();
            handleClose();
        } catch (err) {
            setImportError(err instanceof Error ? err.message : 'Import failed');
            setState('error');
        }
    };

    const totalShuttles = parseResult?.data?.shuttles?.length ?? 0;
    const totalEvents = parseResult?.data?.shuttles?.reduce(
        (sum, s) => sum + (s.events?.length ?? 0), 0
    ) ?? 0;

    const footer = (
        <div className="modal-actions">
            <button className="btn-secondary" onClick={handleClose}>
                Cancel
            </button>
            {state === 'preview' && validation?.valid && (
                <button className="btn-primary" onClick={handleImport}>
                    Import
                </button>
            )}
            {state === 'error' && (
                <button className="btn-primary" onClick={() => setState('preview')}>
                    Back
                </button>
            )}
        </div>
    );

    return (
        <Modal
            isOpen={isOpen}
            onClose={handleClose}
            title="Import Test File"
            footer={footer}
        >
            {state === 'select' && (
                <div className="import-select">
                    <p className="import-description">
                        Select a JSON test file to import. The file should contain shuttle definitions with action queues.
                    </p>
                    <label className="file-drop-zone">
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".json"
                            onChange={handleFileSelect}
                        />
                        <div className="file-drop-content">
                            <span className="file-drop-icon">+</span>
                            <span>Choose a file or drag it here</span>
                        </div>
                    </label>
                </div>
            )}

            {state === 'preview' && (
                <div className="import-preview">
                    <div className="preview-header">
                        <span className="preview-filename">{fileName}</span>
                        <button className="btn-text" onClick={reset}>Change file</button>
                    </div>

                    {/* Parse Error */}
                    {parseResult && !parseResult.success && (
                        <div className="validation-section validation-error">
                            <h4>Parse Error</h4>
                            <p>{parseResult.error}</p>
                        </div>
                    )}

                    {/* Validation Results */}
                    {validation && (
                        <div className={`validation-section ${validation.valid ? 'validation-success' : 'validation-error'}`}>
                            <h4>
                                {validation.valid ? 'Validation Passed' : 'Validation Failed'}
                            </h4>
                            {validation.valid ? (
                                <p>{totalShuttles} shuttle(s) with {totalEvents} total event(s)</p>
                            ) : (
                                <ul className="error-list">
                                    {validation.errors.map((err, i) => (
                                        <li key={i}>
                                            {err.shuttleIndex !== undefined && <strong>Shuttle {err.shuttleIndex + 1}</strong>}
                                            {err.eventIndex !== undefined && <span> Event {err.eventIndex + 1}</span>}
                                            {(err.shuttleIndex !== undefined || err.eventIndex !== undefined) && ': '}
                                            {err.message}
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </div>
                    )}

                    {/* Preview */}
                    {parseResult?.success && parseResult.data && (
                        <div className="preview-section">
                            <h4>Preview</h4>
                            <div className="preview-shuttles">
                                {parseResult.data.shuttles.map((shuttle, i) => (
                                    <div key={i} className="preview-shuttle">
                                        <div className="preview-shuttle-header">
                                            Shuttle {i + 1}
                                            <span className="preview-event-count">
                                                {shuttle.events?.length ?? 0} events
                                            </span>
                                        </div>
                                        <div className="preview-events">
                                            {shuttle.events?.map((evt, j) => (
                                                <span key={j} className={`preview-event event-${evt.type}`}>
                                                    {evt.type}
                                                    {evt.route && ` (${evt.route})`}
                                                    {evt.duration && ` ${evt.duration}s`}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {state === 'importing' && (
                <div className="import-loading">
                    <div className="spinner"></div>
                    <p>Importing shuttles and queuing actions...</p>
                </div>
            )}

            {state === 'error' && (
                <div className="import-error">
                    <h4>Import Failed</h4>
                    <p>{importError}</p>
                </div>
            )}
        </Modal>
    );
}
