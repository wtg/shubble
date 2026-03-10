import { useState, useEffect } from 'react';
import type { ShuttlesState, TestData, TestShuttle } from '../types';
import Modal from './Modal.tsx';
import { buildTestFile, parseTestFile, validateTestData, stringy } from '../utils/testFiles.ts';

interface ExportTestFileModalProps {
  isOpen: boolean;
  onClose: () => void;
  shuttles: ShuttlesState;
}

export default function ExportTestFileModal({
  isOpen,
  onClose,
  shuttles
}: ExportTestFileModalProps) {

  // Order Shuttles With Drag
  const [orderedShuttles, setOrderedShuttles] = useState<string[]>(Object.keys(shuttles));

  // Tracks which shuttles are selected shuttles 2, 6, 9
  const [selectedShuttles, setSelectedShuttles] = useState<string[]>([]);

  // alerts are displyed in modal
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Sync orderedShuttles when new shuttles are added or removed
  useEffect(() => {
    const ids = Object.keys(shuttles);

    setOrderedShuttles((prev) => {
      // Keep existing order for shuttles that still exist
      const existing = prev.filter((id) => ids.includes(id));

      // Add newly created shuttles to the end
      const newOnes = ids.filter((id) => !prev.includes(id));

      return [...existing, ...newOnes];
    });

    // Sync selectedShuttles in future change delete selected shuttle
    setSelectedShuttles((prev) =>
      prev.filter((id) => ids.includes(id)));

    // Clear export error if at least one shuttle has events
    const hasValidShuttle = Object.values(shuttles).some(
      (s) => s.queue?.length > 0
    );

    if (hasValidShuttle) {
      setErrorMessage(null);
    }

  }, [shuttles]);

  if (!isOpen) return null;

  const handleClose = () => {
    onClose();
  };

  // Toggle selection when clicked
  const toggleSelection = (id: string) => {
    setSelectedShuttles((prev) =>
      prev.includes(id)
        ? prev.filter((s) => s !== id)
        : [...prev, id]
    );
  };

  // Drag Shuttle Logic
  const handleDragStart = (e: React.DragEvent<HTMLDivElement>, index: number) => {
    e.dataTransfer.setDragImage(new Image(), 0, 0);
    e.dataTransfer.setData('text/plain', index.toString());
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>, dropIndex: number) => {
    const dragIndex = Number(e.dataTransfer.getData('text/plain'));
    if (dragIndex === dropIndex) return;

    const updated = [...orderedShuttles];
    const [moved] = updated.splice(dragIndex, 1);
    updated.splice(dropIndex, 0, moved);

    setOrderedShuttles(updated);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };


  const handleExport = () => { 

    setErrorMessage(null);

    // If nothing selected then export all in the dragged order
    const shuttlesWithEvents = orderedShuttles.filter(
      (id) => shuttles[id]?.queue?.length > 0
    );

    const shuttlesToExport =
      selectedShuttles.length === 0
        ? shuttlesWithEvents
        : shuttlesWithEvents.filter((id) => selectedShuttles.includes(id));
    
    if (shuttlesToExport.length === 0) {
      setErrorMessage('No events defined. At least one shuttle must contain actions to export.');
      return;
    }

    // For each shuttle, build invidual test file and validate each one
    // before adding shuttles to a list to be exported.
    // If shuttle fails validation, stop export and alert user with error message.
    const builtShuttles: TestShuttle[] = [];
    for (const id of shuttlesToExport) {
      const shuttle = shuttles[id];

      const text = buildTestFile(shuttle.queue);

      // Parse shuttle data
      const parseResult = parseTestFile(text);
      if (!parseResult.success || !parseResult.data) {
        setErrorMessage(parseResult.error || 'Parse failed');
        return;
      }

      //  Validate
      const validation = validateTestData(parseResult.data);
      if (!validation.valid || !validation.data) {
        setErrorMessage(validation.errors[0]?.message || 'Validation failed');
        return;
      }

      // No alerts, add to export list
      builtShuttles.push(validation.data.shuttles[0]);
    }

    const finalData: TestData = {shuttles: builtShuttles};
    
    stringy(finalData);
    onClose();
  };

  // deselects all shuttles and export
  function exportAll(){
    setSelectedShuttles([]);
    handleExport();
  }

  const footer = (
    <div className="modal-actions">
      <button className="btn-secondary" onClick={handleClose}>
        Cancel
      </button>
      <button
        className="btn-secondary"
        onClick={() => exportAll()}
      >
        Export All
      </button>
      <button className="btn-primary" onClick={handleExport}>
        Export
      </button>
    </div>
  );

  return (
    <Modal
      isOpen={isOpen}
      title="Export Test File"
      onClose={handleClose}
      footer={footer}
    >

        <div>

          {/* alerts are displyed in modal */}
          {errorMessage && (
            <div className="export-error">
              {errorMessage}
            </div>
          )}

          <h3>Drag to Reorder. Click to Select.</h3>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            {orderedShuttles.map((id, index) => {
              const isSelected = selectedShuttles.includes(id);
              const hasEvents = shuttles[id]?.queue?.length > 0;

              return (
                <div
                  key={id}
                  draggable
                  className={`export-shuttle-box ${isSelected ? 'selected' : ''} ${!hasEvents ? 'disabled' : ''}`}
                  onDragStart={(e) => handleDragStart(e, index)}
                  onDrop={(e) => handleDrop(e, index)}
                  onDragOver={handleDragOver}
                  onClick={() => {
                    if (!hasEvents) return;
                    toggleSelection(id);
                  }}
                >
                  {id}
                </div>
              );
            })}
          </div>

          <p className="export-shuttle-description">
            Selected shuttles turn blue. If none are selected, all will be exported.
          </p>
        </div>

    </Modal>
  );
}