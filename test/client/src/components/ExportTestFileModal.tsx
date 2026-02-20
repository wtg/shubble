import { useState, useEffect } from 'react';
import type { ShuttlesState, TestData, TestShuttle } from '../types';
import Modal from './Modal.tsx';

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

  // order shuttles drag shuttle4 to be first
  const [orderedShuttles, setOrderedShuttles] = useState<string[]>(
    Object.keys(shuttles)
  );

  // tracks which shuttles are selected shuttles 2, 6, 9
  const [selectedShuttles, setSelectedShuttles] = useState<string[]>([]);

  // sync orderedShuttles when new shuttles are added or removed
  useEffect(() => {
    const ids = Object.keys(shuttles);

    setOrderedShuttles((prev) => {
      // keep existing order for shuttles that still exist
      const existing = prev.filter((id) => ids.includes(id));

      // add newly created shuttles to the end
      const newOnes = ids.filter((id) => !prev.includes(id));

      return [...existing, ...newOnes];
    });

    // sync selectedShuttles in future change delete selected shuttle
    setSelectedShuttles((prev) =>
    prev.filter((id) => ids.includes(id)));

  }, [shuttles]);

  if (!isOpen) return null;

  const handleClose = () => {
    onClose();
  };

  // toggle selection when clicked
  const toggleSelection = (id: string) => {
    setSelectedShuttles((prev) =>
      prev.includes(id)
        ? prev.filter((s) => s !== id)
        : [...prev, id]
    );
  };

  // drag logic
  const handleDragStart = (e: React.DragEvent<HTMLDivElement>, index: number) => {
    // remove the default faded image
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
    // If nothing selected then export all in the dragged order
    const shuttlesToExport =
      selectedShuttles.length === 0
        ? orderedShuttles
        : orderedShuttles.filter((id) => selectedShuttles.includes(id));

    const shuttleArray: TestShuttle[] = shuttlesToExport.map(
      (id) => {
        const shuttle = shuttles[id];
        
        return {
          //Create array, and for each array there is event object that takeks shuttle action, optional parameter for route and duration
          events: shuttle.queue.map((action) => {
            const event: {
              type: typeof action.action;
              route?: string;
              duration?: number;
            } = {
              type: action.action
            };

            // Looping and on_break special because of their .route and .duration parameters
            // entering/exiting has no additional action type
            if (action.action === 'looping' && action.route) {
              event.route = action.route;
            }

            if (action.action === 'on_break' && action.duration !== undefined) {
              event.duration = action.duration;
            }

            return event;
          })
        };
      }
    );

    const exportData: TestData = {
      shuttles: shuttleArray
    };

    // Convert into JSON
    let json = JSON.stringify(exportData, null, 2);

    // magic regex clean up
    json = json.replace(
      /{\n\s+"type":\s+"([^"]+)"(,\n\s+"(route|duration)":\s+("[^"]+"|\d+))?\n\s+}/g,
      (match) => {
        return match
          .replace(/\n\s+/g, ' ')
          .replace(/\s+/g, ' ')
          .replace(/\s+}/, ' }');
      }
    );

    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = 'shuttle_export.json';
    link.click();

    URL.revokeObjectURL(url);
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
          <h3>Drag to Reorder. Click to Select.</h3>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            {orderedShuttles.map((id, index) => {
              const isSelected = selectedShuttles.includes(id);

              return (
                <div
                  key={id}
                  draggable
                  className={`export-shuttle-box ${isSelected ? 'selected' : ''}`}
                  onDragStart={(e) => handleDragStart(e, index)}
                  onDrop={(e) => handleDrop(e, index)}
                  onDragOver={handleDragOver}
                  onClick={() => toggleSelection(id)}
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