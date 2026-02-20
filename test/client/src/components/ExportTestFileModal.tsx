import { useState } from 'react';
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

  const [fileName, setFileName] = useState('shuttle_export');

  if (!isOpen) return null;

  const handleClose = () => {
    onClose();
  }



  const handleExport = () => {
    // Depends on Shuttle.tsx onQueueAction: action, route, duration
    const shuttleArray: TestShuttle[] = Object.values(shuttles).map(
      (shuttle) => ({
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
      })
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
          // clean up regex at the end of each shuttle/line
      }
    );


    // Create blob with JSON
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    
    const finalFileName = fileName.endsWith('.json')
      ? fileName
      : `${fileName}.json`;

    // trigger download and then remove the link
    const link = document.createElement('a');
    link.href = url;
    link.download = finalFileName;
    link.click();

    URL.revokeObjectURL(url);
    onClose();


    
  };

  const footer = (
    <div className="modal-actions">
      <button className="btn-secondary" onClick={handleClose}>
        Cancel
      </button>
      <button className="btn-primary" onClick={handleExport}>
        Export
      </button>

    </div>

  );

  return (
    // fix ui
    <Modal
      isOpen={isOpen}
      title="Import Test File"
      onClose={handleClose}    
      footer={footer}      
    >
      <div>
        Export a JSON test file based on Queued Actions across all shuttles.
        Goal: Select each individual shuttle before exporting them.
        Run testing in test.tsx
        clean up with regular docker clear data
      </div>
    </Modal>
    /*<div className="modal-overlay">
      <div className="modal">
        <h2>Export Test File</h2>

        <div className="modal-body">
          <label>File Name:</label>
          <input
            type="text"
            value={fileName}
            onChange={(e) => setFileName(e.target.value)}
            placeholder="Enter file name..."
          />
        </div>

        <div className="modal-actions">
          <button onClick={onClose}>Cancel</button>
          <button onClick={handleExport}>Export</button>
        </div>
      </div>
    </div>
    */
  );
}

