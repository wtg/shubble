import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import { loadConfig } from './utils/config'

// Dev tools detector
const setupDevToolsDetector = () => {
  let devtoolsOpen = false;
  const threshold = 160;

  const checkDevTools = () => {
    const widthThreshold = window.outerWidth - window.innerWidth > threshold;
    const heightThreshold = window.outerHeight - window.innerHeight > threshold;
    const orientation = widthThreshold ? 'vertical' : 'horizontal';

    if (!(heightThreshold && widthThreshold) &&
        ((window.Firebug && window.Firebug.chrome && window.Firebug.chrome.isInitialized) || widthThreshold || heightThreshold)) {
      if (!devtoolsOpen) {
        devtoolsOpen = true;
      }
    } else {
      if (devtoolsOpen) {
        devtoolsOpen = false;
      }
    }
  };

  // Check periodically
  setInterval(checkDevTools, 500);

  // Also check on window resize
  window.addEventListener('resize', checkDevTools);

  // Initial check
  checkDevTools();
};

setupDevToolsDetector();

// Load config before rendering the app
loadConfig().then(() => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
})
