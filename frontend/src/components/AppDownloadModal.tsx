import { useState, useCallback } from 'react';
import './styles/AppDownloadModal.css';

const STORAGE_KEY = 'shubble-app-modal-dismissed';

// TODO: Replace with real App Store /Google Play URLs
const IOS_URL = 'https://apps.apple.com/us/app/shubble-rpi-shuttles/id6753160820?ppid=ce068241-6c58-493c-bd53-ef447321ea26';
const ANDROID_URL = 'https://play.google.com/store/apps/details?id=edu.rpi.shuttletracker';

function detectMobileDevice(): 'ios' | 'android' | null {
  const ua = navigator.userAgent || navigator.vendor;
  if (/iPad|iPhone|iPod/.test(ua)) return 'ios';
  if (/android/i.test(ua)) return 'android';
  return null;
}

function getInitialState() {
  if (localStorage.getItem(STORAGE_KEY)) return { visible: false, platform: null as 'ios' | 'android' | null };
  const platform = detectMobileDevice();
  return { visible: platform !== null, platform };
}

export default function AppDownloadModal() {
  const [{ visible, platform }, setState] = useState(getInitialState);
  const [dismissing, setDismissing] = useState(false);

  const dismiss = useCallback(() => {
    setDismissing(true);
    setTimeout(() => {
      localStorage.setItem(STORAGE_KEY, '1');
      setState({ visible: false, platform: null });
      setDismissing(false);
    }, 250);
  }, []);

  if (!visible) return null;

  return (
    <div
      className={`app-download-overlay${dismissing ? ' dismissing' : ''}`}
      id="app-download-modal"
    >
      <div className="app-download-card">
        {/* App icon */}
        <div className="app-download-icon">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7m0 9.5a2.5 2.5 0 0 1 0-5a2.5 2.5 0 0 1 0 5" />
          </svg>
        </div>

        <h2>Get the Shubble App</h2>
        <p>Track RPI shuttles in real time with a larger map and a smoother experience.</p>

        <div className="app-download-buttons">
          {platform === 'ios' && (
            <a
              href={IOS_URL}
              className="app-download-btn ios"
              target="_blank"
              rel="noopener noreferrer"
              id="app-download-ios"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512">
                <path d="M318.7 268.7c-.2-36.7 16.4-64.4 50-84.8-18.8-26.9-47.2-41.7-84.7-44.6-35.5-2.8-74.3 20.7-88.5 20.7-15 0-49.4-19.7-76.4-19.7C63.3 141.2 4 184.8 4 273.5q0 39.3 14.4 81.2c12.8 36.7 59 126.7 107.2 125.2 25.2-.6 43-17.9 75.8-17.9 31.8 0 48.3 17.9 76.4 17.9 48.6-.7 90.4-82.5 102.6-119.3-65.2-30.7-61.7-90-61.7-91.9m-56.6-164.2c27.3-32.4 24.8-61.9 24-72.5-24.1 1.4-52 16.4-67.9 34.9-17.5 19.8-27.8 44.3-25.6 71.9 26.1 2 49.9-11.4 69.5-34.3" />
              </svg>
              Download on the App Store
            </a>
          )}

          {platform === 'android' && (
            <a
              href={ANDROID_URL}
              className="app-download-btn android"
              target="_blank"
              rel="noopener noreferrer"
              id="app-download-android"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                <path d="M325.3 234.3L104.6 13l280.8 161.2-60.1 60.1zM47 0C34 6.8 25.3 19.2 25.3 35.3v441.3c0 16.1 8.7 28.5 21.7 35.3l256.6-256L47 0zm425.2 225.6l-58.9-34.1-65.7 64.5 65.7 64.5 60.1-34.1c18-14.3 18-46.5-1.2-60.8zM104.6 499l280.8-161.2-60.1-60.1L104.6 499z" />
              </svg>
              Get it on Google Play
            </a>
          )}
        </div>

        <button
          className="app-download-dismiss"
          onClick={dismiss}
          id="app-download-dismiss"
        >
          Continue to website
        </button>
      </div>
    </div>
  );
}
