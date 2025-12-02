// eslint-disable-next-line @typescript-eslint/no-unused-vars
import * as mapkit from 'apple-mapkit-js-browser';

export {};

declare global {
    namespace mapkit {
        interface CircleOverlay {
            routeKey?: string;
            stopKey?: string;
            stopName?: string;
            style?: mapkit.Style;
        }
        interface Map {
            _hoverCleanup?: () => void;
        }
        const loadedLibraries: string[];
    }
    interface Window {
        initMapKit?: (value?: unknown) => void;
    }
    interface ImportMeta {
        readonly env: {
            [key: string]: string | boolean | undefined;
            VITE_MAPKIT_KEY?: string;
            VITE_DEPLOY_MODE?: string;
            GIT_REV?: string;
        };
    }
}

