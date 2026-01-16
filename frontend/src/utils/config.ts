const isStaging = import.meta.env.VITE_DEPLOY_MODE !== 'production';

export const config = {
    isStaging,
    isDev: isStaging || import.meta.env.DEV,
    apiBaseUrl: import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
};
