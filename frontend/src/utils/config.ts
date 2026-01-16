const isStaging = import.meta.env.VITE_DEPLOY_MODE !== 'production';

interface Config {
    isStaging: boolean;
    isDev: boolean;
    apiBaseUrl: string;
}

export const config: Config = {
    isStaging,
    isDev: isStaging || import.meta.env.DEV as boolean,
    apiBaseUrl: import.meta.env.VITE_BACKEND_URL as string || 'http://localhost:8000'
};
