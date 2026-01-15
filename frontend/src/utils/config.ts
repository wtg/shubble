const isStaging = import.meta.env.VITE_DEPLOY_MODE !== 'production';
console.log(import.meta.env);
const config = {
    isStaging,
    isDev: isStaging || import.meta.env.DEV,
    apiBaseUrl: import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
};

export default config;
