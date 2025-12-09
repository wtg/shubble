const isStaging = import.meta.env.VITE_DEPLOY_MODE !== 'production';

const config = {
    isStaging,
    isDev: isStaging || import.meta.env.DEV
};

export default config;