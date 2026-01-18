interface Config {
    apiBaseUrl: string;
}

let config: Config | null = null;

export async function loadConfig(): Promise<Config> {
    if (config) {
        return config;
    }

    try {
        const response = await fetch('/config.json');
        const json = await response.json() as { apiBaseUrl?: string };

        config = {
            apiBaseUrl: json.apiBaseUrl || 'http://localhost:4000'
        };
    } catch {
        // Fallback for local development without config.json
        config = {
            apiBaseUrl: 'http://localhost:4000'
        };
    }

    return config;
}

export function getConfig(): Config {
    if (!config) {
        throw new Error('Config not loaded. Call loadConfig() first.');
    }
    return config;
}

export default {
    get apiBaseUrl() { return getConfig().apiBaseUrl; }
};
