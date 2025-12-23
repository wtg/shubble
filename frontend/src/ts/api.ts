/**
 * API configuration and utilities
 */

// Get backend URL from environment variable, default to http://localhost:5001
// In production: should be set to the backend URL
const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

/**
 * Get the full API URL for a given endpoint
 * @param endpoint - The API endpoint (e.g., '/api/locations')
 * @returns The full URL
 */
export function getApiUrl(endpoint: string): string {
    // If API_BASE_URL is empty, return endpoint as-is (relative URL, for same origin)
    // Otherwise, concatenate base URL with endpoint
    return API_BASE_URL ? `${API_BASE_URL}${endpoint}` : endpoint;
}

/**
 * Fetch wrapper that automatically uses the configured API base URL
 * @param endpoint - The API endpoint (e.g., '/api/locations')
 * @param options - Fetch options
 * @returns Promise with the fetch response
 */
export async function apiFetch(endpoint: string, options?: RequestInit): Promise<Response> {
    return fetch(getApiUrl(endpoint), options);
}
