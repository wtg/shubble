import { config } from "./config";

export const log = (...args: unknown[]) => {
   if (config.isDev) console.log('[WEB]', ...args);
}