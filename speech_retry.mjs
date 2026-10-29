const RETRY_DELAYS_MS = [1000, 2000, 4000];

export function speechRetryDelay(error, attempt) {
  if (error !== "network" || !Number.isInteger(attempt) || attempt < 1 || attempt > RETRY_DELAYS_MS.length) {
    return null;
  }
  return RETRY_DELAYS_MS[attempt - 1];
}

export const MAX_SPEECH_NETWORK_RETRIES = RETRY_DELAYS_MS.length;
