const SESSION_KEY = 'daie_session';

/**
 * Save session data to localStorage.
 * @param {Object} data - Partial or full session data to merge.
 */
export const saveSessionData = (data) => {
  try {
    const existing = getSessionData();
    const merged = { ...existing, ...data, updatedAt: new Date().toISOString() };
    localStorage.setItem(SESSION_KEY, JSON.stringify(merged));
  } catch (err) {
    console.warn('Could not save session data:', err);
  }
};

/**
 * Get session data from localStorage.
 * @returns {Object}
 */
export const getSessionData = () => {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
};

/**
 * Clear session data from localStorage.
 */
export const clearSessionData = () => {
  try {
    localStorage.removeItem(SESSION_KEY);
  } catch (err) {
    console.warn('Could not clear session data:', err);
  }
};

export default { saveSessionData, getSessionData, clearSessionData };
