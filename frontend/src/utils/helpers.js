/**
 * Utility helper functions
 */

/**
 * Format confidence score as percentage
 * @param {number} confidence - Confidence value (0-1)
 * @returns {string} Formatted percentage string
 */
export const formatConfidence = (confidence) => {
  return `${(confidence * 100).toFixed(2)}%`;
};

/**
 * Format epsilon value for display
 * @param {number} epsilon - Epsilon value
 * @returns {string} Formatted epsilon string
 */
export const formatEpsilon = (epsilon) => {
  return epsilon.toFixed(4);
};

/**
 * Get color class based on confidence level
 * @param {number} confidence - Confidence value (0-1)
 * @returns {string} Tailwind color class
 */
export const getConfidenceColor = (confidence) => {
  if (confidence >= 0.8) return 'text-green-600';
  if (confidence >= 0.5) return 'text-yellow-600';
  return 'text-red-600';
};

/**
 * Get color class based on attack success
 * @param {boolean} success - Whether attack was successful
 * @returns {string} Tailwind color class
 */
export const getSuccessColor = (success) => {
  return success ? 'text-red-600' : 'text-green-600';
};

/**
 * Truncate text to max length
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} Truncated text
 */
export const truncate = (text, maxLength = 50) => {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + '...';
};

/**
 * Debounce function calls
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
export const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

/**
 * Format file size in human-readable format
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted file size
 */
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
};

/**
 * Validate epsilon value
 * @param {number} epsilon - Epsilon value to validate
 * @returns {boolean} Whether epsilon is valid
 */
export const isValidEpsilon = (epsilon) => {
  return typeof epsilon === 'number' && epsilon >= 0 && epsilon <= 1;
};

/**
 * Get attack type display name
 * @param {string} attackType - Attack type ('fgsm' or 'pgd')
 * @returns {string} Display name
 */
export const getAttackDisplayName = (attackType) => {
  const names = {
    fgsm: 'FGSM',
    pgd: 'PGD',
  };
  return names[attackType] || attackType.toUpperCase();
};
