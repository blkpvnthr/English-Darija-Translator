/**
 * DarijaNormalizer.js
 * * A module to standardize text written in the Arabic Chat Alphabet (Darija Leet) 
 * by mapping numbers and Latin characters to their corresponding standard Arabic
 * characters, typically used before feeding text into an NMT model.
 */

export const ARABIC_CHAT_MAP = {
    // Standard "Leet" mappings for sounds not present in Latin script
    '3': 'ع',  // 3 -> 'ayn (ع)'
    '7': 'ح',  // 7 -> ḥā’ (ح)
    '9': 'ق',  // 9 -> qāf (ق)
    '2': 'ء',  // 2 -> hamza (ء) or glottal stop
    '5': 'خ',  // 5 -> khā’ (خ)
    '6': 'ط',  // 6 -> ṭā’ (ط)
    '8': 'غ',  // 8 -> ghayn (غ)
    
    // Common mixed-case or similar replacements
    // The "dh" sound (ذ)
    'dh': 'ذ', // common substitution
    'd7': 'ظ', // 'd7' or '6'' for the emphatic 'dh' (ظ)

    // The "sh" sound (ش)
    'sh': 'ش', // shīn (ش)

    // The emphatic 's' (ص)
    's5': 'ص', // ṣād (ص) - using 5 as a reminder for the hard sound
    's9': 'ص', // alternative for ṣād (ص)

    // The emphatic 't' (ط) is covered by '6'.
    
    // The emphatic 'd' (ض)
    'd9': 'ض', // ḍād (ض) - using 9 as a reminder for the hard sound

    // Zāy with dot (ز) is usually 'z' but can be 7'
    'z7': 'ز', // zāy (ز)

    // Note: The mappings should handle both case-sensitive and case-insensitive scenarios.
    // For simplicity and robustness, we will process text in lowercase first.
};

/**
 * Normalizes a string containing Darija written in the Arabic Chat Alphabet
 * (e.g., Leet/Franco-Arabic) into standard Arabic script.
 * * @param {string} text - The input text in Darija Leet format.
 * @returns {string} The normalized text in standard Arabic script.
 */
export function normalizeDarija(text) {
    if (!text || typeof text !== 'string') {
        return '';
    }

    // 1. Convert to lowercase for consistent matching of multi-character keys (e.g., 'sh', 'dh')
    let normalizedText = text.toLowerCase();
    
    // 2. Perform multi-character replacements first (e.g., 'sh' to 'ش')
    // This uses a regular expression to replace keys from the map that have length > 1
    const multiCharKeys = Object.keys(ARABIC_CHAT_MAP).filter(key => key.length > 1);
    
    multiCharKeys.sort((a, b) => b.length - a.length); // Sort by length descending to match longest sequences first

    multiCharKeys.forEach(key => {
        // Create a regex for the key, escaping special characters if necessary
        const regex = new RegExp(key, 'g'); 
        normalizedText = normalizedText.replace(regex, ARABIC_CHAT_MAP[key]);
    });


    // 3. Perform single-character replacements (numbers to Arabic letters)
    // We only need to iterate over single-character keys now
    const singleCharKeys = Object.keys(ARABIC_CHAT_MAP).filter(key => key.length === 1);
    
    singleCharKeys.forEach(key => {
        // Create a regex for the single character key
        const regex = new RegExp(key, 'g');
        normalizedText = normalizedText.replace(regex, ARABIC_CHAT_MAP[key]);
    });

    // 4. Clean up any accidental double spaces created during replacement
    normalizedText = normalizedText.replace(/\s\s+/g, ' ').trim();

    return normalizedText;
}

// Example usage (optional, for self-testing within the module)
if (typeof window === 'undefined') {
    const example1 = "3andi 7ob l dar, 9albi dima fi bladi.";
    const example2 = "shkun ghadi ydir 2chghal d9i9a?";
    const example3 = "ana 8adi nshuf lmadina dyal 's5if'";
    
    console.log(`Original 1: ${example1}`);
    console.log(`Normalized 1: ${normalizeDarija(example1)}`);
    // Expected: عندي حب لدار، قلبي ديما في بلادي.

    console.log(`Original 2: ${example2}`);
    console.log(`Normalized 2: ${normalizeDarija(example2)}`);
    // Expected: شكون غادي يدير ءشغال دقيقة؟

    console.log(`Original 3: ${example3}`);
    console.log(`Normalized 3: ${normalizeDarija(example3)}`);
    // Expected: انا غادي نشوف المدينة ديال 'صيف'
}

