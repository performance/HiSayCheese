import {genkit} from 'genkit';
import {googleAI} from '@genkit-ai/googleai';

export const ai = genkit({
  plugins: [googleAI()],
  // model: 'googleai/gemini-2.5-pro-preview-04-17'
  model: 'googleai/gemini-2.0-flash',
});
