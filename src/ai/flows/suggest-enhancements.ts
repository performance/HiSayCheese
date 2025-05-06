'use server';

/**
 * @fileOverview An AI agent that analyzes an image and suggests optimal enhancement settings.
 *
 * - suggestEnhancements - A function that handles the enhancement suggestion process.
 * - SuggestEnhancementsInput - The input type for the suggestEnhancements function.
 * - SuggestEnhancementsOutput - The return type for the suggestEnhancements function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const SuggestEnhancementsInputSchema = z.object({
  photoDataUri: z
    .string()
    .describe(
      "A photo to be analyzed, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
});
export type SuggestEnhancementsInput = z.infer<typeof SuggestEnhancementsInputSchema>;

const SuggestEnhancementsOutputSchema = z.object({
  brightness: z.number().describe('The suggested brightness setting.'),
  contrast: z.number().describe('The suggested contrast setting.'),
  saturation: z.number().describe('The suggested saturation setting.'),
  backgroundBlur: z.number().describe('The suggested background blur setting.'),
  faceSmoothing: z.number().describe('The suggested face smoothing setting.'),
});
export type SuggestEnhancementsOutput = z.infer<typeof SuggestEnhancementsOutputSchema>;

export async function suggestEnhancements(input: SuggestEnhancementsInput): Promise<SuggestEnhancementsOutput> {
  return suggestEnhancementsFlow(input);
}

const prompt = ai.definePrompt({
  name: 'suggestEnhancementsPrompt',
  input: {schema: SuggestEnhancementsInputSchema},
  output: {schema: SuggestEnhancementsOutputSchema},
  prompt: `You are an AI expert in image processing and enhancement. Analyze the provided image and suggest optimal enhancement settings to improve the overall quality and make it suitable as a professional headshot.

Consider factors such as lighting, contrast, color balance, and facial features.

Respond with the suggested values for brightness, contrast, saturation, background blur, and face smoothing.

Ensure the suggested values are within a reasonable range (e.g., 0 to 1 for most settings) and are appropriate for a professional headshot.

Here is the image for analysis:

{{media url=photoDataUri}}`,
});

const suggestEnhancementsFlow = ai.defineFlow(
  {
    name: 'suggestEnhancementsFlow',
    inputSchema: SuggestEnhancementsInputSchema,
    outputSchema: SuggestEnhancementsOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
