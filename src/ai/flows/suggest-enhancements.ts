'use server';

/**
 * @fileOverview An AI agent that analyzes an image and suggests optimal enhancement settings, including rationale.
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
  brightness: z.number().min(0).max(1).describe('The suggested brightness setting (0-1).'),
  contrast: z.number().min(0).max(1).describe('The suggested contrast setting (0-1).'),
  saturation: z.number().min(0).max(1).describe('The suggested saturation setting (0-1).'),
  backgroundBlur: z.number().min(0).max(1).describe('The suggested background blur setting (0-1).'),
  faceSmoothing: z.number().min(0).max(1).describe('The suggested face smoothing setting (0-1).'),
  rationale: z.object({
      brightness: z.string().describe('Explanation for the suggested brightness value.'),
      contrast: z.string().describe('Explanation for the suggested contrast value.'),
      saturation: z.string().describe('Explanation for the suggested saturation value.'),
      backgroundBlur: z.string().describe('Explanation for the suggested background blur value.'),
      faceSmoothing: z.string().describe('Explanation for the suggested face smoothing value.'),
  }).describe('Rationale behind each suggested enhancement value.')
});
export type SuggestEnhancementsOutput = z.infer<typeof SuggestEnhancementsOutputSchema>;

export async function suggestEnhancements(input: SuggestEnhancementsInput): Promise<SuggestEnhancementsOutput> {
  return suggestEnhancementsFlow(input);
}

const prompt = ai.definePrompt({
  name: 'suggestEnhancementsPrompt',
  input: {schema: SuggestEnhancementsInputSchema},
  output: {schema: SuggestEnhancementsOutputSchema},
  prompt: `You are an AI expert in image processing and enhancement, specializing in professional headshots. Analyze the provided image and suggest optimal enhancement settings (values between 0.0 and 1.0) to improve its quality.

Provide numerical suggestions for:
- brightness (0=dark, 0.5=normal, 1=bright)
- contrast (0=low, 0.5=normal, 1=high)
- saturation (0=grayscale, 0.5=normal, 1=vibrant)
- backgroundBlur (0=none, 1=max, subtle values are usually best)
- faceSmoothing (0=none, 1=max, subtle values are usually best)

Also, for EACH setting, provide a brief (1-sentence) rationale explaining *why* you chose that specific value based on your analysis of the image (e.g., "Slightly increased brightness to better illuminate the subject's face," or "Subtle blur added to draw focus to the subject.").

Ensure the suggested values are appropriate for a professional headshot, avoiding overly artificial effects.

Respond ONLY with the JSON object matching the output schema.

Image for analysis:
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
    if (!output) {
        throw new Error('AI failed to generate enhancement suggestions.');
    }
    // Basic validation (optional, AI schema should handle this)
    const validatedOutput = SuggestEnhancementsOutputSchema.parse(output);
    return validatedOutput;
  }
);