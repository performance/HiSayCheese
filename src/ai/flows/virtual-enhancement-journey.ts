'use server';

/**
 * @fileOverview A flow for generating an animated virtual hand making adjustments to an image with educational tooltips.
 *
 * - generateEnhancementJourney - A function that generates the enhancement journey.
 * - GenerateEnhancementJourneyInput - The input type for the generateEnhancementJourney function.
 * - GenerateEnhancementJourneyOutput - The return type for the generateEnhancementJourney function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const GenerateEnhancementJourneyInputSchema = z.object({
  photoDataUri: z
    .string()
    .describe(
      "A photo to be enhanced, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
  brightness: z.number().describe('The brightness level to apply to the image.'),
  contrast: z.number().describe('The contrast level to apply to the image.'),
  saturation: z.number().describe('The saturation level to apply to the image.'),
  backgroundBlur: z.number().describe('The background blur level to apply to the image.'),
  faceSmoothing: z.number().describe('The face smoothing level to apply to the image.'),
});
export type GenerateEnhancementJourneyInput = z.infer<typeof GenerateEnhancementJourneyInputSchema>;

const GenerateEnhancementJourneyOutputSchema = z.object({
  enhancedPhotoDataUri: z
    .string()
    .describe("The enhanced photo, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."),
  enhancementSteps: z.array(z.string()).describe('A list of enhancement steps with tooltips.'),
});
export type GenerateEnhancementJourneyOutput = z.infer<typeof GenerateEnhancementJourneyOutputSchema>;

export async function generateEnhancementJourney(input: GenerateEnhancementJourneyInput): Promise<GenerateEnhancementJourneyOutput> {
  return generateEnhancementJourneyFlow(input);
}

const prompt = ai.definePrompt({
  name: 'generateEnhancementJourneyPrompt',
  input: {schema: GenerateEnhancementJourneyInputSchema},
  output: {schema: GenerateEnhancementJourneyOutputSchema},
  prompt: `You are an AI assistant that enhances images and provides a visual journey of the enhancements.

You will receive a photo and a set of enhancement parameters. You will enhance the photo based on these parameters, and you will generate a series of enhancement steps that describe the changes made to the image, along with tooltips explaining the impact of each enhancement.

Here is the photo to be enhanced:
{{media url=photoDataUri}}

Here are the enhancement parameters:
Brightness: {{brightness}}
Contrast: {{contrast}}
Saturation: {{saturation}}
Background Blur: {{backgroundBlur}}
Face Smoothing: {{faceSmoothing}}

Respond in JSON format.
`,
});

const generateEnhancementJourneyFlow = ai.defineFlow(
  {
    name: 'generateEnhancementJourneyFlow',
    inputSchema: GenerateEnhancementJourneyInputSchema,
    outputSchema: GenerateEnhancementJourneyOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
