
'use server';

/**
 * @fileOverview A flow for generating an enhanced image and describing the enhancement steps.
 *
 * - generateEnhancementJourney - A function that generates the enhancement journey.
 * - GenerateEnhancementJourneyInput - The input type for the generateEnhancementJourney function.
 * - GenerateEnhancementJourneyOutput - The return type for the generateEnhancementJourney function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';
import {Tool} from 'genkit/tool';

// Define input schema for enhancement parameters
const EnhancementParamsSchema = z.object({
  brightness: z.number().min(0).max(1).describe('Brightness level (0=dark, 0.5=normal, 1=bright).'),
  contrast: z.number().min(0).max(1).describe('Contrast level (0=low, 0.5=normal, 1=high).'),
  saturation: z.number().min(0).max(1).describe('Saturation level (0=grayscale, 0.5=normal, 1=vibrant).'),
  backgroundBlur: z.number().min(0).max(1).describe('Background blur intensity (0=none, 1=max).'),
  faceSmoothing: z.number().min(0).max(1).describe('Face smoothing intensity (0=none, 1=max).'),
});

// Define input schema for the flow
const GenerateEnhancementJourneyInputSchema = z.object({
  photoDataUri: z
    .string()
    .describe(
      "A photo to be enhanced, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
}).merge(EnhancementParamsSchema); // Merge enhancement params into input

export type GenerateEnhancementJourneyInput = z.infer<typeof GenerateEnhancementJourneyInputSchema>;

// Define output schema
const GenerateEnhancementJourneyOutputSchema = z.object({
  enhancedPhotoDataUri: z
    .string()
    .describe("The enhanced photo, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."),
  enhancementSteps: z.array(z.string()).describe('A list of text descriptions explaining each enhancement step applied.'),
});
export type GenerateEnhancementJourneyOutput = z.infer<typeof GenerateEnhancementJourneyOutputSchema>;

// Exported wrapper function
export async function generateEnhancementJourney(input: GenerateEnhancementJourneyInput): Promise<GenerateEnhancementJourneyOutput> {
  return generateEnhancementJourneyFlow(input);
}

// Define the enhancement steps generation prompt separately for clarity
const enhancementStepsPrompt = ai.definePrompt({
    name: 'generateEnhancementStepsPrompt',
    input: { schema: EnhancementParamsSchema },
    output: { schema: z.object({ enhancementSteps: z.array(z.string()) }) },
    prompt: `Based on the following enhancement parameters, generate a short list of human-readable steps describing the adjustments made. Focus on the *impact* of each significant change (value significantly different from 0.5 or 0).

Parameters:
Brightness: {{brightness}}
Contrast: {{contrast}}
Saturation: {{saturation}}
Background Blur: {{backgroundBlur}}
Face Smoothing: {{faceSmoothing}}

Example for high contrast and slight blur:
- Increased contrast to make features stand out.
- Added a subtle background blur for better focus on the subject.

Respond ONLY with the JSON object containing the 'enhancementSteps' array.`,
});


// Define the main flow
const generateEnhancementJourneyFlow = ai.defineFlow(
  {
    name: 'generateEnhancementJourneyFlow',
    inputSchema: GenerateEnhancementJourneyInputSchema,
    outputSchema: GenerateEnhancementJourneyOutputSchema,
  },
  async (input) => {
    // 1. Generate the enhanced image using the experimental image generation model
    const enhancementInstructions = `Apply the following enhancements:
    - Brightness: Adjust to approximately ${input.brightness.toFixed(2)} (0=dark, 0.5=normal, 1=bright)
    - Contrast: Adjust to approximately ${input.contrast.toFixed(2)} (0=low, 0.5=normal, 1=high)
    - Saturation: Adjust to approximately ${input.saturation.toFixed(2)} (0=grayscale, 0.5=normal, 1=vibrant)
    - Background Blur: Apply blur with intensity around ${input.backgroundBlur.toFixed(2)} (0=none, 1=max), focusing on the main subject's face.
    - Face Smoothing: Apply subtle skin smoothing with intensity around ${input.faceSmoothing.toFixed(2)} (0=none, 1=max).
    Maintain a natural and professional look suitable for a headshot.`;

    console.log("Generating enhanced image with instructions:", enhancementInstructions);

    const { media } = await ai.generate({
      // IMPORTANT: Use the specific experimental model for image generation
      model: 'googleai/gemini-2.0-flash-exp',
      prompt: [
        { media: { url: input.photoDataUri } },
        { text: enhancementInstructions },
      ],
      config: {
        // IMPORTANT: Must request both TEXT and IMAGE modalities
        responseModalities: ['TEXT', 'IMAGE'],
        // Optional: Add safety settings if needed
        // safetySettings: [{ category: HarmCategory.HARASSMENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE }],
        // Optional: Adjust temperature for creativity (lower for more predictable results)
        temperature: 0.3,
      },
      // Removed unsupported output format specifier for this model
      // output: { format: 'json' }
    });

    const enhancedPhotoDataUri = media?.url;
    if (!enhancedPhotoDataUri) {
      throw new Error('Image generation failed or produced no output.');
    }

    console.log("Image generation successful.");


    // 2. Generate the enhancement steps description based *only* on the parameters
    console.log("Generating enhancement steps description...");
    const { output: stepsOutput } = await enhancementStepsPrompt({
        brightness: input.brightness,
        contrast: input.contrast,
        saturation: input.saturation,
        backgroundBlur: input.backgroundBlur,
        faceSmoothing: input.faceSmoothing,
    });

    if (!stepsOutput?.enhancementSteps) {
        console.warn("Failed to generate enhancement steps description, returning empty array.");
        return {
            enhancedPhotoDataUri: enhancedPhotoDataUri,
            enhancementSteps: [], // Return empty array if step generation fails
        };
    }

    console.log("Enhancement steps generated:", stepsOutput.enhancementSteps);


    // 3. Combine image and steps into the final output
    return {
      enhancedPhotoDataUri: enhancedPhotoDataUri,
      enhancementSteps: stepsOutput.enhancementSteps,
    };
  }
);

// Note: The actual image manipulation is now handled by the AI model based on the prompt.
// The 'enhancementSteps' are generated separately based on the input parameters to provide textual feedback.


