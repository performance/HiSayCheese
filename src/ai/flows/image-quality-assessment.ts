
'use server';

/**
 * @fileOverview An AI agent that assesses the quality of an image for use as a headshot.
 *
 * - assessImageQuality - A function that handles the image quality assessment.
 * - ImageQualityAssessmentInput - The input type for the assessImageQuality function.
 * - ImageQualityAssessmentOutput - The return type for the assessImageQuality function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const ImageQualityAssessmentInputSchema = z.object({
  photoDataUri: z
    .string()
    .describe(
      "A photo to be assessed, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
});
export type ImageQualityAssessmentInput = z.infer<typeof ImageQualityAssessmentInputSchema>;

const ImageQualityAssessmentOutputSchema = z.object({
  frontFacingScore: z.number().min(0).max(10).describe("Score from 0 (not front-facing) to 10 (perfectly front-facing)."),
  eyeVisibilityScore: z.number().min(0).max(10).describe("Score from 0 (eyes not visible/closed) to 10 (both eyes clearly visible and open)."),
  lightingQualityScore: z.number().min(0).max(10).describe("Score from 0 (poor lighting) to 10 (excellent, even facial lighting)."),
  focusSharpnessScore: z.number().min(0).max(10).describe("Score from 0 (blurry/out of focus) to 10 (sharp focus on face and eyes)."),
  backgroundAppropriatenessScore: z.number().min(0).max(10).describe("Score from 0 (distracting/inappropriate background) to 10 (clean, professional background)."),
  expressionAppropriatenessScore: z.number().min(0).max(10).describe("Score from 0 (inappropriate/unprofessional expression) to 10 (engaging, professional expression)."),
  headToBodyRatioScore: z.number().min(0).max(10).describe("Score from 0 (poor framing/composition) to 10 (good head & shoulders framing, well-centered)."),
  obstructionScore: z.number().min(0).max(10).describe("Score from 0 (no obstructions) to 10 (significant obstructions covering key facial features). Lower is better."),
  isLikelyPortrait: z.boolean().describe("True if the image is likely a portrait suitable for a headshot, based on face presence and prominence."),
  portraitCriteriaFeedback: z.string().optional().describe("Feedback specific to why an image may not be considered a good portrait (e.g., no face, face too small, not primary subject)."),
  hasProhibitedContent: z.boolean().describe("True if the image is likely to contain prohibited content based on SafeSearch type categories."),
  prohibitedContentFeedback: z.string().optional().describe("Feedback regarding any detected prohibited content categories."),
  overallSuitabilityScore: z.number().min(0).max(10).describe("Overall suitability score as a professional headshot. 0 (unsuitable) to 10 (excellent)."),
  feedback: z.array(z.string()).describe("A list of specific feedback points regarding all assessed criteria."),
});
export type ImageQualityAssessmentOutput = z.infer<typeof ImageQualityAssessmentOutputSchema>;

export async function assessImageQuality(input: ImageQualityAssessmentInput): Promise<ImageQualityAssessmentOutput> {
  return imageQualityAssessmentFlow(input);
}

const prompt = ai.definePrompt({
  name: 'imageQualityAssessmentPrompt',
  input: {schema: ImageQualityAssessmentInputSchema},
  output: {schema: ImageQualityAssessmentOutputSchema},
  prompt: `You are an AI expert in evaluating images for their suitability as professional headshots. Analyze the provided image based on the following criteria, providing a score from 0 to 10 for each:

1.  **Front-Facing Pose**: How directly is the subject facing the camera?
2.  **Eye Visibility**: Are both eyes clearly visible, open, and engaging?
3.  **Lighting Quality**: Is the face well-lit with even illumination and minimal harsh shadows?
4.  **Focus/Sharpness**: Is the image, especially the face and eyes, sharp and in focus?
5.  **Background Appropriateness**: Is the background uncluttered, non-distracting, and professional?
6.  **Expression Appropriateness**: Is the subject's expression suitable for a professional context (e.g., genuine, neutral, engaging)?
7.  **Head-to-Body Ratio/Framing**: Is the subject well-framed (typically head and shoulders) and centered?
8.  **Obstructions**: Are there any items obstructing the face (sunglasses, hats, hair, etc.)? (0=no obstructions, 10=highly obstructed - lower score is better).
9.  **Portrait Suitability**: Is the image primarily a portrait with a prominent face suitable for a headshot? Consider if a face is present, its size relative to the image, and if it's the main subject. Provide `isLikelyPortrait` (boolean) and `portraitCriteriaFeedback` (string, optional).
10. **Prohibited Content Check**: Does the image appear to contain any prohibited content (e.g., explicit nudity, graphic violence, hate symbols)? Provide `hasProhibitedContent` (boolean) and `prohibitedContentFeedback` (string, optional).
11. **Overall Suitability**: Based on all above, provide an overall suitability score.

Provide specific feedback points as an array of strings for each criterion, especially for scores that are not perfect. Ensure your feedback also covers portrait suitability and any prohibited content.

Respond ONLY with the JSON object matching the updated output schema.

Image for assessment:
{{media url=photoDataUri}}`,
});

const imageQualityAssessmentFlow = ai.defineFlow(
  {
    name: 'imageQualityAssessmentFlow',
    inputSchema: ImageQualityAssessmentInputSchema,
    outputSchema: ImageQualityAssessmentOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    if (!output) {
        throw new Error('AI failed to generate image quality assessment.');
    }
    // Basic validation (optional, AI schema should handle this)
    const validatedOutput = ImageQualityAssessmentOutputSchema.parse(output);
    return validatedOutput;
  }
);
