
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
  frontFacingScore: z.number().min(0).max(10).describe("Score from 0 (not front-facing at all) to 10 (perfectly front-facing)."),
  eyeVisibilityScore: z.number().min(0).max(10).describe("Score from 0 (eyes not visible or closed) to 10 (both eyes clearly visible and open)."),
  obstructionScore: z.number().min(0).max(10).describe("Score from 0 (no obstructions like glasses, hats, hair) to 10 (significant obstructions covering key facial features). Lower is better for obstructions."),
  overallSuitabilityScore: z.number().min(0).max(10).describe("Overall suitability score as a professional headshot based on pose, clarity, and obstructions. 0 (unsuitable) to 10 (excellent)."),
  feedback: z.array(z.string()).describe("A list of specific feedback points regarding the image quality, pose, eye visibility, and any obstructions noted."),
});
export type ImageQualityAssessmentOutput = z.infer<typeof ImageQualityAssessmentOutputSchema>;

export async function assessImageQuality(input: ImageQualityAssessmentInput): Promise<ImageQualityAssessmentOutput> {
  return imageQualityAssessmentFlow(input);
}

const prompt = ai.definePrompt({
  name: 'imageQualityAssessmentPrompt',
  input: {schema: ImageQualityAssessmentInputSchema},
  output: {schema: ImageQualityAssessmentOutputSchema},
  prompt: `You are an AI expert in evaluating images for their suitability as professional headshots. Analyze the provided image based on the following criteria:

1.  **Front-Facing Pose**: How directly is the subject facing the camera? A slight angle is acceptable, but the face should be generally oriented towards the viewer. Score 0-10.
2.  **Eye Visibility**: Are both eyes clearly visible, open, and free from shadows or hair? Are they looking towards the camera or slightly off-camera in an engaging way? Score 0-10.
3.  **Obstructions**: Are there any items obstructing the face, such as sunglasses, regular eyeglasses (note if they cause glare or hide eyes), hats, headscarves, or excessive hair covering the face? Score 0-10 (0 means no obstructions, 10 means highly obstructed).
4.  **Overall Suitability**: Based on the above, and general image clarity (focus, lighting on face), provide an overall score for its use as a professional headshot. Score 0-10.

Provide specific feedback points as an array of strings. For example, if eyes are closed, mention it. If a hat is worn, mention it. If the pose is not front-facing, describe it.

Respond ONLY with the JSON object matching the output schema.

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
