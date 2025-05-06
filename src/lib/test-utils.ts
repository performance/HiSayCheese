/**
 * Portrait Image Generator for Testing
 *
 * This script generates URLs for placeholder portrait images with varying quality characteristics
 * for testing image quality assessment features. It uses a combination of:
 * 1. Placeholder image services
 * 2. Pre-selected URLs to royalty-free portrait images with different qualities
 */

// Define a type for the quality scores
interface QualityScores {
  frontFacingPose: number;
  eyeVisibility: number;
  lightingQuality: number;
  focusSharpness: number;
  backgroundAppropriateness: number;
  expressionAppropriateness: number;
  framing: number;
  obstructions: number;
}

// Define a type for a test image object
export interface TestImage {
  url: string;
  quality: "good" | "medium" | "poor" | "placeholder";
  description: string;
  qualityScores: QualityScores;
  overallQualityScore: number;
}

// Function to create an array of test images
export function generateTestImages(): TestImage[] {
 
  // 1. GOOD QUALITY IMAGES
  // Well-lit, properly framed, front-facing headshots with good quality
  const goodQualityImages: TestImage[] = [
    {
      url: "https://images.pexels.com/photos/614810/pexels-photo-614810.jpeg",
      quality: "good",
      description: "Professional headshot with good lighting, clear face, neutral background",
      qualityScores: {
        frontFacingPose: 0.9,
        eyeVisibility: 0.95,
        lightingQuality: 0.9,
        focusSharpness: 0.85,
        backgroundAppropriateness: 0.9,
        expressionAppropriateness: 0.85,
        framing: 0.9,
        obstructions: 0.05  // Lower score means fewer obstructions
      }, 
      overallQualityScore: 0
    },
    {
      url: "https://images.pexels.com/photos/1222271/pexels-photo-1222271.jpeg",
      quality: "good",
      description: "Front-facing portrait with good lighting and framing",
      qualityScores: {
        frontFacingPose: 0.9,
        eyeVisibility: 0.9,
        lightingQuality: 0.85,
        focusSharpness: 0.9,
        backgroundAppropriateness: 0.8,
        expressionAppropriateness: 0.9,
        framing: 0.95,
        obstructions: 0.1
      }, 
      overallQualityScore: 0
    },
    {
      url: "https://images.unsplash.com/photo-1494790108377-be9c29b29330",
      quality: "good",
      description: "Professional female portrait with excellent lighting and framing",
      qualityScores: {
        frontFacingPose: 0.95,
        eyeVisibility: 0.95,
        lightingQuality: 0.95,
        focusSharpness: 0.9,
        backgroundAppropriateness: 0.9,
        expressionAppropriateness: 0.9,
        framing: 0.95,
        obstructions: 0.05
      }, 
      overallQualityScore: 0
    }
  ]; 

  // 2. MEDIUM QUALITY IMAGES
  // Images with minor issues (slightly off framing, lighting not ideal, etc.)
  const mediumQualityImages: TestImage[] = [
    {
      url: "https://images.pexels.com/photos/91227/pexels-photo-91227.jpeg",
      quality: "medium",
      description: "Portrait with slightly uneven lighting and casual pose",
      qualityScores: {
        frontFacingPose: 0.7,
        eyeVisibility: 0.8,
        lightingQuality: 0.6,
        focusSharpness: 0.7,
        backgroundAppropriateness: 0.6,
        expressionAppropriateness: 0.7,
        framing: 0.7,
        obstructions: 0.3
      }, 
      overallQualityScore: 0
    },
    {
      url: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d",
      quality: "medium",
      description: "Slightly off-angle portrait with somewhat distracting background",
      qualityScores: {
        frontFacingPose: 0.6,
        eyeVisibility: 0.7,
        lightingQuality: 0.7,
        focusSharpness: 0.75,
        backgroundAppropriateness: 0.5,
        expressionAppropriateness: 0.6,
        framing: 0.6,
        obstructions: 0.3
      }, 
      overallQualityScore: 0
    },
    {
      url: "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg",
      quality: "medium",
      description: "Portrait with good focus but slightly unusual framing",
      qualityScores: {
        frontFacingPose: 0.8,
        eyeVisibility: 0.85,
        lightingQuality: 0.7,
        focusSharpness: 0.8,
        backgroundAppropriateness: 0.7,
        expressionAppropriateness: 0.75,
        framing: 0.5,
        obstructions: 0.2
      }, 
      overallQualityScore: 0
    }
  ]; 

  // 3. POOR QUALITY IMAGES
  // Images with significant issues (bad lighting, blurry, obstructions, etc.)
  const poorQualityImages: TestImage[] = [
    {
      url: "https://images.unsplash.com/photo-1542909168-82c3e7fdca5c",
      quality: "poor",
      description: "Dark lighting with shadows on face",
      qualityScores: {
        frontFacingPose: 0.4,
        eyeVisibility: 0.5,
        lightingQuality: 0.3,
        focusSharpness: 0.4,
        backgroundAppropriateness: 0.3,
        expressionAppropriateness: 0.4,
        framing: 0.4,
        obstructions: 0.7
      }, 
      overallQualityScore: 0
    },
    {
      url: "https://images.pexels.com/photos/1036623/pexels-photo-1036623.jpeg",
      quality: "poor",
      description: "Portrait with obstructions (sunglasses) and distracting background",
      qualityScores: {
        frontFacingPose: 0.5,
        eyeVisibility: 0.2, // Eyes obstructed
        lightingQuality: 0.6,
        focusSharpness: 0.5,
        backgroundAppropriateness: 0.2, // Distracting background
        expressionAppropriateness: 0.5,
        framing: 0.6,
        obstructions: 0.8 // High obstructions
      }, 
      overallQualityScore: 0
    },
    {
      url: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e",
      quality: "poor",
      description: "Off-angle portrait with unusual cropping and expression",
      qualityScores: {
        frontFacingPose: 0.3, // Off-angle
        eyeVisibility: 0.6,
        lightingQuality: 0.5,
        focusSharpness: 0.4,
        backgroundAppropriateness: 0.4,
        expressionAppropriateness: 0.3, // Unusual expression
        framing: 0.2, // Unusual cropping
        obstructions: 0.4
      }, 
      overallQualityScore: 0
    }
  ]; 

  // 4. PLACEHOLDER IMAGES (using placeholder.com service)
  // These are basic placeholders with different dimensions to test loading aspects
  const placeholderImages: TestImage[] = [
    {
      url: "https://via.placeholder.com/400x600?text=Portrait+Placeholder",
      quality: "placeholder",
      description: "Basic placeholder image for testing loading functionality",
      qualityScores: { // Placeholder scores - can be neutral
        frontFacingPose: 0.5, eyeVisibility: 0.5, lightingQuality: 0.5,
        focusSharpness: 0.5, backgroundAppropriateness: 0.5, expressionAppropriateness: 0.5,
        framing: 0.5, obstructions: 0.5
      },
        overallQualityScore: 0
    },  
    {
      url: "https://via.placeholder.com/800x1200?text=High+Resolution+Portrait",
      quality: "placeholder",
      description: "High resolution placeholder for testing large image loading",
      qualityScores: { // Placeholder scores - can be neutral
        frontFacingPose: 0.5, eyeVisibility: 0.5, lightingQuality: 0.5,
        focusSharpness: 0.5, backgroundAppropriateness: 0.5, expressionAppropriateness: 0.5,
        framing: 0.5, obstructions: 0.5
      }, 
        overallQualityScore: 0
    },  
    {
      url: "https://via.placeholder.com/150x200?text=Low+Res+Portrait",
      quality: "placeholder",
      description: "Low resolution placeholder for testing small image handling",
      qualityScores: { // Placeholder scores - can be neutral
        frontFacingPose: 0.5, eyeVisibility: 0.5, lightingQuality: 0.5,
        focusSharpness: 0.5, backgroundAppropriateness: 0.5, expressionAppropriateness: 0.5,
        framing: 0.5, obstructions: 0.5
      }, 
        overallQualityScore: 0
    }
  ]; 

  // Add quality assessment criteria to each image
  return [...goodQualityImages, ...mediumQualityImages, ...poorQualityImages, ...placeholderImages].map(img => {

    // Create base object
    const enhancedImg: TestImage = {...img}; // Explicitly type

    // Calculate overall quality score (weighted average)
    const weights = {
      frontFacingPose: 0.15,
      eyeVisibility: 0.15,
      lightingQuality: 0.15,
      focusSharpness: 0.15,
      backgroundAppropriateness: 0.1,
      expressionAppropriateness: 0.1,
      framing: 0.1,
      obstructions: 0.1
    };

    let overallScore = 0;
    for (const [key, weight] of Object.entries(weights)) {
      // Ensure key is a valid key of QualityScores
      const score = key === 'obstructions'
        ? 1 - enhancedImg.qualityScores[key as keyof QualityScores] // For obstructions, lower is better, so we invert
        : enhancedImg.qualityScores[key as keyof QualityScores];
      overallScore += score * weight;
    }

    enhancedImg.overallQualityScore = Math.round(overallScore * 100) / 100;

    return enhancedImg;
  });
}

// Export the function
export default generateTestImages;
