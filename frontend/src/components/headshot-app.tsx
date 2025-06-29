'use client';

import type { ChangeEvent } from 'react';
import ImageUpload from './image-upload';
import React, { useState, useRef, useEffect, ChangeEventHandler } from 'react';
import Image from 'next/image';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Upload, Camera, WandSparkles, CheckSquare, Linkedin, Users, Info, AlertCircle, CheckCircle2, Eye, Smile, UserX, Lightbulb, Aperture, Image as ImageIcon, Drama, Ratio, ArrowLeft, ArrowRight } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';
import { suggestEnhancements, SuggestEnhancementsOutput } from '@/ai/flows/suggest-enhancements';
import { generateEnhancementJourney, GenerateEnhancementJourneyOutput } from '@/ai/flows/virtual-enhancement-journey';
import { Checkbox } from '@/components/ui/checkbox';
import { assessImageQuality, ImageQualityAssessmentOutput } from '@/ai/flows/image-quality-assessment';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Skeleton } from '@/components/ui/skeleton';
import { Carousel, CarouselContent, CarouselItem, CarouselNext, CarouselPrevious, type CarouselApi } from '@/components/ui/carousel';
import generateTestImages, { type TestImage } from '@/lib/test-utils';


// --- Enhancement Controls Type ---
interface EnhancementValues {
  brightness: number;
  contrast: number;
  saturation: number;
  backgroundBlur: number;
  faceSmoothing: number;
}

// --- Initial State ---
const initialEnhancements: EnhancementValues = {
  brightness: 0.5,
  contrast: 0.5,
  saturation: 0.5,
  backgroundBlur: 0.2,
  faceSmoothing: 0.3,
};

type SuggestionRationale = SuggestEnhancementsOutput['rationale'];

// --- Carousel Slide Types ---
type CarouselSlide = {
  id: string;
  title: string;
  imageSrc: string | null;
  altText: string;
  caption?: React.ReactNode;
  isAiEnhanced?: boolean; // To differentiate AI enhanced image for styling/filtering
};


// --- Main Component ---
export default function HeadshotApp() {
  const [isClient, setIsClient] = useState(false);
  // --- State and Initial Values ---
  const [testImages, setTestImages] = useState<TestImage[]>([]);
  const [selectedTestImage, setSelectedTestImage] = useState<TestImage | null>(null);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [enhancementValues, setEnhancementValues] = useState<EnhancementValues>(initialEnhancements);
  const [mode, setMode] = useState<string>('professional');
  const [isLoadingAI, setIsLoadingAI] = useState(false);
  const [isProcessingEnhancement, setIsProcessingEnhancement] = useState(false);
  const [enhancementJourney, setEnhancementJourney] = useState<GenerateEnhancementJourneyOutput | null>(null);
  const [suggestionRationale, setSuggestionRationale] = useState<SuggestionRationale | null>(null);
  const [imageQualityAssessment, setImageQualityAssessment] = useState<ImageQualityAssessmentOutput | null>(null);
  const [carouselSlides, setCarouselSlides] = useState<CarouselSlide[]>([]);
  const [isAssessingQuality, setIsAssessingQuality] = useState(false);
  const [isUploading, setIsUploading] = useState(false); // New state for upload status
  const [uploadedObjectKey, setUploadedObjectKey] = useState<string | null>(null); // For S3 object key of original
  const [backendAnalysisResults, setBackendAnalysisResults] = useState<any | null>(null); // For results from /api/images/analyze
  const [backendEnhancedImageUrl, setBackendEnhancedImageUrl] = useState<string | null>(null); // For URL from /api/images/apply-enhancements
  const [manualToken, setManualToken] = useState(''); // For token input field
  const [authToken, setAuthToken] = useState<string | null>(null); // For currently active auth token


  const [isTestMode, setIsTestMode] = useState<boolean>(false);
  const [carouselApi, setCarouselApi] = useState<CarouselApi>();
  const [currentSlide, setCurrentSlide] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { toast } = useToast(); // toast needs to be available before handleSaveToken

  const handleSaveToken = () => {
    if (manualToken.trim() === "") {
      toast({
        title: "Token Empty",
        description: "Please paste a token before saving.",
        variant: "destructive",
      });
      return;
    }
    if (typeof window !== 'undefined') {
      localStorage.setItem('authToken', manualToken);
      setAuthToken(manualToken);
      toast({
        title: "Token Saved",
        description: "Authentication token has been updated and saved to localStorage.",
      });
    }
  };

  // Effect to set initial state and load token on client-side
    useEffect(() => {
      if (typeof window !== 'undefined') { // Ensure localStorage is available
        const storedToken = localStorage.getItem('authToken');
        if (storedToken) {
          setAuthToken(storedToken);
        }
      }
      if (isTestMode) {
            const generatedTestImages = generateTestImages();
            setTestImages(generatedTestImages);
            // Randomly select a test image only on the client-side
            const randomIndex = Math.floor(Math.random() * generatedTestImages.length);
            const randomTestImage = generatedTestImages[randomIndex];

            setSelectedTestImage(randomTestImage);
            setOriginalImage(randomTestImage.url);
            setUploadedImage(randomTestImage.url);

            setImageQualityAssessment({
                feedback: [] as string[],
                overallSuitabilityScore: randomTestImage.overallQualityScore,
                frontFacingScore: randomTestImage.qualityScores.frontFacingPose,
                eyeVisibilityScore: randomTestImage.qualityScores.eyeVisibility,
                lightingQualityScore: randomTestImage.qualityScores.lightingQuality,
                focusSharpnessScore: randomTestImage.qualityScores.focusSharpness,
                backgroundAppropriatenessScore: randomTestImage.qualityScores.backgroundAppropriateness,
                expressionAppropriatenessScore: randomTestImage.qualityScores.expressionAppropriateness,
                headToBodyRatioScore: 0.5, // Placeholder value
                obstructionScore: randomTestImage.qualityScores.obstructions,
            });
        }

      setTimeout(() => setIsClient(true), 10);
  }, [isTestMode]); // Added isTestMode to dependency array if it influences test image generation

  // Update carousel slides when relevant data changes
  useEffect(() => {
    if (!originalImage) {
      setCarouselSlides([]);
      setCurrentSlide(0);
      return;
    }

    const slides: CarouselSlide[] = [
      {
        id: 'original',
        title: 'Original Image',
        imageSrc: originalImage,
        altText: 'Original uploaded image',
      },
    ];

    if (imageQualityAssessment) {
      slides.push({
        id: 'quality-assessed',
        title: 'Image Quality Assessed',
        imageSrc: originalImage,
        altText: 'Image after quality assessment',
        caption: (
          <p className="text-xs text-muted-foreground p-2 bg-background/80 rounded-md mt-1">
            Quality assessed. Scores available in the panel.
            Overall: {imageQualityAssessment.overallSuitabilityScore}/10
          </p>
        ),
      });
    }

    if (suggestionRationale && !enhancementJourney) {
       slides.push({
        id: 'suggestions-ready',
        title: 'AI Suggestions Ready',
        imageSrc: originalImage, // Show original before applying
        altText: 'Image with AI suggestions ready',
        caption: (
          <p className="text-xs text-muted-foreground p-2 bg-background/80 rounded-md mt-1">
            AI suggestions loaded and sliders animated. Click "Apply AI" to see changes.
          </p>
        ),
      });
    }


    if (enhancementJourney?.enhancedPhotoDataUri && !backendEnhancedImageUrl) { // Show Genkit AI if backend one isn't there yet
      slides.push({
        id: 'ai-enhanced-genkit', // differentiate
        title: 'AI Enhanced Headshot (Genkit)',
        imageSrc: enhancementJourney.enhancedPhotoDataUri,
        altText: 'AI Enhanced headshot by Genkit',
        isAiEnhanced: true,
      });
    }
    if (backendEnhancedImageUrl) {
      slides.push({
        id: 'ai-enhanced-backend', // differentiate
        title: 'AI Enhanced Headshot (Backend)',
        imageSrc: backendEnhancedImageUrl,
        altText: 'AI Enhanced headshot by Backend',
        isAiEnhanced: true, // Use same styling for now
      });
    }
    setCarouselSlides(slides);
  }, [originalImage, imageQualityAssessment, suggestionRationale, enhancementJourney, backendEnhancedImageUrl]);

  // Effect for Carousel API
  useEffect(() => {
    if (!carouselApi) {
      return;
    }
    setCurrentSlide(carouselApi.selectedScrollSnap());
    carouselApi.on("select", () => {
      setCurrentSlide(carouselApi.selectedScrollSnap());
    });
     // Ensure carousel scrolls to the latest slide when new slides are added
    if (carouselSlides.length > 0) {
      // Debounce or delay slightly to allow DOM update
      setTimeout(() => carouselApi.scrollTo(carouselSlides.length - 1, false), 100);
    }
  }, [carouselApi, carouselSlides.length]);

  const handleTestImageSelect = (testImage: TestImage) => {
    setSelectedTestImage(testImage);
    // Clear any AI state for the new test image
    setEnhancementJourney(null);
    setSuggestionRationale(null);
    setOriginalImage(testImage.url);
    setUploadedImage(testImage.url);
    setImageQualityAssessment({
        feedback: [] as string[],
        overallSuitabilityScore: testImage.overallQualityScore,
        frontFacingScore: testImage.qualityScores.frontFacingPose,
        eyeVisibilityScore: testImage.qualityScores.eyeVisibility,
        lightingQualityScore: testImage.qualityScores.lightingQuality,
        focusSharpnessScore: testImage.qualityScores.focusSharpness,
        backgroundAppropriatenessScore: testImage.qualityScores.backgroundAppropriateness,
        expressionAppropriatenessScore: testImage.qualityScores.expressionAppropriateness,
        headToBodyRatioScore: 0.5, // Placeholder value
        obstructionScore: testImage.qualityScores.obstructions,
    });

      setCarouselSlides([
          {
              id: 'original',
              title: 'Original Image',
              imageSrc: testImage.url,
              altText: 'Original uploaded image',
          },
      ]);
  };

  // Refactored to handle File object and backend upload
  const processUploadedImage = async (file: File) => {
    setIsUploading(true);
    setIsAssessingQuality(false); // Reset this as assessment is part of this flow now
    setOriginalImage(null);
    setUploadedImage(null);
    setImageQualityAssessment(null);
    setEnhancementJourney(null);
    setSuggestionRationale(null);
    setEnhancementValues(initialEnhancements);
    setCarouselSlides([]); // Clear previous slides
    carouselApi?.scrollTo(0, true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/users/upload-image', { // Changed URL
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorDetail = "Unknown error during upload.";
        try {
          const errorJson = await response.json();
          errorDetail = errorJson.detail || errorDetail;
        } catch (e) {
          // Failed to parse error JSON
          errorDetail = response.statusText || errorDetail;
        }
        toast({
          title: "Upload Failed",
          description: errorDetail,
          variant: "destructive",
        });
        setOriginalImage(null); // Ensure states are cleared on failure
        setUploadedImage(null);
        setCarouselSlides([]);
        return; // Exit early
      }

      const uploadData = await response.json(); // Parse JSON response

      // Generate data URI for client-side operations (assessImageQuality)
      // This is kept because assessImageQuality likely needs a base64 data URI
      const dataUriForAssessment = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });

      // Use the URL from the backend response for display
      setOriginalImage(uploadData.file_url);
      setUploadedImage(uploadData.file_url);
      setUploadedObjectKey(uploadData.object_key); // Store the object key

      setCarouselSlides([
        { id: 'original', title: 'Original Image', imageSrc: uploadData.file_url, altText: 'Original uploaded image' },
      ]);

      toast({
        title: "Image Uploaded",
        description: "Successfully processed. Now performing backend analysis...",
      });

      // Call backend analysis endpoint
      if (uploadData.object_key) {
        if (!authToken) {
          toast({
            title: "Authentication Token Needed",
            description: "Please save an auth token to analyze the image with the backend.",
            variant: "destructive",
          });
        } else {
          try {
            const analysisResponse = await fetch('/api/images/analyze', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${authToken}`,
              },
              body: JSON.stringify({ object_key: uploadData.object_key }),
            });

            if (analysisResponse.ok) {
              const analysisData = await analysisResponse.json();
              setBackendAnalysisResults(analysisData);
              toast({
                title: "Backend Analysis Complete",
                description: "Face detection and quality metrics retrieved.",
              });
              // console.log("Backend Analysis Results:", analysisData);
            } else {
              const errorData = await analysisResponse.json();
              if (analysisResponse.status === 401) {
                toast({
                  title: "Auth Error During Analysis",
                  description: "Token might be invalid or expired. Please save a new token.",
                  variant: "destructive",
                });
              } else {
                toast({
                  title: "Backend Analysis Failed",
                  description: errorData.detail || "Could not analyze image via backend.",
                  variant: "destructive",
                });
              }
            }
          } catch (analysisError) {
            console.error("Error calling backend analysis:", analysisError);
            toast({
              title: "Backend Analysis Error",
              description: `Could not connect to the analysis service. ${analysisError instanceof Error ? analysisError.message : 'Please check your connection.'}`,
              variant: "destructive",
            });
          }
        }
      }
      // --- SYNTAX ERROR WAS HERE: An extra, misplaced `catch` block was removed from this spot ---

      // Now proceed with existing client-side AI quality assessment (can co-exist or be replaced)
      setIsAssessingQuality(true);
      try {
        const assessmentResult = await assessImageQuality({ photoDataUri: dataUriForAssessment });
        setImageQualityAssessment(assessmentResult);
        toast({
          title: "Client-side Quality Assessed",
          description: "Check the assessment panel for client-side details.",
        });
      } catch (error) {
        console.error("Error assessing image quality (client-side):", error);
        toast({
          title: "AI Assessment Error",
          description: `Could not assess image quality. ${error instanceof Error ? error.message : ''}`,
          variant: "destructive",
        });
        setImageQualityAssessment(null); // Clear assessment if it failed
      } finally {
        setIsAssessingQuality(false);
      }

    } catch (error) {
      console.error("Error uploading file:", error);
      toast({
        title: "Upload Error",
        description: `Could not upload image. ${error instanceof Error ? error.message : 'Please check your connection.'}`,
        variant: "destructive",
      });
      setOriginalImage(null);
      setUploadedImage(null);
      setCarouselSlides([]);
    } finally {
      setIsUploading(false);
    }
  };

  const handleImageUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        toast({
          title: "File Too Large",
          description: "Please upload an image smaller than 10MB.",
          variant: "destructive",
        });
        return;
      }
      // Call processUploadedImage with the File object
      processUploadedImage(file);
    } else if (file) {
       toast({
         title: "Invalid File Type",
         description: "Please upload a valid image file (JPG, PNG, WEBP).",
         variant: "destructive",
       });
    }
    // Reset file input to allow re-uploading the same file if needed
     if (fileInputRef.current) {
       fileInputRef.current.value = "";
     }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('border-primary');

    const file = event.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        toast({
          title: "File Too Large",
          description: "Please upload an image smaller than 10MB.",
          variant: "destructive",
        });
        return;
      }
      // Call processUploadedImage with the File object
      processUploadedImage(file);
    } else if (file) {
       toast({
         title: "Invalid File Type",
         description: "Please upload a valid image file (JPG, PNG, WEBP).",
         variant: "destructive",
       });
    }
  };

   const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
     event.preventDefault();
     event.stopPropagation();
     event.currentTarget.classList.add('border-primary');
   };

   const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
     event.preventDefault();
     event.stopPropagation();
     event.currentTarget.classList.remove('border-primary');
   };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleCameraCapture = () => {
    toast({
      title: "Camera Feature",
      description: "Camera capture is not yet implemented.",
    });
  };

  const handleSliderChange = (key: keyof EnhancementValues, value: number[]) => {
    setEnhancementValues((prev) => ({ ...prev, [key]: value[0] }));
    // When user manually changes slider, reset AI enhanced image preview if current slide shows it
    // And revert `uploadedImage` to original for client-side filtering, unless we are on the "AI Enhanced" slide specifically.
    const currentCarouselSlide = carouselSlides[currentSlide];
    if (currentCarouselSlide?.id !== 'ai-enhanced' && enhancementJourney) {
       setUploadedImage(originalImage); // Allow client-side preview with manual slider changes
    }
  };

  const animateSlider = (key: keyof EnhancementValues, targetValue: number, duration: number = 500) => {
    return new Promise<void>((resolve) => {
      setEnhancementValues(prev => {
          const startValue = prev[key];
          const startTime = performance.now();

          const step = (currentTime: number) => {
            const elapsedTime = currentTime - startTime;
            const progress = Math.min(elapsedTime / duration, 1);
            const currentValue = startValue + (targetValue - startValue) * progress;

            setEnhancementValues((prevStep) => ({ ...prevStep, [key]: currentValue }));

            if (progress < 1) {
              requestAnimationFrame(step);
            } else {
              setEnhancementValues((prevStep) => ({ ...prevStep, [key]: targetValue }));
              resolve();
            }
          };
          requestAnimationFrame(step);
          return prev; // Return previous state to avoid immediate re-render from this setter
      });
    });
  };


  const handleSuggestEnhancements = async () => {
    if (!originalImage) {
      toast({
        title: "No Image",
        description: "Please upload an image first.",
        variant: "destructive",
      });
      return;
    }
    setIsLoadingAI(true);
    setEnhancementJourney(null); // Clear previous AI journey
    setSuggestionRationale(null);
    setUploadedImage(originalImage); // Reset to original for client-side preview to take effect if user changes sliders


    try {
      const suggestions: SuggestEnhancementsOutput = await suggestEnhancements({ photoDataUri: originalImage });
      const animationDurationPerSlider = 400;
      setSuggestionRationale(suggestions.rationale);

      // Animate sliders to suggested values
      await animateSlider('brightness', suggestions.brightness, animationDurationPerSlider);
      await animateSlider('contrast', suggestions.contrast, animationDurationPerSlider);
      await animateSlider('saturation', suggestions.saturation, animationDurationPerSlider);
      await animateSlider('backgroundBlur', suggestions.backgroundBlur, animationDurationPerSlider);
      await animateSlider('faceSmoothing', suggestions.faceSmoothing, animationDurationPerSlider);

      toast({
        title: "AI Suggestions Loaded",
        description: "Sliders animated to suggested values. Hover over (i) for rationale. Click 'Apply AI' to see changes on the image.",
      });
      // After suggestions, ensure the "suggestions-ready" slide is shown if it exists
      const suggestionsSlideIndex = carouselSlides.findIndex(s => s.id === 'suggestions-ready');
      if (suggestionsSlideIndex !== -1) {
        setTimeout(() => carouselApi?.scrollTo(suggestionsSlideIndex, false), 100);
      }

    } catch (error) {
      console.error("Error suggesting enhancements:", error);
      toast({
        title: "AI Error",
        description: `Could not get AI enhancement suggestions. ${error instanceof Error ? error.message : ''}`,
        variant: "destructive",
      });
      setSuggestionRationale(null);
    } finally {
      setIsLoadingAI(false);
    }
  };

  const handleApplyEnhancements = async () => {
     if (!originalImage) {
       toast({
         title: "No Image",
         description: "Please upload an image first.",
         variant: "destructive",
       });
       return;
     }
     setIsProcessingEnhancement(true);
     setEnhancementJourney(null); // Clear previous Genkit journey
     setBackendEnhancedImageUrl(null); // Clear previous backend enhanced image

    if (!uploadedObjectKey) {
      toast({
        title: "Error",
        description: "No uploaded image object key found. Please upload image again.",
        variant: "destructive",
      });
      setIsProcessingEnhancement(false);
      return;
    }

    // Map frontend enhancementValues (0-1 range, 0.5 is normal) to backend expectations
    const backendEnhancements = {
      brightness_target: (enhancementValues.brightness - 0.5) * 100, // Maps 0-1 (0.5 normal) to -50 to 50 (0 normal)
      contrast_target: enhancementValues.contrast / 0.5,         // Maps 0-1 (0.5 normal) to 0-2 (1.0 normal)
      saturation_target: enhancementValues.saturation / 0.5,     // Maps 0-1 (0.5 normal) to 0-2 (1.0 normal)
      background_blur_radius: Math.round(enhancementValues.backgroundBlur * 25), // Maps 0-1 to 0-25 pixels
      face_smooth_intensity: enhancementValues.faceSmoothing,        // Direct map 0-1
      // crop_rect is not explicitly handled by sliders in this example, send null or omit
      crop_rect: null, // Or derive from some other state if crop feature exists
    };

    const requestBody = {
      original_object_key: uploadedObjectKey,
      enhancements: backendEnhancements,
      mode: mode, // Pass current mode
    };

    if (!authToken) {
      toast({
        title: "Authentication Token Needed",
        description: "Please save an auth token to apply enhancements with the backend.",
        variant: "destructive",
      });
      setIsProcessingEnhancement(false);
      return;
    }

    try {
      const response = await fetch('/api/images/apply-enhancements', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`,
        },
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        const enhancementData = await response.json();
        setBackendEnhancedImageUrl(enhancementData.enhanced_file_url);
        toast({
          title: "Enhancements Applied (Backend)",
          description: "Image successfully enhanced by the backend.",
        });
        // Scroll to the new backend-enhanced slide
        setTimeout(() => {
          const backendEnhancedSlideIndex = carouselSlides.findIndex(s => s.id === 'ai-enhanced-backend');
          if (backendEnhancedSlideIndex !== -1) {
            carouselApi?.scrollTo(backendEnhancedSlideIndex, false);
          } else { // If slide not yet created due to state update lag
             setTimeout(() => {
                const currentSlides = carouselSlides;
                const newIndex = currentSlides.findIndex(s => s.id === 'ai-enhanced-backend');
                if (newIndex !== -1) carouselApi?.scrollTo(newIndex, false);
             }, 300);
          }
        }, 100);
      } else {
        const errorData = await response.json();
        if (response.status === 401) {
          toast({
            title: "Auth Error During Enhancement",
            description: "Token might be invalid or expired. Please save a new token.",
            variant: "destructive",
          });
        } else {
          toast({
            title: "Backend Enhancement Failed",
            description: errorData.detail || "Could not apply enhancements via backend.",
            variant: "destructive",
          });
        }
      }
    } catch (error) {
      console.error("Error applying backend enhancements:", error);
      toast({
        title: "Enhancement Error",
        description: `Could not connect to backend enhancement service. ${error instanceof Error ? error.message : ''}`,
        variant: "destructive",
      });
    } finally {
      setIsProcessingEnhancement(false);
    }
   };

   const getImageStyle = (isAiEnhancedSlide: boolean | undefined): React.CSSProperties => {
     // If it's an AI-enhanced slide (either Genkit or Backend), or no original image, don't apply client-side filters.
     if (isAiEnhancedSlide || !originalImage) return {};
     // Otherwise, apply client-side filters based on slider values for preview on non-AI-enhanced slides.
     return {
       filter: `
         brightness(${1 + (enhancementValues.brightness - 0.5) * 1})
         contrast(${1 + (enhancementValues.contrast - 0.5) * 1})
         saturate(${1 + (enhancementValues.saturation - 0.5) * 1})
         blur(${(enhancementValues.backgroundBlur * 5)}px)
       `,
     };
   };


   const RationaleTooltip = ({ rationale }: { rationale: string | undefined }) => {
     if (!rationale) return null;
     return (
       <TooltipProvider>
         <Tooltip delayDuration={100}>
           <TooltipTrigger asChild>
             <span className="ml-1 cursor-help text-muted-foreground hover:text-foreground">
               <Info className="w-3 h-3" />
             </span>
           </TooltipTrigger>
           <TooltipContent side="top" align="start" className="max-w-xs text-xs p-2 z-[100]"> {/* Increased z-index */}
             <p>{rationale}</p>
           </TooltipContent>
         </Tooltip>
       </TooltipProvider>
     );
   };

  const QualityScoreIcon = ({ label, score, icon: Icon, lowIsGood = false, outOf = 10 }: { label: string, score: number, icon: React.ElementType, lowIsGood?: boolean, outOf?:number }) => {
    const displayScore = outOf === 10 ? `${score}/${outOf}` : `${(score * 100 / outOf).toFixed(0)}%`;
    let scoreColorClass = 'text-primary';
    const percentage = score * (100 / outOf);

    if ((!lowIsGood && percentage < 40) || (lowIsGood && percentage >= 70)) scoreColorClass = 'text-destructive';
    else if ((!lowIsGood && percentage < 70) || (lowIsGood && percentage >= 40)) scoreColorClass = 'text-yellow-500';

    return (
      <TooltipProvider>
        <Tooltip delayDuration={100}>
          <TooltipTrigger asChild>
            <div className="flex flex-col items-center text-center p-2 rounded-md hover:bg-accent/10 transition-colors cursor-default">
              <Icon className={`w-7 h-7 mb-1 ${lowIsGood && score > (outOf * 0.6) ? 'text-destructive' : (score < (outOf * 0.4) && !lowIsGood ? 'text-destructive' : 'text-muted-foreground')}`} />
              <span className={`font-semibold text-xs ${scoreColorClass}`}>{displayScore}</span>
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="max-w-xs text-xs p-2 z-[100]"> {/* Increased z-index */}
            <p>{label}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  };

  return (
    <div className="flex flex-col h-screen bg-secondary">
      <header className="bg-card border-b shadow-sm flex-shrink-0">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <WandSparkles className="w-4 h-4" />
            <h1 className="text-2xl font-bold text-primary">Headshot Handcrafter</h1>
          </div>
          <div className="flex items-center gap-2"> {/* Reduced gap-4 to gap-2 */}
            <input
              type="text"
              placeholder="Paste JWT Token"
              value={manualToken}
              onChange={(e) => setManualToken(e.target.value)}
              className="px-2 py-1 text-xs border rounded-md h-8 focus:ring-primary focus:border-primary" // Basic styling
            />
            <Button onClick={handleSaveToken} size="sm" variant="outline" className="h-8 text-xs">Save Token</Button>
            <div className="flex items-center gap-2 ml-2"> {/* Added ml-2 for spacing */}
                <Checkbox id="test-mode" onCheckedChange={setIsTestMode} />
                <Label htmlFor="test-mode" className="text-xs">Test Mode</Label>
            </div>
            <Select onValueChange={setMode}>
              <SelectTrigger className="w-[150px] h-8 text-xs"> {/* Adjusted width & height */}
                <SelectValue placeholder="Mode" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="professional">
                  <div className="flex items-center gap-2">
                    <WandSparkles className="w-4 h-4" /> Professional
                  </div>
                </SelectItem>
                <SelectItem value="passport">
                  <div className="flex items-center gap-2">
                    <CheckSquare className="w-4 h-4" /> Passport Photo
                  </div>
                </SelectItem>
                <SelectItem value="linkedin">
                  <div className="flex items-center gap-2">
                    <Linkedin className="w-4 h-4" /> LinkedIn Optimized
                  </div>
                </SelectItem>
                <SelectItem value="team">
                  <div className="flex items-center gap-2">
                    <Users className="w-4 h-4" /> Team/Corporate
                  </div>
                </SelectItem>
              </SelectContent>
               </Select>
          </div>
        </div>
      </header>
      <main className="flex-grow container mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-3 gap-8 overflow-hidden max-h-[calc(100vh-8rem)]">
          {(!isTestMode && !originalImage) ? (
          <div className="overflow-y-auto lg:col-span-2 flex flex-col items-center justify-center p-4 relative bg-card rounded-lg border shadow-sm overflow-hidden min-h-[calc(100vh-16rem)] h-full"
          > <ImageUpload
                handleImageUpload={handleImageUpload}
                handleDrop={handleDrop}
                handleDragOver={handleDragOver}
                handleDragLeave={handleDragLeave}
                triggerFileInput={triggerFileInput}
                fileInputRef={fileInputRef}
                isAssessingQuality={isAssessingQuality}
                isLoadingAI={isLoadingAI}
                isProcessingEnhancement={isProcessingEnhancement}
                isUploading={isUploading} // Pass new state
                testImages={testImages}
                selectedTestImage={selectedTestImage}
                handleTestImageSelect={handleTestImageSelect}
          />
          </div>
        ) : (
          <>
            <div className="overflow-y-auto lg:col-span-2 flex flex-col items-center justify-center p-4 relative bg-card rounded-lg border shadow-sm overflow-hidden min-h-[calc(100vh-16rem)] h-full">
              {isClient ? (
                <Carousel setApi={setCarouselApi} className="w-full h-full">
                  <CarouselContent className="h-full">
                    {carouselSlides.map((slide) => (
                      <CarouselItem key={slide.id} className="h-full">
                        <div className="relative w-full h-full flex items-center justify-center"> {/* Adjust height for caption */}
                          {slide.imageSrc && (
                            <Image
                              src={slide.imageSrc}
                              alt={slide.altText}
                              fill
                              style={{ objectFit: 'contain', ...getImageStyle(slide.isAiEnhanced) }}
                              className="transition-all duration-300"
                              data-ai-hint="professional headshot portrait"
                              sizes="(max-width: 768px) 100vw, (max-width: 1200px) 66vw, 50vw"
                              priority={slide.id === 'original'}
                            />
                          )}
                        </div>
                        {slide.caption && (
                            <div className="text-center text-sm mt-1 max-w-full truncate px-2">
                                {slide.caption}
                            </div>
                        )}
                      </CarouselItem>
                    ))}
                  </CarouselContent>
                  {carouselSlides.length > 1 && (
                    <>
                        <CarouselPrevious
                            variant="ghost"
                            className="absolute left-2 top-1/2 -translate-y-1/2 z-10 bg-card/50 hover:bg-card/80 disabled:opacity-30"
                            disabled={!carouselApi?.canScrollPrev() || isLoadingAI || isProcessingEnhancement || isAssessingQuality || isUploading}
                        />
                        <CarouselNext
                            variant="ghost"
                            className="absolute right-2 top-1/2 -translate-y-1/2 z-10 bg-card/50 hover:bg-card/80 disabled:opacity-30"
                            disabled={!carouselApi?.canScrollNext() || isLoadingAI || isProcessingEnhancement || isAssessingQuality || isUploading}
                        />
                    </>
                  )}
                  {(isLoadingAI || isProcessingEnhancement || isAssessingQuality || isUploading) && (
                      <div className="absolute inset-0 bg-background/80 flex flex-col items-center justify-center z-20 rounded-lg">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
                        <p className="text-foreground text-lg font-medium">
                          {isUploading ? 'Uploading Image...' : (isAssessingQuality ? 'AI Assessing Quality...' : (isLoadingAI ? 'AI Analyzing & Animating...' : 'Applying AI Enhancements...'))}
                        </p>
                      </div>
                  )}
                </Carousel>
              ): (
                <div className="flex flex-col items-center justify-center h-full">
                  <div className="animate-pulse rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
                  <p className="text-muted-foreground">Loading content...</p>
                </div>
              )}
            </div>
            {/* --- LAYOUT FIX: All right-side panels are now inside this single div which acts as the third grid column --- */}
            <div className="lg:col-span-1 flex flex-col gap-4 overflow-y-auto pr-1 pb-4 h-full">
              {/* Test Images Panel */}
              {isClient && isTestMode && (
                <Card className="flex-shrink-0">
                  <CardHeader>
                    <CardTitle className="text-xl">Test Images</CardTitle>
                  </CardHeader>
                  <CardContent className="grid grid-cols-3 gap-2">
                    {testImages.map((testImage, index) => (
                      <img key={index} src={testImage.url} alt={testImage.description}
                        className={`w-full h-24 object-cover cursor-pointer rounded-md ${selectedTestImage === testImage ? 'ring-2 ring-primary' : ''}`}
                        onClick={() => handleTestImageSelect(testImage)} />
                    ))}
                  </CardContent>
                </Card>
              )}

              {/* Image Quality Assessment Panel */}
              <Card className="flex-shrink-0">
                <CardHeader><CardTitle className="text-xl">Image Quality</CardTitle></CardHeader>
                <CardContent>
                  {isAssessingQuality && !imageQualityAssessment && (
                    <div className="grid grid-cols-4 md:grid-cols-8 gap-1">
                      {Array.from({ length: 9 }).map((_, i) => <Skeleton key={i} className="h-16 w-full" />)}
                    </div>
                  )}
                  {!isAssessingQuality && !originalImage && (
                    <p className="text-sm text-muted-foreground p-4 text-center">Upload an image for quality assessment.</p>
                  )}
                  {imageQualityAssessment && (
                    isClient && (<>
                      <div className="grid grid-cols-4 sm:grid-cols-8 gap-1 mb-3">
                        <QualityScoreIcon label="Front-Facing Pose" score={imageQualityAssessment.frontFacingScore} icon={Smile} />
                        <QualityScoreIcon label="Eye Visibility" score={imageQualityAssessment.eyeVisibilityScore} icon={Eye} />
                        <QualityScoreIcon label="Lighting Quality" score={imageQualityAssessment.lightingQualityScore} icon={Lightbulb} />
                        <QualityScoreIcon label="Focus/Sharpness" score={imageQualityAssessment.focusSharpnessScore} icon={Aperture} />
                        <QualityScoreIcon label="Background" score={imageQualityAssessment.backgroundAppropriatenessScore} icon={ImageIcon} />
                        <QualityScoreIcon label="Expression" score={imageQualityAssessment.expressionAppropriatenessScore} icon={Drama} />
                        <QualityScoreIcon label="Framing/Ratio" score={imageQualityAssessment.headToBodyRatioScore} icon={Ratio} />
                        <QualityScoreIcon label="Obstructions" score={imageQualityAssessment.obstructionScore} icon={UserX} lowIsGood={true} />
                      </div>
                      <div className="border-t pt-3 mt-3 text-center">
                        <Label className="text-sm font-medium text-muted-foreground mb-1 block">Overall Suitability</Label>
                        <QualityScoreIcon label="Overall Suitability" score={imageQualityAssessment.overallSuitabilityScore} icon={CheckCircle2} />
                      </div>

                      {imageQualityAssessment.feedback.length > 0 && (
                        <div className="mt-4 pt-3 border-t">
                          <h4 className="text-sm font-semibold mb-1">AI Feedback:</h4>
                          <ul className="list-disc pl-5 space-y-1 text-xs text-muted-foreground">
                            {imageQualityAssessment.feedback.map((item, index) => (
                              <li key={index}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                    </>)
                  )}
                </CardContent>
              </Card>

              {/* Enhancement Controls Panel */}
              <Card className="flex-shrink-0">
                <CardHeader>
                  <CardTitle className="text-xl">Enhancement Controls</CardTitle>
                </CardHeader>
                {isClient && (
                  <>
                    <CardContent className="space-y-4">
                      {(Object.keys(initialEnhancements) as Array<keyof EnhancementValues>).map((key) => (
                        <div key={key} className="space-y-2">
                          <div className="flex justify-between items-center">
                            <Label htmlFor={key} className="capitalize font-medium flex items-center text-sm">
                              {key.replace(/([A-Z])/g, ' $1').replace('Background B', 'Bg B')}
                              <RationaleTooltip rationale={suggestionRationale?.[key]} />
                            </Label>
                            <span className="text-sm font-medium text-primary w-10 text-right">{enhancementValues[key].toFixed(2)}</span>
                          </div>
                          <Slider
                            id={key}
                            min={0}
                            max={1}
                            step={0.01}
                            value={[enhancementValues[key]]}
                            onValueChange={(val) => handleSliderChange(key, val)}
                            disabled={!originalImage || isLoadingAI || isProcessingEnhancement || isAssessingQuality || isUploading}
                            aria-label={`${key} slider`}
                            className="[&>span:last-child]:transition-transform [&>span:last-child]:duration-100 [&>span:last-child]:ease-linear h-2"
                          />
                        </div>
                      ))}
                    </CardContent>
                    <CardFooter className="flex flex-col gap-3 pt-4">
                      <Button
                        onClick={handleSuggestEnhancements}
                        disabled={!originalImage || isLoadingAI || isProcessingEnhancement || isAssessingQuality || isUploading}
                        className="w-full"
                      >
                        <WandSparkles className="mr-2 h-4 w-4" /> Suggest & Animate (AI)
                        {isLoadingAI && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-foreground ml-2"></div>}
                      </Button>
                      <Button
                        onClick={handleApplyEnhancements}
                        disabled={!originalImage || isLoadingAI || isProcessingEnhancement || isAssessingQuality || isUploading}
                        className="w-full bg-accent hover:bg-accent/90 text-accent-foreground"
                      >
                        Apply AI & See Journey
                        {isProcessingEnhancement && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-accent-foreground ml-2"></div>}
                      </Button>
                    </CardFooter>
                  </>
                )}
              </Card>

              {enhancementJourney && carouselSlides[currentSlide]?.id === 'ai-enhanced' && !isProcessingEnhancement && (
                  <Card className="flex-shrink-0">
                     <CardHeader><CardTitle className="text-xl">Enhancement Journey</CardTitle></CardHeader>
                     <CardContent>
                       <ul className="list-decimal pl-5 space-y-2 text-sm">
                         {enhancementJourney.enhancementSteps.length > 0 ? (
                            enhancementJourney.enhancementSteps.map((step, index) => (
                              <li key={index}>{step}</li>
                            ))
                         ) : (
                            <li className="text-muted-foreground italic">No significant changes described by AI.</li>
                         )
                          }
                        </ul>
                      </CardContent>
                    </Card>
                  )}
              </div>
          </>
        )}
      </main>

      <footer className="bg-card border-t mt-auto flex-shrink-0">
        <div className="container mx-auto px-4 py-3 text-center text-muted-foreground text-xs">
          © {new Date().getFullYear()} Headshot Handcrafter. All rights reserved.
        </div>
      </footer>
    </div>
  );
}