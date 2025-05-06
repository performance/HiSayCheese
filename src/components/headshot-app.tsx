
'use client';

import type { ChangeEvent } from 'react';
import React, { useState, useRef, useEffect } from 'react';
import Image from 'next/image';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Upload, Camera, WandSparkles, CheckSquare, Linkedin, Users, Info } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';
import { suggestEnhancements, SuggestEnhancementsOutput } from '@/ai/flows/suggest-enhancements';
import { generateEnhancementJourney, GenerateEnhancementJourneyOutput } from '@/ai/flows/virtual-enhancement-journey'; // Import AI flow
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'; // Import Tooltip components


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

// --- Main Component ---
export default function HeadshotApp() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [originalImage, setOriginalImage] = useState<string | null>(null); // Store original for comparison
  const [enhancementValues, setEnhancementValues] = useState<EnhancementValues>(initialEnhancements);
  const [mode, setMode] = useState<string>('professional'); // Default mode
  const [isLoadingAI, setIsLoadingAI] = useState(false);
  const [isProcessingEnhancement, setIsProcessingEnhancement] = useState(false);
  const [showBeforeAfter, setShowBeforeAfter] = useState(false); // State for before/after view
  const [enhancementJourney, setEnhancementJourney] = useState<GenerateEnhancementJourneyOutput | null>(null);
  const [uploadedImageIsAiEnhanced, setUploadedImageIsAiEnhanced] = useState(false); // Track if current view is AI enhanced
  const [suggestionRationale, setSuggestionRationale] = useState<SuggestionRationale | null>(null); // State for rationale


  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageContainerRef = useRef<HTMLDivElement>(null); // Ref for image container card
  const { toast } = useToast();

  const handleImageUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      // Basic validation (can be expanded)
      if (file.size > 10 * 1024 * 1024) { // Example: Limit to 10MB
        toast({
          title: "File Too Large",
          description: "Please upload an image smaller than 10MB.",
          variant: "destructive",
        });
        return;
      }

      const reader = new FileReader();
      reader.onloadend = () => {
        const dataUri = reader.result as string;
        setUploadedImage(dataUri);
        setOriginalImage(dataUri); // Store original on upload
        setShowBeforeAfter(false); // Reset view on new image
        setUploadedImageIsAiEnhanced(false); // Reset AI enhancement flag
        setEnhancementJourney(null); // Reset journey steps
        setSuggestionRationale(null); // Reset rationale
        // Reset enhancements or keep them based on desired UX
        // setEnhancementValues(initialEnhancements);
         toast({
           title: "Image Uploaded",
           description: "Ready for enhancement.",
         });
      };
      reader.readAsDataURL(file);
    } else if (file) {
       toast({
         title: "Invalid File Type",
         description: "Please upload a valid image file (JPG, PNG, WEBP).",
         variant: "destructive",
       });
    }
     // Reset file input value to allow uploading the same file again
     if (fileInputRef.current) {
       fileInputRef.current.value = "";
     }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('border-primary'); // Remove drag over styling

    const file = event.dataTransfer.files?.[0];
     if (file && file.type.startsWith('image/')) {
       // Reuse the same validation and loading logic
       if (file.size > 10 * 1024 * 1024) {
         toast({
           title: "File Too Large",
           description: "Please upload an image smaller than 10MB.",
           variant: "destructive",
         });
         return;
       }
       const reader = new FileReader();
       reader.onloadend = () => {
         const dataUri = reader.result as string;
         setUploadedImage(dataUri);
         setOriginalImage(dataUri);
         setShowBeforeAfter(false);
         setUploadedImageIsAiEnhanced(false); // Reset AI enhancement flag
         setEnhancementJourney(null); // Reset journey steps
         setSuggestionRationale(null); // Reset rationale
         toast({
           title: "Image Uploaded",
           description: "Ready for enhancement.",
         });
       };
       reader.readAsDataURL(file);
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
     event.currentTarget.classList.add('border-primary'); // Add drag over styling
   };

   const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
     event.preventDefault();
     event.stopPropagation();
     event.currentTarget.classList.remove('border-primary'); // Remove drag over styling
   };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  // Placeholder for camera functionality
  const handleCameraCapture = () => {
    toast({
      title: "Camera Feature",
      description: "Camera capture is not yet implemented.",
    });
    // Implementation would involve using navigator.mediaDevices.getUserMedia
  };

  const handleSliderChange = (key: keyof EnhancementValues, value: number[]) => {
    setEnhancementValues((prev) => ({ ...prev, [key]: value[0] }));
    // Live preview is handled by CSS filters now
  };

  const animateSlider = (key: keyof EnhancementValues, targetValue: number, duration: number = 500) => {
    return new Promise<void>((resolve) => {
      const startValue = enhancementValues[key];
      const startTime = performance.now();

      const step = (currentTime: number) => {
        const elapsedTime = currentTime - startTime;
        const progress = Math.min(elapsedTime / duration, 1);
        const currentValue = startValue + (targetValue - startValue) * progress;

        setEnhancementValues((prev) => ({ ...prev, [key]: currentValue }));

        if (progress < 1) {
          requestAnimationFrame(step);
        } else {
          // Ensure final value is set precisely
          setEnhancementValues((prev) => ({ ...prev, [key]: targetValue }));
          resolve();
        }
      };

      requestAnimationFrame(step);
    });
  };


  const handleSuggestEnhancements = async () => {
    if (!originalImage) { // Use original image for suggestions
      toast({
        title: "No Image",
        description: "Please upload an image first.",
        variant: "destructive",
      });
      return;
    }
    setIsLoadingAI(true);
    setShowBeforeAfter(false); // Ensure enhanced preview is active if we apply filters
    setUploadedImageIsAiEnhanced(false); // Suggestions don't mean AI enhanced final image yet
    setEnhancementJourney(null); // Clear previous journey steps
    setSuggestionRationale(null); // Clear previous rationale


    // Scroll image into view before starting animation
    // imageContainerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    // Add a small delay to allow scrolling to start/complete
    // await new Promise(resolve => setTimeout(resolve, 300)); // No longer needed with flex layout


    try {
      const suggestions: SuggestEnhancementsOutput = await suggestEnhancements({ photoDataUri: originalImage });

      const animationDurationPerSlider = 400; // ms

      setSuggestionRationale(suggestions.rationale); // Store rationale

      // Animate sliders one by one to suggested values
      await animateSlider('brightness', suggestions.brightness, animationDurationPerSlider);
      await animateSlider('contrast', suggestions.contrast, animationDurationPerSlider);
      await animateSlider('saturation', suggestions.saturation, animationDurationPerSlider);
      await animateSlider('backgroundBlur', suggestions.backgroundBlur, animationDurationPerSlider);
      await animateSlider('faceSmoothing', suggestions.faceSmoothing, animationDurationPerSlider);


      toast({
        title: "AI Suggestions Applied",
        description: "Sliders animated to suggested values. Hover over (i) for rationale.",
      });
    } catch (error) {
      console.error("Error suggesting enhancements:", error);
      toast({
        title: "AI Error",
        description: `Could not get AI enhancement suggestions. ${error instanceof Error ? error.message : ''}`,
        variant: "destructive",
      });
       // Reset sliders if AI fails - Optional: Maybe keep the user's last values?
      // setEnhancementValues(initialEnhancements);
      setSuggestionRationale(null);
    } finally {
      setIsLoadingAI(false);
    }
  };

  const handleApplyEnhancements = async () => {
     if (!originalImage) { // Ensure original image exists
       toast({
         title: "No Image",
         description: "Please upload an image first.",
         variant: "destructive",
       });
       return;
     }
     setIsProcessingEnhancement(true);
     setEnhancementJourney(null); // Reset journey
     setShowBeforeAfter(false); // Ensure enhanced view is active
     // Keep rationale visible if it exists, don't clear it here
     // setSuggestionRationale(null);


     // Scroll image into view before starting AI processing
     // imageContainerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      // Add a small delay
     // await new Promise(resolve => setTimeout(resolve, 300)); // No longer needed


     try {
       const journeyResult = await generateEnhancementJourney({
         photoDataUri: originalImage, // Send original image for enhancement
         ...enhancementValues, // Send CURRENT slider values
       });
       setUploadedImage(journeyResult.enhancedPhotoDataUri); // Update image with AI-enhanced version
       setEnhancementJourney(journeyResult); // Store the journey steps
       setUploadedImageIsAiEnhanced(true); // Mark that the image is now AI-enhanced
       setShowBeforeAfter(false); // Ensure the 'enhanced' view shows the new AI image
       toast({
         title: "Enhancement Applied",
         description: "Image enhanced by AI using current settings. See journey steps.",
       });
     } catch (error) {
       console.error("Error applying enhancement journey:", error);
       toast({
         title: "Enhancement Error",
         description: `Could not apply AI enhancements. ${error instanceof Error ? error.message : ''}`,
         variant: "destructive",
       });
       // Don't revert image here, keep showing the preview based on sliders
       // setUploadedImage(originalImage);
       setUploadedImageIsAiEnhanced(false);
     } finally {
       setIsProcessingEnhancement(false);
     }

   };

   const toggleBeforeAfter = () => {
      if (!originalImage || !uploadedImage) return; // Only allow toggle if both images exist

     // If we are showing 'Before' (original) OR if the current view is NOT AI enhanced,
     // then switch to showing the 'Enhanced' (uploadedImage, which must be AI enhanced)
     if (showBeforeAfter || !uploadedImageIsAiEnhanced) {
        setShowBeforeAfter(false);
     } else {
     // Otherwise, if we are showing the AI-enhanced image, switch to showing the 'Before' (original)
        setShowBeforeAfter(true);
     }
   }

   // --- Apply CSS Filters for Live Preview (Simplified) ---
   const getImageStyle = (): React.CSSProperties => {
      // If showing 'Before' view OR if the image is already AI enhanced (don't double-apply filters)
     if (showBeforeAfter || uploadedImageIsAiEnhanced) return {};

     // Apply filters only for live preview before AI enhancement
     // Note: Blur and Smoothing are complex, AI applies them, CSS filters are just basic previews
     return {
       filter: `
         brightness(${1 + (enhancementValues.brightness - 0.5) * 1})
         contrast(${1 + (enhancementValues.contrast - 0.5) * 1})
         saturate(${1 + (enhancementValues.saturation - 0.5) * 1})
         blur(${(enhancementValues.backgroundBlur * 5)}px)
       `,
       // Face smoothing is not feasible with pure CSS filters in a realistic way.
     };
   };


   // --- Virtual Hand Animation Placeholder ---
   // (Removed as focus is on core functionality)


   // --- Rationale Tooltip ---
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
           <TooltipContent side="top" align="start" className="max-w-xs text-xs p-2 z-50"> {/* Added z-index */}
             <p>{rationale}</p>
           </TooltipContent>
         </Tooltip>
       </TooltipProvider>
     );
   };


  return (
    <div className="flex flex-col h-screen bg-secondary"> {/* Use h-screen */}
      {/* Header */}
      <header className="bg-card border-b shadow-sm flex-shrink-0"> {/* Removed sticky, added flex-shrink-0 */}
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-primary">Headshot Handcrafter</h1>
          <div className="flex items-center gap-4">
             <Select value={mode} onValueChange={setMode}>
                <SelectTrigger className="w-[180px]">
                  <SelectValue placeholder="Select Mode" />
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
              {/* Add Pro tier button/indicator later */}
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-grow container mx-auto px-4 py-8 flex flex-col lg:flex-row gap-8 overflow-hidden"> {/* Use flex-grow and overflow-hidden */}

        {/* Image Container */}
        <div
            ref={imageContainerRef}
            className="flex-grow lg:w-2/3 h-full flex flex-col items-center justify-center p-4 relative bg-card rounded-lg border shadow-sm overflow-hidden" // Adjusted styles
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
        >
          {uploadedImage ? (
            <div className="relative w-full h-full flex items-center justify-center">
              <Image
                src={showBeforeAfter ? originalImage! : uploadedImage}
                alt={showBeforeAfter ? "Original image" : (uploadedImageIsAiEnhanced ? "AI Enhanced image" : "Uploaded image preview")}
                fill
                style={{ objectFit: 'contain', ...getImageStyle() }}
                className="transition-all duration-300"
                data-ai-hint="professional headshot"
                priority
              />
               {/* Before/After Toggle Button */}
               {originalImage && uploadedImage && uploadedImageIsAiEnhanced && (
                 <Button
                   variant="outline"
                   size="sm"
                   className="absolute top-2 right-2 z-10 bg-card/80 hover:bg-card"
                   onClick={toggleBeforeAfter}
                   disabled={isProcessingEnhancement || isLoadingAI}
                 >
                   {showBeforeAfter ? 'Show AI Enhanced' : 'Show Original'}
                 </Button>
               )}

               {/* Processing Overlay */}
               {(isLoadingAI || isProcessingEnhancement) && (
                  <div className="absolute inset-0 bg-background/80 flex flex-col items-center justify-center z-20 rounded-lg">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
                    <p className="text-foreground text-lg font-medium">
                      {isLoadingAI ? 'AI Analyzing & Animating...' : 'Applying AI Enhancements...'}
                    </p>
                  </div>
                )}
            </div>
          ) : (
            <div className="text-center text-muted-foreground flex flex-col items-center justify-center h-full border-2 border-dashed border-border rounded-lg p-8">
              <Upload className="w-12 h-12 mb-4 text-primary" />
              <p className="mb-2 font-medium">Drag & drop your image here</p>
              <p className="mb-4 text-sm">or</p>
              <div className="flex flex-col sm:flex-row gap-4">
                <Button onClick={triggerFileInput}>
                  <Upload className="mr-2" /> Upload Image
                </Button>
                <Button variant="outline" onClick={handleCameraCapture}>
                  <Camera className="mr-2" /> Use Camera
                </Button>
              </div>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageUpload}
                accept="image/jpeg,image/png,image/webp"
                className="hidden"
              />
              <p className="mt-4 text-xs">Supports JPG, PNG, WEBP (Max 10MB)</p>
            </div>
          )}
        </div>


        {/* Controls Container - Allow vertical scroll */}
        <div className="lg:w-1/3 flex flex-col gap-4 overflow-y-auto pr-2"> {/* Use remaining space or fixed width, add overflow-y-auto */}
          <Card className="flex-shrink-0"> {/* Basic controls card */}
            <CardHeader>
              <CardTitle>Enhancement Controls</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {(Object.keys(initialEnhancements) as Array<keyof EnhancementValues>).map((key) => (
                <div key={key} className="space-y-2">
                  <div className="flex justify-between items-center">
                     <Label htmlFor={key} className="capitalize font-medium flex items-center"> {/* Added flex items-center */}
                       {key.replace(/([A-Z])/g, ' $1').replace('Background B', 'Bg B')}
                       <RationaleTooltip rationale={suggestionRationale?.[key]} /> {/* Add rationale tooltip */}
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
                    disabled={!uploadedImage || isLoadingAI || isProcessingEnhancement}
                    aria-label={`${key} slider`}
                    className="[&>span:last-child]:transition-transform [&>span:last-child]:duration-100 [&>span:last-child]:ease-linear" // Animate thumb position
                  />
                </div>
              ))}
            </CardContent>
             <CardFooter className="flex flex-col gap-4 pt-4">
                <Button
                   onClick={handleSuggestEnhancements}
                   disabled={!originalImage || isLoadingAI || isProcessingEnhancement}
                   className="w-full"
                 >
                   <WandSparkles className="mr-2" /> Suggest & Animate (AI)
                   {isLoadingAI && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-foreground ml-2"></div>}
                 </Button>
                 <Button
                   onClick={handleApplyEnhancements}
                   disabled={!originalImage || isLoadingAI || isProcessingEnhancement}
                   className="w-full bg-accent hover:bg-accent/90 text-accent-foreground"
                 >
                   Apply AI & See Journey
                   {isProcessingEnhancement && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-accent-foreground ml-2"></div>}
                 </Button>
              </CardFooter>
          </Card>

           {/* Enhancement Journey/Results Panel */}
           {enhancementJourney && !isProcessingEnhancement && (
              <Card className="flex-shrink-0"> {/* Don't grow, just show content */}
                 <CardHeader><CardTitle>Enhancement Journey Steps</CardTitle></CardHeader>
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
      </main>

      {/* Footer */}
      <footer className="bg-card border-t mt-auto flex-shrink-0"> {/* Use flex-shrink-0 */}
        <div className="container mx-auto px-4 py-4 text-center text-muted-foreground text-sm">
          &copy; {new Date().getFullYear()} Headshot Handcrafter. All rights reserved.
        </div>
      </footer>
    </div>
  );
}
