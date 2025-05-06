
'use client';

import type { ChangeEvent } from 'react';
import React, { useState, useRef, useEffect } from 'react';
import Image from 'next/image';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Upload, Camera, WandSparkles, CheckSquare, Linkedin, Users } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';
import { suggestEnhancements, SuggestEnhancementsOutput } from '@/ai/flows/suggest-enhancements';
import { generateEnhancementJourney, GenerateEnhancementJourneyOutput } from '@/ai/flows/virtual-enhancement-journey'; // Import AI flow


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
    setShowBeforeAfter(false); // Ensure enhanced view is active
    setUploadedImageIsAiEnhanced(false); // Suggestions don't mean AI enhanced final image yet

    // Scroll image into view before starting animation
    imageContainerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    // Add a small delay to allow scrolling to start/complete
    await new Promise(resolve => setTimeout(resolve, 300));


    try {
      const suggestions: SuggestEnhancementsOutput = await suggestEnhancements({ photoDataUri: originalImage });

      const animationDurationPerSlider = 400; // ms

      // Animate sliders one by one
      await animateSlider('brightness', suggestions.brightness, animationDurationPerSlider);
      await animateSlider('contrast', suggestions.contrast, animationDurationPerSlider);
      await animateSlider('saturation', suggestions.saturation, animationDurationPerSlider);
      await animateSlider('backgroundBlur', suggestions.backgroundBlur, animationDurationPerSlider);
      await animateSlider('faceSmoothing', suggestions.faceSmoothing, animationDurationPerSlider);


      toast({
        title: "AI Suggestions Applied",
        description: "Enhancement sliders updated with AI recommendations.",
      });
    } catch (error) {
      console.error("Error suggesting enhancements:", error);
      toast({
        title: "AI Error",
        description: "Could not get AI enhancement suggestions. Please try again.",
        variant: "destructive",
      });
       // Reset sliders if AI fails
      setEnhancementValues(initialEnhancements);
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

     // Scroll image into view before starting AI processing
     imageContainerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      // Add a small delay
     await new Promise(resolve => setTimeout(resolve, 300));


     try {
       const journeyResult = await generateEnhancementJourney({
         photoDataUri: originalImage, // Send original image for enhancement
         ...enhancementValues, // Send current slider values
       });
       setUploadedImage(journeyResult.enhancedPhotoDataUri); // Update image with AI-enhanced version
       setEnhancementJourney(journeyResult); // Store the journey steps
       setUploadedImageIsAiEnhanced(true); // Mark that the image is now AI-enhanced
       setShowBeforeAfter(false); // Ensure the 'enhanced' view shows the new AI image
       toast({
         title: "Enhancement Applied",
         description: "Image enhanced by AI. See the visual journey steps below!",
       });
     } catch (error) {
       console.error("Error applying enhancement journey:", error);
       toast({
         title: "Enhancement Error",
         description: "Could not apply AI enhancements. Please try again.",
         variant: "destructive",
       });
       setUploadedImage(originalImage); // Revert to original image on error
       setUploadedImageIsAiEnhanced(false);
     } finally {
       setIsProcessingEnhancement(false);
     }

   };

   const toggleBeforeAfter = () => {
      if (!originalImage || !uploadedImage) return; // Only allow toggle if both images exist

     // If we are showing 'Before' (original) OR if the current view is NOT AI enhanced,
     // then switch to showing the 'Enhanced' (uploadedImage, which might be AI enhanced or just preview)
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
     return {
       filter: `
         brightness(${1 + (enhancementValues.brightness - 0.5) * 1})
         contrast(${1 + (enhancementValues.contrast - 0.5) * 1})
         saturate(${1 + (enhancementValues.saturation - 0.5) * 1})
       `,
       // Background blur and face smoothing are complex and applied by AI
     };
   };


   // --- Virtual Hand Animation Placeholder ---
   // (Keep as placeholder for now, requires significant effort)
   const VirtualHand = () => {
     const [position, setPosition] = useState({ x: 50, y: 50 });
     useEffect(() => {}, [enhancementJourney]); // Trigger based on journey

     return (
       <div
         className="absolute text-4xl transition-transform duration-500 ease-in-out pointer-events-none" // Added pointer-events-none
        style={{ top: `${position.y}%`, left: `${position.x}%`, transform: 'translate(-50%, -50%)' }} // Center hand on position
        aria-hidden="true"
       >
         <span role="img" aria-label="Hand pointing">👉</span>
       </div>
     );
   };

   // --- Tooltip Component Placeholder ---
    const EnhancementTooltip = ({ label, value }: { label: string; value: number }) => (
      <div className="text-xs text-muted-foreground mt-1">
        {label}: {value.toFixed(2)}
      </div>
    );


  return (
    <div className="flex flex-col min-h-screen bg-secondary">
      {/* Header */}
      <header className="bg-card border-b shadow-sm sticky top-0 z-10">
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

      {/* Main Content */}
      <main className="flex-grow container mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Image Container */}
        <Card
            ref={imageContainerRef} // Assign ref here
            className="lg:col-span-2 h-[60vh] lg:h-auto flex flex-col items-center justify-center p-4 relative overflow-hidden scroll-mt-20" // Added scroll-mt-20 for sticky header offset
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
        >
          {uploadedImage ? (
            <div className="relative w-full h-full max-w-full max-h-full flex items-center justify-center">
              <Image
                src={showBeforeAfter ? originalImage! : uploadedImage} // Show original or enhanced based on state
                alt={showBeforeAfter ? "Original image" : (uploadedImageIsAiEnhanced ? "AI Enhanced image" : "Uploaded image preview")} // Dynamic alt text
                fill
                style={{ objectFit: 'contain', ...getImageStyle() }} // Use contain and apply styles
                className="transition-all duration-300"
                data-ai-hint="professional headshot"
                priority // Prioritize loading the main image
              />
              {/* Placeholder for Virtual Hand */}
              {/* {enhancementJourney && !isProcessingEnhancement && <VirtualHand />} */}

               {/* Before/After Toggle Button */}
               {originalImage && uploadedImage && uploadedImageIsAiEnhanced && ( // Show only when AI enhanced image exists
                 <Button
                   variant="outline"
                   size="sm"
                   className="absolute top-2 right-2 z-10 bg-card/80 hover:bg-card"
                   onClick={toggleBeforeAfter}
                   disabled={isProcessingEnhancement || isLoadingAI} // Disable during processing
                 >
                   {showBeforeAfter ? 'Show AI Enhanced' : 'Show Original'}
                 </Button>
               )}

               {/* Processing Overlay */}
               {(isLoadingAI || isProcessingEnhancement) && (
                  <div className="absolute inset-0 bg-background/80 flex flex-col items-center justify-center z-20 rounded-lg"> {/* Added rounded-lg */}
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
                    <p className="text-foreground text-lg font-medium"> {/* Added font-medium */}
                      {isLoadingAI ? 'AI Analyzing & Animating...' : 'Applying AI Enhancements...'}
                    </p>
                  </div>
                )}
            </div>
          ) : (
            <div className="text-center text-muted-foreground flex flex-col items-center justify-center h-full border-2 border-dashed border-border rounded-lg p-8">
              <Upload className="w-12 h-12 mb-4 text-primary" /> {/* Added text-primary */}
              <p className="mb-2 font-medium">Drag & drop your image here</p> {/* Added font-medium */}
              <p className="mb-4 text-sm">or</p>
              <div className="flex flex-col sm:flex-row gap-4"> {/* Responsive button layout */}
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
          {/* Placeholder for overlays: Face Detection, Smoothing Mask */}
        </Card>


        {/* Controls Container */}
        <Card className="h-fit sticky top-24 self-start"> {/* Make controls sticky & align top */}
          <CardHeader>
            <CardTitle>Enhancement Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {(Object.keys(initialEnhancements) as Array<keyof EnhancementValues>).map((key) => (
              <div key={key} className="space-y-2">
                <div className="flex justify-between items-center">
                   <Label htmlFor={key} className="capitalize font-medium"> {/* Added font-medium */}
                     {key.replace(/([A-Z])/g, ' $1').replace('Background B', 'Bg B')} {/* Improve label formatting */}
                   </Label>
                   <span className="text-sm font-medium text-primary w-10 text-right">{enhancementValues[key].toFixed(2)}</span> {/* Fixed width & right align */}
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
                  className="[&>span>span]:transition-transform [&>span>span]:duration-100 [&>span>span]:ease-linear" // Add smooth transition for animation
                />
                {/* Placeholder for educational tooltips */}
                {/* <EnhancementTooltip label={key} value={enhancementValues[key]} /> */}
              </div>
            ))}
          </CardContent>
           <CardFooter className="flex flex-col gap-4 pt-4"> {/* Added pt-4 */}
              <Button
                 onClick={handleSuggestEnhancements}
                 disabled={!originalImage || isLoadingAI || isProcessingEnhancement} // Disable if no original image
                 className="w-full"
               >
                 <WandSparkles className="mr-2" /> Suggest & Animate (AI)
                 {isLoadingAI && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-foreground ml-2"></div>}
               </Button>
               <Button
                 onClick={handleApplyEnhancements}
                 disabled={!originalImage || isLoadingAI || isProcessingEnhancement} // Disable if no original image
                 className="w-full bg-accent hover:bg-accent/90 text-accent-foreground" // Ensure contrast
               >
                 Apply AI & See Journey
                 {isProcessingEnhancement && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-accent-foreground ml-2"></div>}
               </Button>
            </CardFooter>
        </Card>

         {/* Enhancement Journey/Results Panel */}
         {enhancementJourney && !isProcessingEnhancement && (
            <Card className="mt-8 lg:col-span-3">
               <CardHeader><CardTitle>Enhancement Journey Steps</CardTitle></CardHeader>
               <CardContent>
                 <ul className="list-decimal pl-5 space-y-2 text-sm"> {/* Use decimal list */}
                   {enhancementJourney.enhancementSteps.map((step, index) => (
                     <li key={index}>{step}</li> // Render steps directly
                   ))}
                 </ul>
               </CardContent>
             </Card>
          )}
      </main>

      {/* Footer */}
      <footer className="bg-card border-t mt-auto">
        <div className="container mx-auto px-4 py-4 text-center text-muted-foreground text-sm">
          &copy; {new Date().getFullYear()} Headshot Handcrafter. All rights reserved.
        </div>
      </footer>
    </div>
  );
}

    