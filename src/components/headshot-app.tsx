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
// import { generateEnhancementJourney, GenerateEnhancementJourneyOutput } from '@/ai/flows/virtual-enhancement-journey'; // Import AI flow


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
  // const [enhancementJourney, setEnhancementJourney] = useState<GenerateEnhancementJourneyOutput | null>(null);


  const fileInputRef = useRef<HTMLInputElement>(null);
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
    // Apply real-time visual update (can be complex, start with simple CSS filters)
    // For now, state update is sufficient, actual image processing happens on 'Apply' or auto
  };

  const handleSuggestEnhancements = async () => {
    if (!uploadedImage) {
      toast({
        title: "No Image",
        description: "Please upload an image first.",
        variant: "destructive",
      });
      return;
    }
    setIsLoadingAI(true);
    try {
      const suggestions: SuggestEnhancementsOutput = await suggestEnhancements({ photoDataUri: uploadedImage });
      setEnhancementValues({
        brightness: suggestions.brightness,
        contrast: suggestions.contrast,
        saturation: suggestions.saturation,
        backgroundBlur: suggestions.backgroundBlur,
        faceSmoothing: suggestions.faceSmoothing,
      });
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
    } finally {
      setIsLoadingAI(false);
    }
  };

  const handleApplyEnhancements = async () => {
     if (!uploadedImage || !originalImage) {
       toast({
         title: "No Image",
         description: "Please upload an image first.",
         variant: "destructive",
       });
       return;
     }
     setIsProcessingEnhancement(true);
     // setEnhancementJourney(null); // Reset journey

     // TODO: Integrate the actual generateEnhancementJourney AI flow
     // This section is commented out as it requires the AI flow to be fully functional and integrated.
     /*
     try {
       const journeyResult = await generateEnhancementJourney({
         photoDataUri: originalImage, // Send original image for enhancement
         ...enhancementValues,
       });
       setUploadedImage(journeyResult.enhancedPhotoDataUri); // Update image with enhanced version
       setEnhancementJourney(journeyResult); // Store the journey steps
       setShowBeforeAfter(true); // Show the before/after view
       toast({
         title: "Enhancement Applied",
         description: "Image enhanced successfully. See the visual journey!",
       });
     } catch (error) {
       console.error("Error applying enhancement journey:", error);
       toast({
         title: "Enhancement Error",
         description: "Could not apply enhancements. Please try again.",
         variant: "destructive",
       });
     } finally {
       setIsProcessingEnhancement(false);
     }
     */

     // --- Placeholder Logic (Remove when AI flow is integrated) ---
     // Simulate processing time
     await new Promise(resolve => setTimeout(resolve, 1500));
     // In a real scenario, you'd get the enhanced image URI from the AI
     // For now, we just toggle the before/after view to show the original again
     setShowBeforeAfter(true);
      toast({
        title: "Enhancement Processed (Placeholder)",
        description: "Visual journey simulation complete.",
      });
     setIsProcessingEnhancement(false);
     // --- End Placeholder Logic ---
   };

   const toggleBeforeAfter = () => {
     setShowBeforeAfter(prev => !prev);
   }

   // --- Apply CSS Filters for Live Preview (Simplified) ---
   const getImageStyle = (): React.CSSProperties => {
     if (showBeforeAfter) return {}; // Don't apply filters in 'before' view

     return {
       filter: `
         brightness(${1 + (enhancementValues.brightness - 0.5) * 1})
         contrast(${1 + (enhancementValues.contrast - 0.5) * 1})
         saturate(${1 + (enhancementValues.saturation - 0.5) * 1})
       `,
       // Background blur and face smoothing are more complex and typically require canvas or AI processing
     };
   };


   // --- Virtual Hand Animation Placeholder ---
   const VirtualHand = () => {
     // Basic placeholder, real implementation needs SVG and animation logic
     const [position, setPosition] = useState({ x: 50, y: 50 }); // Example position state

     useEffect(() => {
       // Simulate hand moving based on which slider is active or based on enhancementJourney steps
       // This is highly complex and requires mapping slider positions, etc.
     }, [enhancementValues]); // Or trigger based on enhancementJourney

     return (
       <div
         className="absolute text-4xl transition-transform duration-500 ease-in-out"
        //  style={{ transform: `translate(${position.x}px, ${position.y}px)` }}
        style={{ top: `${position.y}%`, left: `${position.x}%` }} // Example positioning
        aria-hidden="true"
       >
         <span role="img" aria-label="Hand pointing">👉</span>
       </div>
     );
   };

   // --- Tooltip Component Placeholder ---
    const EnhancementTooltip = ({ label, value }: { label: string; value: number }) => (
      <div className="text-xs text-muted-foreground mt-1">
        {label}: {value.toFixed(2)} {/* Example tooltip content */}
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
            className="lg:col-span-2 h-[60vh] lg:h-auto flex flex-col items-center justify-center p-4 relative overflow-hidden"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
        >
          {uploadedImage ? (
            <div className="relative w-full h-full max-w-full max-h-full flex items-center justify-center">
              <Image
                src={showBeforeAfter ? originalImage! : uploadedImage} // Show original or enhanced based on state
                alt={showBeforeAfter ? "Original image" : "Uploaded image preview"}
                fill
                style={{ objectFit: 'contain', ...getImageStyle() }} // Use contain and apply styles
                className="transition-all duration-300"
                data-ai-hint="professional headshot"
              />
              {/* Placeholder for Virtual Hand */}
              {/* {enhancementJourney && <VirtualHand />} */}

               {/* Before/After Toggle Button */}
               {originalImage && (
                 <Button
                   variant="outline"
                   size="sm"
                   className="absolute top-2 right-2 z-10 bg-card/80 hover:bg-card"
                   onClick={toggleBeforeAfter}
                 >
                   {showBeforeAfter ? 'Show Enhanced' : 'Show Original'}
                 </Button>
               )}

               {/* Processing Overlay */}
               {(isLoadingAI || isProcessingEnhancement) && (
                  <div className="absolute inset-0 bg-background/70 flex flex-col items-center justify-center z-20">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
                    <p className="text-foreground text-lg">
                      {isLoadingAI ? 'AI Analyzing...' : 'Applying Enhancements...'}
                    </p>
                  </div>
                )}
            </div>
          ) : (
            <div className="text-center text-muted-foreground flex flex-col items-center justify-center h-full border-2 border-dashed border-border rounded-lg p-8">
              <Upload className="w-12 h-12 mb-4" />
              <p className="mb-2">Drag & drop your image here</p>
              <p className="mb-4 text-sm">or</p>
              <div className="flex gap-4">
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
        <Card className="h-fit sticky top-24"> {/* Make controls sticky */}
          <CardHeader>
            <CardTitle>Enhancement Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {Object.entries(enhancementValues).map(([key, value]) => (
              <div key={key} className="space-y-2">
                <div className="flex justify-between items-center">
                   <Label htmlFor={key} className="capitalize">
                     {key.replace(/([A-Z])/g, ' $1').replace('Background B', 'Bg B')} {/* Improve label formatting */}
                   </Label>
                   <span className="text-sm font-medium text-primary">{value.toFixed(2)}</span>
                 </div>
                <Slider
                  id={key}
                  min={0}
                  max={1}
                  step={0.01}
                  value={[value]}
                  onValueChange={(val) => handleSliderChange(key as keyof EnhancementValues, val)}
                  disabled={!uploadedImage || isLoadingAI || isProcessingEnhancement}
                  aria-label={`${key} slider`}
                />
                {/* Placeholder for educational tooltips */}
                {/* <EnhancementTooltip label={key} value={value} /> */}
              </div>
            ))}
          </CardContent>
           <CardFooter className="flex flex-col gap-4">
              <Button
                 onClick={handleSuggestEnhancements}
                 disabled={!uploadedImage || isLoadingAI || isProcessingEnhancement}
                 className="w-full"
               >
                 <WandSparkles className="mr-2" /> Suggest Enhancements (AI)
                 {isLoadingAI && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-foreground ml-2"></div>}
               </Button>
               <Button
                 onClick={handleApplyEnhancements}
                 disabled={!uploadedImage || isLoadingAI || isProcessingEnhancement}
                 className="w-full bg-accent hover:bg-accent/90"
               >
                 Apply & See Journey
                 {isProcessingEnhancement && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-accent-foreground ml-2"></div>}
               </Button>
            </CardFooter>
        </Card>
         {/* Placeholder for Analysis Results Panel */}
         {/* {enhancementJourney && (
            <Card className="mt-8 lg:col-span-3">
               <CardHeader><CardTitle>Enhancement Journey</CardTitle></CardHeader>
               <CardContent>
                 <ul className="list-disc pl-5 space-y-2">
                   {enhancementJourney.enhancementSteps.map((step, index) => (
                     <li key={index}>{step}</li> // Render tooltips alongside
                   ))}
                 </ul>
               </CardContent>
             </Card>
          )} */}
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
