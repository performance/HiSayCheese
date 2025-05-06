
'use client';

import type { ChangeEvent } from 'react';
import React, { useState, useRef, useEffect } from 'react';
import Image from 'next/image';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Upload, Camera, WandSparkles, CheckSquare, Linkedin, Users, Info, AlertCircle, CheckCircle2, Eye, Smile, UserX } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';
import { suggestEnhancements, SuggestEnhancementsOutput } from '@/ai/flows/suggest-enhancements';
import { generateEnhancementJourney, GenerateEnhancementJourneyOutput } from '@/ai/flows/virtual-enhancement-journey';
import { assessImageQuality, ImageQualityAssessmentOutput } from '@/ai/flows/image-quality-assessment';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Progress } from '@/components/ui/progress'; // Import Progress
import { Skeleton } from '@/components/ui/skeleton'; // Import Skeleton

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
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [enhancementValues, setEnhancementValues] = useState<EnhancementValues>(initialEnhancements);
  const [mode, setMode] = useState<string>('professional');
  const [isLoadingAI, setIsLoadingAI] = useState(false);
  const [isProcessingEnhancement, setIsProcessingEnhancement] = useState(false);
  const [showBeforeAfter, setShowBeforeAfter] = useState(false);
  const [enhancementJourney, setEnhancementJourney] = useState<GenerateEnhancementJourneyOutput | null>(null);
  const [uploadedImageIsAiEnhanced, setUploadedImageIsAiEnhanced] = useState(false);
  const [suggestionRationale, setSuggestionRationale] = useState<SuggestionRationale | null>(null);
  
  const [imageQualityAssessment, setImageQualityAssessment] = useState<ImageQualityAssessmentOutput | null>(null);
  const [isAssessingQuality, setIsAssessingQuality] = useState(false);


  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageContainerRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const processUploadedImage = async (dataUri: string) => {
    setUploadedImage(dataUri);
    setOriginalImage(dataUri);
    setShowBeforeAfter(false);
    setUploadedImageIsAiEnhanced(false);
    setEnhancementJourney(null);
    setSuggestionRationale(null);
    setImageQualityAssessment(null); // Reset assessment

    toast({
      title: "Image Uploaded",
      description: "Ready for enhancement. Assessing image quality...",
    });

    setIsAssessingQuality(true);
    try {
      const assessmentResult = await assessImageQuality({ photoDataUri: dataUri });
      setImageQualityAssessment(assessmentResult);
      toast({
        title: "Image Quality Assessed",
        description: "Check the assessment panel for details.",
      });
    } catch (error) {
      console.error("Error assessing image quality:", error);
      toast({
        title: "AI Assessment Error",
        description: `Could not assess image quality. ${error instanceof Error ? error.message : ''}`,
        variant: "destructive",
      });
      setImageQualityAssessment(null);
    } finally {
      setIsAssessingQuality(false);
    }
  };

  const handleImageUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
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
        processUploadedImage(dataUri);
      };
      reader.readAsDataURL(file);
    } else if (file) {
       toast({
         title: "Invalid File Type",
         description: "Please upload a valid image file (JPG, PNG, WEBP).",
         variant: "destructive",
       });
    }
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
         processUploadedImage(dataUri);
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
          setEnhancementValues((prev) => ({ ...prev, [key]: targetValue }));
          resolve();
        }
      };
      requestAnimationFrame(step);
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
    setShowBeforeAfter(false);
    setUploadedImageIsAiEnhanced(false);
    setEnhancementJourney(null);
    setSuggestionRationale(null);

    try {
      const suggestions: SuggestEnhancementsOutput = await suggestEnhancements({ photoDataUri: originalImage });
      const animationDurationPerSlider = 400;
      setSuggestionRationale(suggestions.rationale);

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
     setEnhancementJourney(null);
     setShowBeforeAfter(false);

     try {
       const journeyResult = await generateEnhancementJourney({
         photoDataUri: originalImage,
         ...enhancementValues,
       });
       setUploadedImage(journeyResult.enhancedPhotoDataUri);
       setEnhancementJourney(journeyResult);
       setUploadedImageIsAiEnhanced(true);
       setShowBeforeAfter(false);
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
       setUploadedImageIsAiEnhanced(false);
     } finally {
       setIsProcessingEnhancement(false);
     }
   };

   const toggleBeforeAfter = () => {
      if (!originalImage || !uploadedImage) return;
     if (showBeforeAfter || !uploadedImageIsAiEnhanced) {
        setShowBeforeAfter(false);
     } else {
        setShowBeforeAfter(true);
     }
   }

   const getImageStyle = (): React.CSSProperties => {
     if (showBeforeAfter || uploadedImageIsAiEnhanced) return {};
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
           <TooltipContent side="top" align="start" className="max-w-xs text-xs p-2 z-50">
             <p>{rationale}</p>
           </TooltipContent>
         </Tooltip>
       </TooltipProvider>
     );
   };

  const ScoreBar = ({ label, score, icon: Icon, lowIsGood = false }: { label: string, score: number, icon?: React.ElementType, lowIsGood?: boolean }) => {
    const percentage = score * 10; // Assuming score is 0-10
    let colorClass = 'bg-green-500';
    if ((!lowIsGood && percentage < 40) || (lowIsGood && percentage >= 70)) colorClass = 'bg-red-500';
    else if ((!lowIsGood && percentage < 70) || (lowIsGood && percentage >= 40)) colorClass = 'bg-yellow-500';

    return (
      <div className="space-y-1">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium flex items-center">
            {Icon && <Icon className="w-4 h-4 mr-2 text-muted-foreground" />}
            {label}
          </span>
          <span className="font-semibold text-primary">{score}/10</span>
        </div>
        <Progress value={percentage} className={`h-2 ${colorClass}`} />
      </div>
    );
  };


  return (
    <div className="flex flex-col h-screen bg-secondary">
      <header className="bg-card border-b shadow-sm flex-shrink-0">
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
          </div>
        </div>
      </header>

      <main className="flex-grow container mx-auto px-4 py-8 flex flex-col lg:flex-row gap-8 overflow-hidden">
        <div
            ref={imageContainerRef}
            className="flex-grow lg:w-2/3 h-full flex flex-col items-center justify-center p-4 relative bg-card rounded-lg border shadow-sm overflow-hidden"
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
               {originalImage && uploadedImage && uploadedImageIsAiEnhanced && (
                 <Button
                   variant="outline"
                   size="sm"
                   className="absolute top-2 right-2 z-10 bg-card/80 hover:bg-card"
                   onClick={toggleBeforeAfter}
                   disabled={isProcessingEnhancement || isLoadingAI || isAssessingQuality}
                 >
                   {showBeforeAfter ? 'Show AI Enhanced' : 'Show Original'}
                 </Button>
               )}

               {(isLoadingAI || isProcessingEnhancement || isAssessingQuality) && (
                  <div className="absolute inset-0 bg-background/80 flex flex-col items-center justify-center z-20 rounded-lg">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
                    <p className="text-foreground text-lg font-medium">
                      {isAssessingQuality ? 'AI Assessing Quality...' : (isLoadingAI ? 'AI Analyzing & Animating...' : 'Applying AI Enhancements...')}
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
                <Button onClick={triggerFileInput} disabled={isAssessingQuality || isLoadingAI || isProcessingEnhancement}>
                  <Upload className="mr-2" /> Upload Image
                </Button>
                <Button variant="outline" onClick={handleCameraCapture} disabled={isAssessingQuality || isLoadingAI || isProcessingEnhancement}>
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

        <div className="lg:w-1/3 flex flex-col gap-4 overflow-y-auto pr-2">
          {/* Image Quality Assessment Panel */}
          <Card className="flex-shrink-0">
            <CardHeader><CardTitle>Image Quality Assessment</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              {isAssessingQuality && !imageQualityAssessment && (
                <>
                  <Skeleton className="h-8 w-3/4" />
                  <Skeleton className="h-2 w-full" />
                  <Skeleton className="h-8 w-3/4 mt-2" />
                  <Skeleton className="h-2 w-full" />
                  <Skeleton className="h-8 w-3/4 mt-2" />
                  <Skeleton className="h-2 w-full" />
                  <Skeleton className="h-8 w-3/4 mt-2" />
                  <Skeleton className="h-2 w-full" />
                  <Skeleton className="h-6 w-1/2 mt-2" />
                  <Skeleton className="h-4 w-full mt-1" />
                  <Skeleton className="h-4 w-full mt-1" />
                </>
              )}
              {!isAssessingQuality && !uploadedImage && (
                <p className="text-sm text-muted-foreground">Upload an image to see its quality assessment.</p>
              )}
              {imageQualityAssessment && (
                <>
                  <ScoreBar label="Front-Facing Pose" score={imageQualityAssessment.frontFacingScore} icon={Smile} />
                  <ScoreBar label="Eye Visibility" score={imageQualityAssessment.eyeVisibilityScore} icon={Eye} />
                  <ScoreBar label="Obstructions" score={imageQualityAssessment.obstructionScore} icon={UserX} lowIsGood={true} />
                   <div className="space-y-1">
                      <div className="flex items-center justify-between text-sm">
                           <span className="font-medium flex items-center">
                            <CheckCircle2 className="w-4 h-4 mr-2 text-muted-foreground" />
                            Overall Suitability
                           </span>
                           <span className="font-semibold text-primary">{imageQualityAssessment.overallSuitabilityScore}/10</span>
                         </div>
                         <Progress value={imageQualityAssessment.overallSuitabilityScore * 10} className="h-2 bg-primary/20 [&>div]:bg-primary" />
                    </div>

                  {imageQualityAssessment.feedback.length > 0 && (
                    <div>
                      <h4 className="text-sm font-semibold mb-1 mt-3">AI Feedback:</h4>
                      <ul className="list-disc pl-5 space-y-1 text-xs text-muted-foreground">
                        {imageQualityAssessment.feedback.map((item, index) => (
                          <li key={index}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>

          <Card className="flex-shrink-0">
            <CardHeader>
              <CardTitle>Enhancement Controls</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {(Object.keys(initialEnhancements) as Array<keyof EnhancementValues>).map((key) => (
                <div key={key} className="space-y-2">
                  <div className="flex justify-between items-center">
                     <Label htmlFor={key} className="capitalize font-medium flex items-center">
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
                    disabled={!uploadedImage || isLoadingAI || isProcessingEnhancement || isAssessingQuality}
                    aria-label={`${key} slider`}
                    className="[&>span:last-child]:transition-transform [&>span:last-child]:duration-100 [&>span:last-child]:ease-linear"
                  />
                </div>
              ))}
            </CardContent>
             <CardFooter className="flex flex-col gap-4 pt-4">
                <Button
                   onClick={handleSuggestEnhancements}
                   disabled={!originalImage || isLoadingAI || isProcessingEnhancement || isAssessingQuality}
                   className="w-full"
                 >
                   <WandSparkles className="mr-2" /> Suggest & Animate (AI)
                   {isLoadingAI && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-foreground ml-2"></div>}
                 </Button>
                 <Button
                   onClick={handleApplyEnhancements}
                   disabled={!originalImage || isLoadingAI || isProcessingEnhancement || isAssessingQuality}
                   className="w-full bg-accent hover:bg-accent/90 text-accent-foreground"
                 >
                   Apply AI & See Journey
                   {isProcessingEnhancement && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-accent-foreground ml-2"></div>}
                 </Button>
              </CardFooter>
          </Card>

           {enhancementJourney && !isProcessingEnhancement && (
              <Card className="flex-shrink-0">
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

      <footer className="bg-card border-t mt-auto flex-shrink-0">
        <div className="container mx-auto px-4 py-4 text-center text-muted-foreground text-sm">
          &copy; {new Date().getFullYear()} Headshot Handcrafter. All rights reserved.
        </div>
      </footer>
    </div>
  );
}

