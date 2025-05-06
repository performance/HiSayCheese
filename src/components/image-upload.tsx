import React, { ChangeEvent, DragEvent, RefObject, ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import { Upload, Camera } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import Image from 'next/image';
import { TestImage } from '@/lib/test-utils';

interface ImageUploadProps extends React.HTMLAttributes<HTMLDivElement> {
  handleImageUpload: (event: ChangeEvent<HTMLInputElement>) => void;
  handleDrop: (event: DragEvent<HTMLDivElement>) => void;
  handleDragOver: (event: DragEvent<HTMLDivElement>) => void;
  handleDragLeave: (event: DragEvent<HTMLDivElement>) => void;
  triggerFileInput: () => void;
  fileInputRef: RefObject<HTMLInputElement>;
  isAssessingQuality: boolean;
  isLoadingAI: boolean;
  isProcessingEnhancement: boolean;
  testImages: TestImage[];
  selectedTestImage: TestImage | null;
  handleTestImageSelect: (testImage: TestImage) => void;
}

const ImageUpload: React.FC<ImageUploadProps> = ({
  handleImageUpload,
  handleDrop,
  handleDragOver,
  handleDragLeave,
  triggerFileInput,
  fileInputRef,
  isAssessingQuality,
  isLoadingAI,
  isProcessingEnhancement,
  testImages,
  selectedTestImage,
  handleTestImageSelect
}) => {
  const isDisabled = isAssessingQuality || isLoadingAI || isProcessingEnhancement;

  return (
    <div
      className="overflow-y-auto lg:col-span-2 flex flex-col items-center justify-center p-4 relative bg-card rounded-lg border shadow-sm overflow-hidden min-h-[calc(100vh-16rem)] h-full"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <div className="text-center text-muted-foreground flex flex-col items-center justify-center h-full border-2 border-dashed border-border rounded-lg p-8">
        <Upload className="w-12 h-12 mb-4 text-primary" />
        <p className="mb-2 font-medium">Drag & drop your image here</p>
        <p className="mb-4 text-sm">or</p>
        <div className="flex flex-col sm:flex-row gap-4">
          <Button onClick={triggerFileInput} disabled={isDisabled}>
            <Upload className="mr-2" /> Upload Image
          </Button>
          <Button variant="outline" onClick={() => {}} disabled={isDisabled}>
            <Camera className="mr-2" /> Use Camera {/* Camera capture not yet implemented */}
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
      {testImages.length > 0 && (
          <Card className="mt-8 flex-shrink-0">
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
      </div>
    </div>
  );
};

export default ImageUpload;