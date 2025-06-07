
"use client"

import * as React from "react"
import { ArrowLeft, ArrowRight } from "lucide-react"
import { cva, type VariantProps } from "class-variance-authority"
import useEmblaCarousel, {
  type EmblaCarouselType as EmblaApiType,
  type EmblaOptionsType,
  type EmblaPluginType,
} from "embla-carousel-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

const carouselVariants = cva("relative", {
  variants: {
    orientation: {
      horizontal: "flex-row",
      vertical: "flex-col",
    },
  },
  defaultVariants: {
    orientation: "horizontal",
  },
})

type CarouselContextProps = {
  carouselRef: React.RefObject<HTMLDivElement>
  api: CarouselApi | undefined
  orientation: "horizontal" | "vertical"
  scrollPrev: () => void
  scrollNext: () => void
  canScrollPrev: boolean
  canScrollNext: boolean
  handleKeyDown: (event: React.KeyboardEvent<HTMLDivElement>) => void
  opts: EmblaOptionsType
  plugins: EmblaPluginType[] | undefined
  direction?: EmblaOptionsType["direction"]
}

const CarouselContext = React.createContext<CarouselContextProps | null>(null)

function useCarousel() {
  const context = React.useContext(CarouselContext)

  if (!context) {
    throw new Error("useCarousel must be used within a <Carousel />")
  }

  return context
}

export type CarouselApi = EmblaApiType | undefined;

const Carousel = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> &
    VariantProps<typeof carouselVariants> &
    Partial<EmblaOptionsType> & { // Use EmblaOptionsType directly
      plugins?: EmblaPluginType[] // Use EmblaPluginType array
      orientation?: "horizontal" | "vertical"
      setApi?: (api: CarouselApi) => void
    }
>(
  (
    {
      orientation = "horizontal",
      opts,
      setApi,
      plugins,
      className,
      children,
      ...props
    },
    ref
  ) => {
    const carouselRef = React.useRef<HTMLDivElement>(null)
    // Directly use the hook, ensure it's imported correctly
    const [emblaRef, emblaApi] = useEmblaCarousel(
      {
        ...opts,
        axis: orientation === "horizontal" ? "x" : "y",
        ...(opts || {}), // Spread opts here
      },
      plugins
    )

    const [canScrollPrev, setCanScrollPrev] = React.useState(false)
    const [canScrollNext, setCanScrollNext] = React.useState(false)

    const onSelect = React.useCallback((api: EmblaApiType) => { // Use EmblaApiType
      if (!api) {
        return
      }
      setCanScrollPrev(api.canScrollPrev())
      setCanScrollNext(api.canScrollNext())
    }, [])

    const scrollPrev = React.useCallback(() => {
      emblaApi?.scrollPrev()
    }, [emblaApi])

    const scrollNext = React.useCallback(() => {
      emblaApi?.scrollNext()
    }, [emblaApi])

    const handleKeyDown = React.useCallback(
      (event: React.KeyboardEvent<HTMLDivElement>) => {
        if (event.key === "ArrowLeft") {
          event.preventDefault()
          scrollPrev()
        } else if (event.key === "ArrowRight") {
          event.preventDefault()
          scrollNext()
        }
      },
      [scrollPrev, scrollNext]
    )

    React.useEffect(() => {
      if (!emblaApi) return

      onSelect(emblaApi)
      emblaApi.on("reInit", onSelect)
      emblaApi.on("select", onSelect)

      if (setApi) {
        setApi(emblaApi)
      }

      return () => {
        emblaApi?.off("select", onSelect)
      }
    }, [emblaApi, onSelect, setApi])

    return (
      <CarouselContext.Provider
        value={{
          carouselRef,
          api: emblaApi,
          opts: opts || {},
          orientation:
            orientation || (opts?.axis === "y" ? "vertical" : "horizontal"),
          scrollPrev,
          scrollNext,
          canScrollPrev,
          canScrollNext,
          handleKeyDown,
          direction: opts?.direction,
          plugins,
        }}
      >
        <div
          ref={ref}
          onKeyDownCapture={handleKeyDown}
          className={cn(
            "relative focus:outline-none",
            carouselVariants({ orientation }),
            className
          )}
          role="region"
          aria-roledescription="carousel"
          {...props}
        >
          <div ref={emblaRef} className="overflow-hidden">
            <div
              ref={carouselRef} // This ref is for the inner container, not emblaRef
              className={cn(
                "flex",
                orientation === "horizontal" ? "-ml-4" : "-mt-4 flex-col"
              )}
            >
              {children}
            </div>
          </div>
        </div>
      </CarouselContext.Provider>
    )
  }
)
Carousel.displayName = "Carousel"

const CarouselContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  const { carouselRef, orientation } = useCarousel() // carouselRef from context is for the track

  // This div is the one embla-carousel-react actually needs for its main container
  // The children will be the CarouselItems
  return (
    <div ref={carouselRef} className="overflow-hidden"> 
      <div
        ref={ref} // This ref is for the direct child div containing the slides
        className={cn(
          "flex",
          orientation === "horizontal" ? "" : "flex-col",
          className
        )}
        {...props}
      />
    </div>
  )
})
CarouselContent.displayName = "CarouselContent"

const CarouselItem = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  const { orientation } = useCarousel()

  return (
    <div
      ref={ref}
      role="group"
      aria-roledescription="slide"
      className={cn(
        "min-w-0 shrink-0 grow-0 basis-full",
        orientation === "horizontal" ? "pl-4" : "pt-4",
        className
      )}
      {...props}
    />
  )
})
CarouselItem.displayName = "CarouselItem"

const CarouselPrevious = React.forwardRef<
  HTMLButtonElement,
  React.ComponentProps<typeof Button>
>(({ className, variant = "outline", size = "icon", ...props }, ref) => {
  const {
    orientation,
    scrollPrev,
    canScrollPrev,
    handleKeyDown, // Added to enable keyboard navigation
    direction,
  } = useCarousel()
  const isRTL = direction === "rtl"

  return (
    <Button
      ref={ref}
      variant={variant}
      size={size}
      className={cn(
        "absolute h-8 w-8 rounded-full",
        orientation === "horizontal"
          ? "-left-12 top-1/2 -translate-y-1/2"
          : "-top-12 left-1/2 -translate-x-1/2 rotate-90",
        className
      )}
      disabled={!canScrollPrev}
      onClick={(e) => { // Ensure default behavior is prevented if needed
        e.preventDefault() 
        scrollPrev()
      }}
      onKeyDown={handleKeyDown} // Added for keyboard control
      {...props}
    >
      {isRTL ? (
        <ArrowRight className="h-4 w-4" />
      ) : (
        <ArrowLeft className="h-4 w-4" />
      )}
      <span className="sr-only">Previous slide</span>
    </Button>
  )
})
CarouselPrevious.displayName = "CarouselPrevious"

const CarouselNext = React.forwardRef<
  HTMLButtonElement,
  React.ComponentProps<typeof Button>
>(({ className, variant = "outline", size = "icon", ...props }, ref) => {
  const { orientation, scrollNext, canScrollNext, handleKeyDown, direction } = // Added handleKeyDown
    useCarousel()
  const isRTL = direction === "rtl"

  return (
    <Button
      ref={ref}
      variant={variant}
      size={size}
      className={cn(
        "absolute h-8 w-8 rounded-full",
        orientation === "horizontal"
          ? "-right-12 top-1/2 -translate-y-1/2"
          : "-bottom-12 left-1/2 -translate-x-1/2 rotate-90",
        className
      )}
      disabled={!canScrollNext}
      onClick={(e) => { // Ensure default behavior is prevented if needed
         e.preventDefault()
         scrollNext()
      }}
      onKeyDown={handleKeyDown} // Added for keyboard control
      {...props}
    >
      {isRTL ? (
        <ArrowLeft className="h-4 w-4" />
      ) : (
        <ArrowRight className="h-4 w-4" />
      )}
      <span className="sr-only">Next slide</span>
    </Button>
  )
})
CarouselNext.displayName = "CarouselNext"

export {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselPrevious,
  CarouselNext,
}
