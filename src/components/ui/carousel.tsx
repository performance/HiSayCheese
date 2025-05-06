"use client"

import * as React from "react"
import { ArrowLeft, ArrowRight } from "lucide-react"
import { cva, type VariantProps } from "class-variance-authority"

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
  opts: EmblaCarouselType["options"]
  plugins: EmblaCarouselType["plugins"]
  direction?: EmblaCarouselType["options"]["direction"]
}

const CarouselContext = React.createContext<CarouselContextProps | null>(null)

function useCarousel() {
  const context = React.useContext(CarouselContext)

  if (!context) {
    throw new Error("useCarousel must be used within a <Carousel />")
  }

  return context
}

// We need to lazy load the embla-carousel-react library
// as it uses parts of the browser API (window)
// and that makes it incompatible with server components.
const useEmblaCarousel = React.lazy(() =>
  import("embla-carousel-react").then((mod) => ({
    default: mod.useEmblaCarousel,
  }))
)

type EmblaCarouselType = NonNullable<
  ReturnType<typeof useEmblaCarousel>[1]
>["embla"]

const Carousel = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> &
    VariantProps<typeof carouselVariants> &
    Partial<CarouselContextProps["opts"]> & {
      plugins?: CarouselContextProps["plugins"]
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
    const [emblaRef, emblaApi] = useEmblaCarousel(
      {
        ...opts,
        axis: orientation === "horizontal" ? "x" : "y",
      },
      plugins
    )

    const [canScrollPrev, setCanScrollPrev] = React.useState(false)
    const [canScrollNext, setCanScrollNext] = React.useState(false)

    const onSelect = React.useCallback((api: CarouselApi) => {
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

      return () => {
        emblaApi?.off("select", onSelect)
      }
    }, [emblaApi, onSelect])

    return (
      <CarouselContext.Provider
        value={{
          carouselRef,
          api: emblaApi,
          opts,
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
        <React.Suspense fallback={null}>
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
                ref={carouselRef}
                className={cn(
                  "flex",
                  orientation === "horizontal" ? "-ml-4" : "-mt-4 flex-col"
                )}
              >
                {children}
              </div>
            </div>
          </div>
        </React.Suspense>
      </CarouselContext.Provider>
    )
  }
)
Carousel.displayName = "Carousel"

const CarouselContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  const { carouselRef, orientation } = useCarousel()

  return (
    <div ref={carouselRef} className="overflow-hidden">
      <div
        ref={ref}
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
    handleKeyDown,
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
      onClick={(e) => {
        e.preventDefault()
        scrollPrev()
      }}
      onKeyDown={handleKeyDown}
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
  const { orientation, scrollNext, canScrollNext, handleKeyDown, direction } =
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
      onClick={(e) => {
        e.preventDefault()
        scrollNext()
      }}
      onKeyDown={handleKeyDown}
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
  type CarouselApi,
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselPrevious,
  CarouselNext,
}

// Define CarouselApi type based on Embla's API
// Ensure this is compatible with the lazy-loaded module
export type CarouselApi = EmblaCarouselType | undefined;
