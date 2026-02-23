import type { ImageFile } from '../global'
import { Thumbnail } from './Thumbnail'

interface GalleryProps {
  images: ImageFile[]
  selectedPath: string | null
  onSelect: (image: ImageFile) => void
}

export function Gallery({ images, selectedPath, onSelect }: GalleryProps) {
  if (images.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-neutral-600 text-sm">
        No images found in this folder
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-4">
      <div className="grid gap-2" style={{
        gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))'
      }}>
        {images.map((img) => (
          <Thumbnail
            key={img.path}
            image={img}
            selected={img.path === selectedPath}
            onClick={() => onSelect(img)}
          />
        ))}
      </div>
    </div>
  )
}
