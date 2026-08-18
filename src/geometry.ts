export interface DisplayPoint {
  x: number;
  y: number;
}

/** Map a normalized media point through CSS object-fit: cover. */
export function mapCoverPoint(
  normalizedX: number,
  normalizedY: number,
  containerWidth: number,
  containerHeight: number,
  mediaWidth: number,
  mediaHeight: number,
  mirrorX = false,
): DisplayPoint {
  if (containerWidth <= 0 || containerHeight <= 0 || mediaWidth <= 0 || mediaHeight <= 0) {
    return { x: 0, y: 0 };
  }
  const scale = Math.max(containerWidth / mediaWidth, containerHeight / mediaHeight);
  const renderedWidth = mediaWidth * scale;
  const renderedHeight = mediaHeight * scale;
  const offsetX = (containerWidth - renderedWidth) / 2;
  const offsetY = (containerHeight - renderedHeight) / 2;
  const x = mirrorX ? 1 - normalizedX : normalizedX;
  return {
    x: offsetX + x * renderedWidth,
    y: offsetY + normalizedY * renderedHeight,
  };
}

export interface Point2D {
  x: number;
  y: number;
}

/** Returns the area enclosed by a 2D polygon using the shoelace formula. */
export function polygonArea(points: readonly Point2D[]): number {
  if (points.length < 3) return 0;
  let twiceArea = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    twiceArea += current.x * next.y - next.x * current.y;
  }
  return Math.abs(twiceArea) / 2;
}
