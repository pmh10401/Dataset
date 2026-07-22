import { timingSafeEqual } from "node:crypto";

export function hasValidBearerToken(header: string | null, expected: string | undefined) {
  if (!header?.startsWith("Bearer ") || !expected) return false;
  const actualBuffer = Buffer.from(header.slice(7));
  const expectedBuffer = Buffer.from(expected);
  return actualBuffer.length === expectedBuffer.length && timingSafeEqual(actualBuffer, expectedBuffer);
}
