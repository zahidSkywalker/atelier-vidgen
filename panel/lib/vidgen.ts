/**
 * Atelier VidGen — shared types + helpers.
 */

export type JobMode = "t2v" | "i2v";
export type JobStatus = "QUEUED" | "RENDERING" | "DONE" | "FAILED";
export type Aspect = "portrait" | "landscape";

export const ASPECT_DIMS: Record<Aspect, { w: number; h: number }> = {
  portrait: { w: 480, h: 864 },
  landscape: { w: 864, h: 480 },
};

export interface JobDTO {
  id: string;
  mode: JobMode;
  prompt: string;
  aspect: Aspect;
  seed: number | null;
  audio: boolean;
  superRes: boolean;
  refImage: string | null;
  status: JobStatus;
  progress: number;
  elapsedMs: number;
  videoPath: string | null;
  error: string | null;
  gpu: string | null;
  createdAt: string;
  finishedAt: string | null;
}

export const MAX_PROMPT = 1200;
