"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Clapperboard,
  Download,
  Film,
  ImageIcon,
  Loader2,
  Sparkles,
  Type,
  Upload,
  Zap,
  AlertTriangle,
  CheckCircle2,
  Clock3,
  RefreshCw,
} from "lucide-react";
import type { JobDTO } from "@/lib/vidgen";

const SAMPLE_PROMPTS = [
  "A lone lighthouse on a cliff at dusk, waves crashing below, warm lantern beam sweeping across the water, cinematic anamorphic feel",
  "Neon rain over an empty night market, steam rising from food stalls, reflections shimmering on wet asphalt, slow dolly forward",
  "Golden-hour drone shot gliding over misty tea terraces, soft haze, birds crossing the frame, serene and vast",
];

function statusPill(status: JobDTO["status"]) {
  switch (status) {
    case "QUEUED":
      return (
        <Badge variant="outline" className="border-amber-500/50 text-amber-400 gap-1">
          <Clock3 className="h-3 w-3" /> Queued
        </Badge>
      );
    case "RENDERING":
      return (
        <Badge variant="outline" className="border-yellow-500/50 text-yellow-300 gap-1">
          <Loader2 className="h-3 w-3 animate-spin" /> Rendering
        </Badge>
      );
    case "DONE":
      return (
        <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 gap-1">
          <CheckCircle2 className="h-3 w-3" /> Done
        </Badge>
      );
    case "FAILED":
      return (
        <Badge variant="outline" className="border-red-500/50 text-red-400 gap-1">
          <AlertTriangle className="h-3 w-3" /> Failed
        </Badge>
      );
  }
}

function elapsed(ms: number) {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

function timeAgo(iso: string) {
  const diff = Date.now() - new Date(iso).getTime();
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return new Date(iso).toLocaleDateString();
}

export default function VidGenPanel() {
  const [mode, setMode] = useState<"t2v" | "i2v">("t2v");
  const [prompt, setPrompt] = useState("");
  const [aspect, setAspect] = useState<"portrait" | "landscape">("portrait");
  const [seed, setSeed] = useState("");
  const [audio, setAudio] = useState(true);
  const [superRes, setSuperRes] = useState(false);
  const [refPreview, setRefPreview] = useState<string | null>(null);
  const [refName, setRefName] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [jobs, setJobs] = useState<JobDTO[]>([]);
  const [loadingJobs, setLoadingJobs] = useState(true);
  const [ticking, setTicking] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/jobs", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setJobs(data.jobs ?? []);
    } catch {
      // silent — panel keeps last known state
    } finally {
      setLoadingJobs(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, [refresh]);

  // re-render elapsed timers every second while jobs are active
  useEffect(() => {
    const hasActive = jobs.some((j) => j.status === "RENDERING" || j.status === "QUEUED");
    if (!hasActive) return;
    const t = setInterval(() => setTicking((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [jobs]);

  const onPickFile = (f: File | null) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      toast.error("Please pick an image file (jpg/png).");
      return;
    }
    setRefPreview(URL.createObjectURL(f));
    setRefName(f.name);
  };

  const submit = async () => {
    if (submitting) return;
    if (!prompt.trim()) {
      toast.error("Give me a prompt first, sir.");
      return;
    }
    if (mode === "i2v" && !refPreview) {
      toast.error("Image-to-video needs a reference image.");
      return;
    }
    setSubmitting(true);
    try {
      const fd = new FormData();
      fd.set("mode", mode);
      fd.set("prompt", prompt.trim());
      fd.set("aspect", aspect);
      fd.set("seed", seed);
      fd.set("audio", String(audio));
      fd.set("superRes", String(superRes));
      if (mode === "i2v" && fileRef.current?.files?.[0]) {
        fd.set("ref", fileRef.current.files[0]);
      }
      const res = await fetch("/api/jobs", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      toast.success("Job queued — the lane will pick it up.", {
        description: `id ${data.job.id.slice(0, 10)}…`,
      });
      setPrompt("");
      setSeed("");
      setRefPreview(null);
      setRefName(null);
      if (fileRef.current) fileRef.current.value = "";
      refresh();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "submit failed");
    } finally {
      setSubmitting(false);
    }
  };

  const active = jobs.filter((j) => j.status === "QUEUED" || j.status === "RENDERING");
  const done = jobs.filter((j) => j.status === "DONE");
  const failed = jobs.filter((j) => j.status === "FAILED");

  return (
    <div className="min-h-screen flex flex-col bg-zinc-950">
      {/* ambient glow */}
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 opacity-40"
        style={{
          background:
            "radial-gradient(60% 40% at 50% 0%, rgba(245,158,11,0.10), transparent 70%), radial-gradient(40% 30% at 80% 100%, rgba(16,185,129,0.05), transparent 70%)",
        }}
      />

      <header className="relative z-10 border-b border-zinc-800/80 bg-zinc-950/70 backdrop-blur sticky top-0">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-amber-400 to-amber-600 grid place-items-center shrink-0 shadow-lg shadow-amber-900/30">
              <Clapperboard className="h-5 w-5 text-zinc-950" />
            </div>
            <div className="min-w-0">
              <h1 className="font-semibold tracking-tight leading-none truncate">
                Atelier <span className="text-amber-400">VidGen</span>
              </h1>
              <p className="text-[11px] text-zinc-500 leading-none mt-1 truncate">
                Kandinsky 6.0 Lite · synced audio · 5 s clips
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Badge variant="outline" className="border-zinc-700 text-zinc-400 hidden sm:inline-flex gap-1">
              <Zap className="h-3 w-3 text-amber-400" /> Kaggle B lane
            </Badge>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Refresh jobs"
              className="text-zinc-400 hover:text-amber-400"
              onClick={() => {
                refresh();
                toast.info("Queue refreshed.");
              }}
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </header>

      <main className="relative z-10 flex-1 w-full max-w-6xl mx-auto px-4 sm:px-6 py-6 sm:py-8">
        <div className="grid gap-6 lg:grid-cols-[420px_1fr] items-start">
          {/* ── Composer ─────────────────────────────────────── */}
          <Card className="border-zinc-800 bg-zinc-900/60 backdrop-blur">
            <CardHeader className="pb-4">
              <CardTitle className="flex items-center gap-2 text-lg">
                <Sparkles className="h-4 w-4 text-amber-400" /> Compose a shot
              </CardTitle>
              <CardDescription>
                One prompt, one 5-second cinematic clip. 10-step distilled render.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <Tabs
                value={mode}
                onValueChange={(v) => setMode(v as "t2v" | "i2v")}
              >
                <TabsList className="grid w-full grid-cols-2 bg-zinc-800/80">
                  <TabsTrigger value="t2v" className="gap-1.5 data-[state=active]:text-amber-400">
                    <Type className="h-4 w-4" /> Text → Video
                  </TabsTrigger>
                  <TabsTrigger value="i2v" className="gap-1.5 data-[state=active]:text-amber-400">
                    <ImageIcon className="h-4 w-4" /> Image → Video
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="t2v" className="mt-4 space-y-4">
                  <p className="text-xs text-zinc-500">
                    Describe the shot: subject, motion, light, camera. The model does 5 s @ 24 fps.
                  </p>
                </TabsContent>

                <TabsContent value="i2v" className="mt-4 space-y-4">
                  <div
                    role="button"
                    tabIndex={0}
                    aria-label="Upload reference image"
                    onClick={() => fileRef.current?.click()}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") fileRef.current?.click();
                    }}
                    className="border border-dashed border-zinc-700 rounded-lg p-4 text-center cursor-pointer hover:border-amber-500/60 hover:bg-amber-500/5 transition-colors"
                  >
                    {refPreview ? (
                      <div className="flex flex-col items-center gap-2">
                        <img
                          src={refPreview}
                          alt="Reference frame preview"
                          className="max-h-36 rounded-md border border-zinc-800"
                        />
                        <span className="text-xs text-zinc-400 truncate max-w-full">{refName}</span>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-2 text-zinc-500 py-2">
                        <Upload className="h-6 w-6" />
                        <span className="text-sm">Drop a reference frame or click to browse</span>
                        <span className="text-[11px]">jpg / png · max 12 MB · this frame opens the clip</span>
                      </div>
                    )}
                  </div>
                  <input
                    ref={fileRef}
                    type="file"
                    accept="image/jpeg,image/png"
                    className="sr-only"
                    aria-label="Reference image input"
                    onChange={(e) => onPickFile(e.target.files?.[0] ?? null)}
                  />
                  <p className="text-xs text-zinc-500">
                    The clip animates out of your image — best for keeping a character or scene consistent.
                  </p>
                </TabsContent>
              </Tabs>

              <div className="space-y-2">
                <Label htmlFor="prompt">Prompt</Label>
                <Textarea
                  id="prompt"
                  placeholder="A lone lighthouse on a cliff at dusk, waves crashing, lantern beam sweeping…"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={4}
                  maxLength={1200}
                  className="bg-zinc-950/70 border-zinc-800 focus-visible:ring-amber-500/50 resize-none"
                />
                <div className="flex justify-between text-[11px] text-zinc-600">
                  <button
                    type="button"
                    className="underline-offset-2 hover:underline hover:text-amber-400"
                    onClick={() => setPrompt(SAMPLE_PROMPTS[Math.floor(Math.random() * SAMPLE_PROMPTS.length)])}
                  >
                    Need inspiration? Drop a sample shot →
                  </button>
                  <span>{prompt.length}/1200</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="aspect">Aspect</Label>
                  <div className="grid grid-cols-2 gap-1 rounded-lg bg-zinc-800/80 p-1">
                    {(["portrait", "landscape"] as const).map((a) => (
                      <button
                        key={a}
                        type="button"
                        onClick={() => setAspect(a)}
                        className={`rounded-md py-1.5 text-xs capitalize transition-colors ${
                          aspect === a
                            ? "bg-amber-500 text-zinc-950 font-semibold"
                            : "text-zinc-400 hover:text-zinc-200"
                        }`}
                      >
                        {a === "portrait" ? "9:16 · 480×864" : "16:9 · 864×480"}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="seed">Seed (optional)</Label>
                  <Input
                    id="seed"
                    inputMode="numeric"
                    placeholder="random"
                    value={seed}
                    onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ""))}
                    className="bg-zinc-950/70 border-zinc-800 focus-visible:ring-amber-500/50"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between rounded-lg border border-zinc-800 bg-zinc-950/50 px-3 py-2.5">
                <div>
                  <Label htmlFor="audio" className="text-sm">
                    Synchronized audio
                  </Label>
                  <p className="text-[11px] text-zinc-500">Kandinsky generates the sound bed in the same pass</p>
                </div>
                <Switch id="audio" checked={audio} onCheckedChange={setAudio} />
              </div>

              <div className="flex items-center justify-between rounded-lg border border-zinc-800 bg-zinc-950/50 px-3 py-2.5 opacity-60">
                <div>
                  <Label htmlFor="sr" className="text-sm">
                    Full-HD super-res
                  </Label>
                  <p className="text-[11px] text-zinc-500">2.25× upscale · adds ~15 min — coming in M4</p>
                </div>
                <Switch id="sr" checked={superRes} disabled onCheckedChange={setSuperRes} />
              </div>

              <Button
                onClick={submit}
                disabled={submitting}
                className="w-full bg-amber-500 hover:bg-amber-400 text-zinc-950 font-semibold h-11"
              >
                {submitting ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" /> Queuing…
                  </>
                ) : (
                  <>
                    <Film className="h-4 w-4" /> Render clip
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* ── Queue + Gallery ─────────────────────────────── */}
          <div className="space-y-6 min-w-0">
            <Card className="border-zinc-800 bg-zinc-900/60 backdrop-blur">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Clock3 className="h-4 w-4 text-amber-400" /> Render queue
                  {active.length > 0 && (
                    <Badge className="bg-amber-500/15 text-amber-400 border-amber-500/30 ml-1">
                      {active.length}
                    </Badge>
                  )}
                </CardTitle>
                <CardDescription>
                  Kaggle B worker claims jobs from the exchange — expect minutes, not seconds.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 max-h-72 overflow-y-auto vidgen-scroll">
                {loadingJobs ? (
                  <div className="flex items-center gap-2 text-sm text-zinc-500 py-6 justify-center">
                    <Loader2 className="h-4 w-4 animate-spin" /> Loading queue…
                  </div>
                ) : active.length === 0 ? (
                  <p className="text-sm text-zinc-600 py-6 text-center">
                    Queue is clear. The lane sleeps until you feed it.
                  </p>
                ) : (
                  active.map((j) => (
                    <div
                      key={j.id}
                      data-tick={ticking}
                      className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3 space-y-2"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          {statusPill(j.status)}
                          <Badge variant="outline" className="border-zinc-700 text-zinc-500 uppercase text-[10px]">
                            {j.mode}
                          </Badge>
                          <span className="text-[11px] text-zinc-600">{timeAgo(j.createdAt)}</span>
                        </div>
                        <span className="text-xs text-zinc-500 tabular-nums shrink-0">
                          {j.status === "RENDERING" ? `on lane · ${elapsed(j.elapsedMs)}` : "waiting"}
                        </span>
                      </div>
                      <p className="text-sm text-zinc-300 line-clamp-2">{j.prompt}</p>
                      <Progress value={j.status === "RENDERING" ? Math.min(90, 10 + j.progress) : 4} className="h-1.5" />
                    </div>
                  ))
                )}
              </CardContent>
            </Card>

            <Card className="border-zinc-800 bg-zinc-900/60 backdrop-blur">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Film className="h-4 w-4 text-amber-400" /> Gallery
                  {done.length > 0 && (
                    <Badge variant="outline" className="border-zinc-700 text-zinc-400 ml-1">
                      {done.length}
                    </Badge>
                  )}
                </CardTitle>
                <CardDescription>Every finished clip stays here. Tap to play, download to keep.</CardDescription>
              </CardHeader>
              <CardContent>
                {done.length === 0 && failed.length === 0 ? (
                  <p className="text-sm text-zinc-600 py-10 text-center">
                    No clips yet — your first render lands here.
                  </p>
                ) : (
                  <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                    {done.map((j) => (
                      <div
                        key={j.id}
                        className={`group rounded-lg border border-zinc-800 bg-zinc-950/60 overflow-hidden flex flex-col ${
                          j.aspect === "portrait" ? "max-w-[240px]" : ""
                        }`}
                      >
                        {j.videoPath ? (
                          <video
                            src={j.videoPath}
                            controls
                            playsInline
                            preload="metadata"
                            className="w-full bg-black"
                          />
                        ) : (
                          <div className="aspect-video grid place-items-center text-zinc-700">
                            <Film className="h-8 w-8" />
                          </div>
                        )}
                        <div className="p-3 space-y-2 flex-1 flex flex-col">
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <Badge variant="outline" className="border-zinc-700 text-zinc-500 uppercase text-[10px]">
                              {j.mode}
                            </Badge>
                            {j.audio && (
                              <Badge variant="outline" className="border-emerald-500/40 text-emerald-400/90 text-[10px]">
                                audio
                              </Badge>
                            )}
                            <span className="text-[11px] text-zinc-600 ml-auto">{timeAgo(j.createdAt)}</span>
                          </div>
                          <p className="text-xs text-zinc-400 line-clamp-2 flex-1" title={j.prompt}>
                            {j.prompt}
                          </p>
                          <div className="flex items-center justify-between">
                            <span className="text-[10px] text-zinc-600">
                              {j.seed != null ? `seed ${j.seed}` : "random seed"}
                            </span>
                            {j.videoPath && (
                              <a
                                href={j.videoPath}
                                download={`vidgen_${j.id}.mp4`}
                                className="inline-flex items-center gap-1 text-xs text-amber-400 hover:text-amber-300"
                                aria-label={`Download clip ${j.id}`}
                              >
                                <Download className="h-3.5 w-3.5" /> mp4
                              </a>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                    {failed.map((j) => (
                      <div key={j.id} className="rounded-lg border border-red-900/50 bg-red-950/20 p-3 space-y-1">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="border-red-500/50 text-red-400 text-[10px]">
                            FAILED
                          </Badge>
                          <span className="text-[11px] text-zinc-600">{timeAgo(j.createdAt)}</span>
                        </div>
                        <p className="text-xs text-zinc-400 line-clamp-2">{j.prompt}</p>
                        <p className="text-[11px] text-red-400/80">{j.error || "render failed"}</p>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      <footer className="relative z-10 mt-auto border-t border-zinc-800/80 bg-zinc-950/80 backdrop-blur pb-[env(safe-area-inset-bottom)]">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-zinc-600">
          <span>
            Atelier VidGen · single lane: <span className="text-zinc-400">Kaggle B (xion678)</span> · 30 GPU-h/week
          </span>
          <span className="flex items-center gap-1.5">
            <Separator orientation="vertical" className="h-3 hidden sm:block" />
            Kandinsky 6.0 Video Lite — MIT license
          </span>
        </div>
      </footer>
    </div>
  );
}
