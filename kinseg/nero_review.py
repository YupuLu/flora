"""Review videos and page: our subtasks next to FLORA's, on the main camera of each episode.

Each episode becomes an MP4 in the layout of FLORA's review videos (main camera, right panel, caption
footer), with the active subtask of both methods in the footer, the current kinematic phase (arm and base
motion, gripper events) in the right panel, and three time lines below: our phases, our subtasks and
FLORA's segments, with a cursor. ``index.html`` lists every episode with its video and both segment
lists, and our per-phase captions.

    python kinseg/nero_review.py --jobs jobs.jsonl --kinematic <run> --captions <caption run> \\
        --segments <combine segments> --flora <FLORA run> --out <dir> [--method C5]

Needs ffmpeg with libass. Times are seconds from the start of the video.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

W, H = 1280, 1200
FOOT_Y, FOOT_H = 761, 300
LINE_Y = 1075                              # first time line
BAR_X, BAR_W, BAR_H, BAR_GAP = 150, 1100, 22, 36
TEXT = "&H00F4F2ED&"
DIM = "&H00B8A99A&"
FILL = ("&H00C08A3E&", "&H006F5A2E&")      # alternating segment colours (BGR)
CURSOR = "&H003050FF&"
TIMELINES = (("our phases", "phases"), ("ours", "ours"), ("FLORA", "flora"))
METHODS = {"C1": "the model's subtasks, C1", "C5": "cuts anchored at confirmed grasps, C5"}


def at(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    return f"{cs // 360000}:{cs // 6000 % 60:02}:{cs // 100 % 60:02}.{cs % 100:02}"


def escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def wrap(text: str, size: int, width: int) -> List[str]:
    """Greedy word wrap for DejaVu Sans at ``size`` px in ``width`` px (about 0.55 em per character)."""
    limit = max(10, int(width / (size * 0.55)))
    lines: List[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            if line and len(line) + 1 + len(word) > limit:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}" if line else word
        lines.append(line)
    return lines


class Subtitles:
    def __init__(self, duration: float):
        self.duration = duration
        self.events: List[str] = []

    def text(self, start, end, text, x, y, size=22, colour=TEXT):
        tags = "{" + rf"\an7\pos({x},{y})\fs{size}\c{colour}" + "}"
        self.events.append(f"Dialogue: 0,{at(start)},{at(end)},Default,,0,0,0,,{tags}{escape(text)}")

    def static(self, text, x, y, size=22, width=380, colour=TEXT):
        self.text(0, self.duration, "\n".join(wrap(text, size, width)), x, y, size, colour)

    def box(self, start, end, x, y, w, h, colour, layer=0):
        tags = "{" + rf"\an7\pos({x},{y})\c{colour}\bord0\shad0\p1" + "}"
        self.events.append(f"Dialogue: {layer},{at(start)},{at(end)},Default,,0,0,0,,{tags}"
                           f"m 0 0 l {w} 0 l {w} {h} l 0 {h}{{\\p0}}")

    def ass(self) -> str:
        return ("[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\nWrapStyle: 2\n"
                "ScaledBorderAndShadow: yes\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, "
                "SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
                "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
                "Style: Default,DejaVu Sans,22,&H00F4F2ED,&H00FFFFFF,&H001C140F,&H00000000,0,0,0,0,100,100,0,0,"
                "1,0,0,7,0,0,0,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
                "Effect, Text\n" % (W, H)) + "\n".join(self.events) + "\n"


def active(items: List[Dict[str, object]], t: float) -> List[Tuple[int, Dict[str, object]]]:
    return [(i, s) for i, s in enumerate(items) if s["start_s"] <= t < s["end_s"]]


def phase_status(phase: Dict[str, object]) -> str:
    arm = {"STABLE": "arms still", "BOTH": "both arms move"}.get(phase["arm"], f"{phase['arm'].lower()} arm moves")
    base = "base moves" if phase.get("base", "STABLE") != "STABLE" else "base still"
    events = [f"{e['arm'].lower()} gripper {'closes' if e['kind'] == 'close' else 'opens'} at {e['start_sec']:.1f} s"
              for e in phase["gripper_events"]]
    return "\n".join([f"{arm}, {base}"] + events)


def subtitles(job, duration, phases, ours, flora) -> str:
    sub = Subtitles(duration)
    sub.static(f"Ours vs FLORA / {job['group']}", 24, 18, 30, 1230)
    sub.static(job["id"], 24, 59, 20, 1230)
    sub.static("DATASET TASK LABEL", 865, 110, 18, colour=DIM)
    sub.static(job["task"], 865, 136, 23)
    sub.static("OURS: kinematic phases, Captioner (three cameras, our template), combined by a language model; "
               "FLORA: TempSeg + Captioner (task brief)", 865, 230, 17, colour=DIM)
    sub.static(f"Source {duration:.2f} s; phases {len(phases)}; our subtasks {len(ours)}; FLORA segments {len(flora)}",
               865, 345, 18)
    sub.static("CURRENT PHASE (measured)", 865, 430, 18, colour=DIM)
    for i, p in enumerate(phases):
        sub.text(p["start_sec"], p["end_sec"], f"#{i + 1}  {p['start_sec']:.1f}-{p['end_sec']:.1f} s\n"
                 + phase_status(p), 865, 458, 20)
    for t in range(math.ceil(duration)):
        sub.text(t, min(t + 1, duration), f"{t // 60:02}:{t % 60:02} / {int(duration) // 60:02}:{int(duration) % 60:02}",
                 865, 700, 24)
    # footer: the active subtask of each method
    bounds = sorted({0.0, duration, *[min(duration, max(0.0, s[k])) for s in ours + flora for k in ("start_s", "end_s")]})
    for start, end in zip(bounds, bounds[1:]):
        if end - start < 0.005:
            continue
        mid = (start + end) / 2
        for label, items, key, y in (("OURS", ours, "instruction", 772), ("FLORA", flora, "caption", 912)):
            lines = [f"#{i + 1}  {s['start_s']:.1f}-{s['end_s']:.1f} s | {s.get(key) or s.get('status', '')}"
                     for i, s in active(items, mid)] or ["(no segment)"]
            for size in (21, 20, 19, 18, 17, 16, 15, 14):   # fit the row's 135 px
                wrapped = wrap("\n".join(lines), size, 1120)
                if len(wrapped) * size * 1.3 < 135:
                    break
            sub.text(start, end, "\n".join(wrapped), 120, y, size)
    sub.static("OURS", 24, 772, 21, colour=DIM)
    sub.static("FLORA", 24, 912, 21, colour=DIM)
    # time lines
    for row, (name, key) in enumerate(TIMELINES):
        y = LINE_Y + row * BAR_GAP
        sub.static(name, 24, y - 2, 18, colour=DIM)
        items = {"phases": [dict(start_s=p["start_sec"], end_s=p["end_sec"]) for p in phases],
                 "ours": ours, "flora": flora}[key]
        for i, s in enumerate(items):
            x0 = BAR_X + BAR_W * s["start_s"] / duration
            x1 = BAR_X + BAR_W * s["end_s"] / duration
            sub.box(0, duration, round(x0), y, max(1, round(x1 - x0) - 1), BAR_H, FILL[i % 2])
    step = 0.2
    for k in range(math.ceil(duration / step)):
        t = k * step
        x = BAR_X + BAR_W * min(t, duration) / duration
        sub.box(t, min(t + step, duration), round(x), LINE_Y - 4, 3, 2 * BAR_GAP + BAR_H + 8, CURSOR, layer=1)
    return sub.ass()


def passthrough() -> List[str]:
    """ffmpeg's flag that keeps every source frame and its time: -fps_mode (5.1 and later), else -vsync."""
    help_text = subprocess.run(["ffmpeg", "-hide_banner", "-h", "long"], capture_output=True, text=True).stdout
    return ["-fps_mode", "passthrough"] if "-fps_mode" in help_text else ["-vsync", "passthrough"]


def render(job, ass_text: str, out: Path) -> None:
    ass = out.with_suffix(".ass")
    ass.write_text(ass_text)
    filters = ("setpts=PTS-STARTPTS,scale=800:640:force_original_aspect_ratio=decrease:force_divisible_by=2:"
               "flags=lanczos,pad=800:640:(ow-iw)/2:(oh-ih)/2:color=0x101b29,setsar=1,"
               f"pad={W}:{H}:24:98:color=0x101b29,"
               "drawbox=x=848:y=98:w=408:h=640:color=0x1b2b3d:t=fill,"
               f"drawbox=x=12:y={FOOT_Y}:w=1256:h={FOOT_H - 8}:color=0x1b2b3d:t=fill,"
               f"ass=filename={ass}")
    temp = out.with_name(out.stem + "_rendering.mp4")
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning", "-y", "-threads", "2",
                    "-i", job["video"], "-vf", filters, "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset",
                    "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", *passthrough(), "-movflags", "+faststart",
                    str(temp)], check=True)
    temp.replace(out)


def page(rows, method: str) -> str:
    cards = []
    for job, phases, captions, ours, flora in rows:
        def listing(items, key):
            return "\n".join(f"{s['start_s']:6.1f}-{s['end_s']:6.1f}  {html.escape(str(s.get(key) or s.get('status', '')))}"
                             for s in items)
        caps = "\n".join(f"{c['start_s']:6.1f}-{c['end_s']:6.1f}  {html.escape(c.get('caption') or '')}" for c in captions)
        cards.append(
            f"<details><summary><b>{html.escape(job['group'])}</b> {job['id']} · {job['duration_s']:.1f} s · "
            f"ours {len(ours)} · FLORA {len(flora)}</summary>"
            f"<p>Task: {html.escape(job['task'])}</p>"
            f"<video controls preload=\"none\" width=\"960\" src=\"media/{job['id']}.mp4\"></video>"
            f"<div class=\"cols\"><div><h4>Ours</h4><pre>{listing(ours, 'instruction')}</pre></div>"
            f"<div><h4>FLORA</h4><pre>{listing(flora, 'caption')}</pre></div></div>"
            f"<details><summary>Our phases and their captions ({len(captions)})</summary><pre>{caps}</pre></details>"
            "</details>")
    return ("<!doctype html><meta charset=\"utf-8\"><title>Nero Dog: ours vs FLORA</title><style>body{font:15px/1.4 "
            "system-ui,sans-serif;margin:24px;background:#101923;color:#edf1f5}pre{white-space:pre-wrap;"
            "background:#1b2b3d;padding:8px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}"
            "summary{cursor:pointer;margin:6px 0}</style><h1>Nero Dog: ours vs FLORA</h1>"
            "<p>Ours: kinematic phases (autolabel, full_body settings, speed mode 1), Flora-Captioner per phase "
            "(three cameras, our template: task brief, objects, previous instruction, notes), combined into "
            "subtasks by gpt-6.1-sol (instruction v6; " + METHODS[method] + "). FLORA: "
            "TempSeg + Captioner of the seven-task pilot (20261105_seven_tasks_10_each). Model predictions; "
            "not verified.</p>" + "".join(cards))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--kinematic", required=True)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--flora", required=True, help="FLORA run folder with captioner/<id>/annotations.json")
    ap.add_argument("--method", default="C1", choices=sorted(METHODS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-video", action="store_true", help="only the page")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "media").mkdir(parents=True, exist_ok=True)
    kin = {json.loads(p.read_text())["id"]: json.loads(p.read_text())
           for p in Path(args.kinematic).glob("predictions/*/*.json")}
    rows = []
    for line in Path(args.jobs).read_text().splitlines():
        job = json.loads(line)
        seg_path = Path(args.segments) / f"{job['id']}.json"
        if not seg_path.exists():
            continue
        case = kin[job["id"]]
        ours = json.loads(seg_path.read_text())[args.method]
        captions = json.loads((Path(args.captions) / f"{job['id']}.json").read_text())["rounds"][0]
        flora = json.loads((Path(args.flora) / "captioner" / job["id"] / "annotations.json").read_text())["segments"]
        rows.append((job, case["predictions"], captions, ours, flora))
        if not args.no_video:
            render(job, subtitles(job, case["duration_sec"], case["predictions"], ours, flora),
                   out / "media" / f"{job['id']}.mp4")
            print("rendered", job["id"], flush=True)
    (out / "index.html").write_text(page(rows, args.method))
    print(f"wrote {out / 'index.html'} ({len(rows)} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
