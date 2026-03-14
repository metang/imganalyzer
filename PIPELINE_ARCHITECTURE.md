# PIPELINE ARCHITECTURE: FULL REPLACEMENT GUIDE
## Replacing BLIP2 & Cloud AI with Qwen3.5 & SigLIP-v2.5

**Date:** 2026-03-11 17:30:37
**Working Directory:** D:\Code\imganalyzer

---

## EXECUTIVE SUMMARY

The imganalyzer pipeline is a **resource-aware, multi-threaded GPU scheduler** that processes images through 10+ analysis modules. BLIP2 and Cloud AI are currently:

- **BLIP2**: Exclusive GPU module (6 GB VRAM) → imganalyzer/analysis/ai/local.py → writes to nalysis_blip2 table
- **Cloud AI**: Cloud thread pool module → imganalyzer/analysis/ai/cloud.py → writes to nalysis_cloud_ai table  
- **Aesthetic**: Piggy-backs on cloud_ai or runs standalone → writes to nalysis_aesthetic table

### Key Architectural Constraints:
1. **Prerequisite Locks**: Cloud AI + Aesthetic must run AFTER objects detection (has_people flag)
2. **Exclusive GPU**: BLIP2 runs alone in Phase 1 (blocks all other GPU work)
3. **Privacy Gate**: Both skip images with has_people=true
4. **Atomic Writes**: Results written in single DB transaction (delete+insert)
5. **FTS5 Index**: Results marked for deferred search index rebuild (batched every 60s)

---

## 1. MODULE DISPATCH & REGISTRATION

**File: imganalyzer/pipeline/modules.py (782 lines)**

### Dispatch Table (Lines 261-300)

The central _run_dispatch() method routes to individual runners:

| Module | Handler | Input | Output Table |
|--------|---------|-------|--------------|
| blip2 | _run_blip2() L377 | Image file | analysis_blip2 |
| cloud_ai | _run_cloud_ai() L424 | Image file | analysis_cloud_ai |
| aesthetic | _run_aesthetic() L463 | Image file | analysis_aesthetic |
| objects | _run_objects() L387 | Image file | analysis_objects |
| ocr | _run_ocr() L401 | Image file | analysis_ocr |
| faces | _run_faces() L411 | Image file | analysis_faces |
| metadata | _run_metadata() L304 | Image headers | analysis_metadata |
| technical | _run_technical() L329 | Image file | analysis_technical |
| embedding | _run_embedding() L547 | Text/image | embeddings |

---

### BLIP2 Runner (Lines 377-385)

```python
def _run_blip2(self, image_id: int, path: Path) -> dict[str, Any]:
    image_data = self._cached_read_image(path, image_id)
    
    from imganalyzer.pipeline.passes.blip2 import run_blip2
    result = run_blip2(image_data, self.repo, image_id, self.conn)
    
    if self.verbose:
        console.print(f"  [dim]BLIP-2 done for image {image_id}[/dim]")
    return result
```

**Actual inference:** imganalyzer/pipeline/passes/blip2.py → calls LocalAI.analyze() from imganalyzer/analysis/ai/local.py

**Returns:**
`json
{
  "description": "str",
  "scene_type": "str",
  "main_subject": "str",
  "lighting": "str",
  "mood": "str",
  "keywords": ["str", "..."]
}
`

---

### Cloud AI Runner (Lines 424-461)

```python
def _run_cloud_ai(self, image_id: int, path: Path) -> dict[str, Any]:
    # Step 1: Privacy gate — skip if image has people
    local_data = self.repo.get_analysis(image_id, "local_ai")
    if local_data and local_data.get("has_people"):
        return {}
    
    # Step 2: Load image
    image_data = self._cached_read_image(path, image_id)
    
    # Step 3: Call cloud LLM
    from imganalyzer.analysis.ai.cloud import CloudAI
    result = CloudAI(backend=self.cloud_provider).analyze(path, image_data)
    
    # Step 4: Extract aesthetic fields
    aesthetic_score = result.pop("aesthetic_score", None)
    aesthetic_label = result.pop("aesthetic_label", None)
    aesthetic_reason = result.pop("aesthetic_reason", None)
    
    # Step 5: Atomic DB write (BOTH tables in one transaction)
    with _transaction(self.conn):
        self.repo.upsert_cloud_ai(image_id, self.cloud_provider, result)
        if aesthetic_score is not None:
            self.repo.upsert_aesthetic(image_id, {
                "aesthetic_score": aesthetic_score,
                "aesthetic_label": aesthetic_label or "",
                "aesthetic_reason": aesthetic_reason or "",
                "provider": self.cloud_provider,
            })
    
    return result
```

**Key Pattern:** Cloud AI writes to BOTH nalysis_cloud_ai AND nalysis_aesthetic tables in a single transaction (Lines 445-455)

---

### Aesthetic Runner (Lines 463-529)

```python
def _run_aesthetic(self, image_id: int, path: Path) -> dict[str, Any]:
    # Step 1: Privacy gate
    local_data = self.repo.get_analysis(image_id, "local_ai")
    if local_data and local_data.get("has_people"):
        return {}
    
    # Step 2: Check if cloud_ai is running (avoid double API call)
    cloud_ai_active = self.conn.execute(
        """SELECT 1 FROM job_queue
           WHERE image_id = ? AND module = 'cloud_ai'
             AND status IN ('pending', 'running')
           LIMIT 1""",
        (image_id,),
    ).fetchone()
    if cloud_ai_active:
        return {}  # Defer to cloud_ai
    
    # Step 3: Check if cloud_ai already wrote aesthetic
    existing = self.repo.get_analysis(image_id, "aesthetic")
    if existing and existing.get("aesthetic_score") is not None:
        return existing  # Already populated
    
    # Step 4: Fallback — aesthetic runs solo
    image_data = self._cached_read_image(path, image_id)
    from imganalyzer.analysis.ai.cloud import CloudAI
    result = CloudAI(backend=self.cloud_provider).analyze(path, image_data)
    
    aesthetic_data = {
        "aesthetic_score": result.get("aesthetic_score"),
        "aesthetic_label": result.get("aesthetic_label"),
        "aesthetic_reason": result.get("aesthetic_reason", ""),
        "provider": self.cloud_provider,
    }
    
    # Step 5: Write to DB
    with _transaction(self.conn):
        self.repo.upsert_aesthetic(image_id, aesthetic_data)
    
    return aesthetic_data
```

**Key Logic:** Aesthetic is "parasitic" on cloud_ai — checks if cloud_ai is running (L476-488), and if cloud_ai already wrote aesthetic (L494-505).

---

## 2. DB UPSERT PATTERN

**File: imganalyzer/db/repository.py (1404 lines)**

### Module → Table Mapping (Lines 20-34)

```python
MODULE_TABLE_MAP: dict[str, str] = {
    "metadata":   "analysis_metadata",
    "technical":  "analysis_technical",
    "local_ai":   "analysis_local_ai",
    "blip2":      "analysis_blip2",
    "objects":    "analysis_objects",
    "ocr":        "analysis_ocr",
    "faces":      "analysis_faces",
    "cloud_ai":   "analysis_cloud_ai",
    "aesthetic":  "analysis_aesthetic",
    "perception": "analysis_perception",
    "embedding":  "embeddings",
}
```

### upsert_cloud_ai (Lines 214-236)

```python
def upsert_cloud_ai(self, image_id: int, provider: str, data: dict[str, Any]) -> None:
    """Atomic write of cloud AI result for a specific provider."""
    # Step 1: JSON-encode list fields
    for key in ("keywords", "detected_objects", "dominant_colors_ai"):
        if key in data and isinstance(data[key], list):
            data[key] = json.dumps(data[key])
    
    # Step 2: Filter to known columns (schema-safe)
    data = self._filter_to_known_columns("analysis_cloud_ai", data)
    
    # Step 3: Apply override mask (user overrides protect fields)
    data = self._apply_override_mask(image_id, "analysis_cloud_ai", data)
    
    # Step 4: DELETE old row (idempotent upsert)
    self.conn.execute(
        "DELETE FROM analysis_cloud_ai WHERE image_id = ? AND provider = ?",
        [image_id, provider],
    )
    
    # Step 5: INSERT new row (atomic)
    data["provider"] = provider
    cols = ["image_id"] + list(data.keys()) + ["analyzed_at"]
    placeholders = ", ".join(["?"] * len(cols))
    col_str = ", ".join(cols)
    vals = [image_id] + list(data.values()) + [_now()]
    self.conn.execute(
        f"INSERT INTO analysis_cloud_ai ({col_str}) VALUES ({placeholders})", vals
    )
```

### upsert_aesthetic (Lines 238-248)

```python
def upsert_aesthetic(self, image_id: int, data: dict[str, Any]) -> None:
    """Atomic write of the aesthetic analysis result."""
    data = self._apply_override_mask(image_id, "analysis_aesthetic", data)
    self.conn.execute("DELETE FROM analysis_aesthetic WHERE image_id = ?", [image_id])
    cols = ["image_id"] + list(data.keys()) + ["analyzed_at"]
    placeholders = ", ".join(["?"] * len(cols))
    col_str = ", ".join(cols)
    vals = [image_id] + list(data.values()) + [_now()]
    self.conn.execute(
        f"INSERT INTO analysis_aesthetic ({col_str}) VALUES ({placeholders})", vals
    )
```

### upsert_blip2 (Lines 262-275)

```python
def upsert_blip2(self, image_id: int, data: dict[str, Any]) -> None:
    """Atomic write of the BLIP-2 captioning result."""
    if "keywords" in data and isinstance(data["keywords"], list):
        data["keywords"] = json.dumps(data["keywords"])
    data = self._filter_to_known_columns("analysis_blip2", data)
    data = self._apply_override_mask(image_id, "analysis_blip2", data)
    self.conn.execute("DELETE FROM analysis_blip2 WHERE image_id = ?", [image_id])
    cols = ["image_id"] + list(data.keys()) + ["analyzed_at"]
    placeholders = ", ".join(["?"] * len(cols))
    col_str = ", ".join(cols)
    vals = [image_id] + list(data.values()) + [_now()]
    self.conn.execute(
        f"INSERT INTO analysis_blip2 ({col_str}) VALUES ({placeholders})", vals
    )
```

**Pattern:** Always DELETE then INSERT for idempotency. The worker wraps these in _transaction(self.conn) blocks.

---

## 3. GPU PHASE SCHEDULING

**File: imganalyzer/pipeline/scheduler.py (479 lines)**

### Module Classification (Lines 27-63)

```python
GPU_MODULES: frozenset[str] = frozenset({
    "objects", "blip2", "ocr", "faces", "embedding", "perception",
})
LOCAL_IO_MODULES: frozenset[str] = frozenset({"metadata", "technical"})
CLOUD_MODULES: frozenset[str] = frozenset({"cloud_ai", "aesthetic"})
IO_MODULES: frozenset[str] = LOCAL_IO_MODULES | CLOUD_MODULES

# Dependency graph: module -> prerequisite that must complete first.
_PREREQUISITES: dict[str, str] = {
    "cloud_ai":  "objects",
    "aesthetic": "objects",
    "ocr":       "objects",
    "faces":     "objects",
}

# GPU modules that support batched forward passes.
_BATCH_CAPABLE: frozenset[str] = frozenset({"objects", "blip2", "embedding"})

# Ordered phases for GPU execution.
_GPU_PHASES: list[list[str]] = [
    ["objects"],              # Phase 0: Unlock dependencies (has_person flag)
    ["blip2"],                # Phase 1: Exclusive (large model, 6 GB)
    ["faces", "ocr", "embedding"],  # Phase 2: Co-resident (~2.75 GB total)
]

# GPU modules that run independently
INDEPENDENT_GPU_MODULES: frozenset[str] = frozenset({"perception"})
```

**Critical:** BLIP2 runs ALONE in Phase 1. Cloud_ai and aesthetic don't run until objects completes.

---

## 4. VRAM BUDGET TRACKING

**File: imganalyzer/pipeline/vram_budget.py (165 lines)**

### VRAM Allocations (Lines 19-30)

```python
_MODULE_VRAM_GB: dict[str, float] = {
    "objects":    2.4,   # GroundingDINO mixed fp16/fp32, batch=4
    "blip2":     6.0,   # BLIP-2 FlanT5-XL fp16 + generation working set
    "ocr":       1.3,   # TrOCR large-printed fp16
    "faces":     1.0,   # InsightFace buffalo_l ONNX (1 GB arena cap)
    "embedding": 0.95,  # CLIP ViT-L/14 fp16, batch=16
    "perception": 13.8, # UniPercept 4-bit NF4 quantized (14 GiB max_memory cap)
}

# Modules that must run alone (peak VRAM > 50% of a typical budget).
_EXCLUSIVE_MODULES: frozenset[str] = frozenset({"blip2", "perception"})
\\\

**For Qwen3.5 replacement:**
- Measure actual VRAM usage (model + batch activations)
- If Qwen3.5 + SigLIP_embeddings ≤ 3 GB → can move to Phase 2
- If > 3 GB → keep as exclusive Phase 1

---

## 5. WORKER DISPATCH

**File: imganalyzer/pipeline/worker.py (907 lines)**

### Module Classification (Lines 66-89)

```python
GPU_MODULES = {"local_ai", "embedding", "blip2", "objects", "ocr", "faces", "perception"}
LOCAL_IO_MODULES = {"metadata", "technical"}
CLOUD_MODULES = {"cloud_ai", "aesthetic"}
IO_MODULES = LOCAL_IO_MODULES | CLOUD_MODULES

# Modules whose output contributes to the FTS5 search index.
_FTS_MODULES = {"metadata", "local_ai", "blip2", "faces", "cloud_ai"}

# The `objects` pass must complete for an image before cloud/aesthetic may run
_PREREQUISITES: dict[str, str] = {
    "cloud_ai": "objects",
    "aesthetic": "objects",
    "ocr": "objects",
    "faces": "objects",
    "embedding": "objects",
}
```

### GPU Phase Execution Loop (Lines 450-490)

```python
phase_labels = [
    "Phase 0 — object detection (people flag)",
    "Phase 1 — BLIP-2 captioning (exclusive GPU)",
    "Phase 2 — faces + OCR + embeddings (co-resident GPU)",
]

for phase_idx in range(len(scheduler.gpu_phases)):
    if self._shutdown.is_set():
        break
    
    phase_modules = scheduler.modules_for_phase(phase_idx)
    has_pending = any(
        self.queue.pending_count(module=mod) > 0
        for mod in phase_modules
    )
    if not has_pending:
        continue
    
    console.print(f"[dim]{phase_labels[phase_idx]}[/dim]")
    
    with (
        self.profiler.span("gpu_phase", phase=phase_idx),
        ThreadPoolExecutor(max_workers=self.workers)       as local_pool,
        ThreadPoolExecutor(max_workers=effective_cloud_workers) as cloud_pool,
    ):
        scheduler.run_gpu_phase(
            phase_idx,
            claim_fn=_claim_fn,
            process_batch_fn=self._process_job_batch,
            process_single_fn=self._process_job,
            submit_io_fn=_submit_io_jobs,
            collect_fn=_collect_futures,
            advance_fn=_advance_fn,
            flush_fn=self._maybe_periodic_flush,
            local_pool=local_pool,
            cloud_pool=cloud_pool,
            stats=stats,
            unload_fn=unload_gpu_model,
            prefetch_fn=_prefetch_image,
            cancel_futures_fn=_cancel_futures,
        )
```

### Batch Sizes (Lines 595-601)

```python
_GPU_BATCH_SIZES: dict[str, int] = {
    "objects":   4,    # GroundingDINO
    "blip2":     1,    # BLIP-2 (beam search requires low batch)
    "embedding": 16,   # CLIP ViT-L/14
    "faces":     8,    # InsightFace
    "ocr":       4,    # TrOCR
}
```

**For Qwen3.5:** Test batch sizes; may support 2-4 depending on VRAM.

---

## 6. JOB QUEUE & ENQUEUEING

**File: imganalyzer/db/queue.py (598 lines)**

### Enqueue Logic (Lines 30-84)

```python
def enqueue(
    self,
    image_id: int,
    module: str,
    priority: int = 0,
    force: bool = False,
    _auto_commit: bool = True,
) -> int | None:
    """Add a job unless an identical pending/running/done job exists."""
    existing = self.conn.execute(
        "SELECT id, status FROM job_queue WHERE image_id = ? AND module = ?",
        [image_id, module],
    ).fetchone()
    
    if existing:
        if existing["status"] in ("pending", "running"):
            return None  # already queued
        if not force:
            return None  # don't re-enqueue without force
        # force=True: reset to pending
        self.conn.execute(
            """UPDATE job_queue
               SET status = 'pending', attempts = 0, error_message = NULL,
                   skip_reason = NULL, started_at = NULL, completed_at = NULL,
                   queued_at = ?, priority = ?,
                   last_node_id = NULL, last_node_role = NULL
                WHERE id = ?""",
            [_now(), priority, existing["id"]],
        )
        if _auto_commit:
            self.conn.commit()
        return existing["id"]
    
    cur = self.conn.execute(
        """INSERT INTO job_queue (image_id, module, priority, status, queued_at)
           VALUES (?, ?, ?, 'pending', ?)""",
        [image_id, module, priority, _now()],
    )
    if _auto_commit:
        self.conn.commit()
    return cur.lastrowid
```

### Claim (Atomic) (Lines 104-153)

```python
def claim(
    self,
    batch_size: int = 1,
    module: str | None = None,
    node_id: str = "master",
    node_role: str = "master",
) -> list[dict[str, Any]]:
    """Atomically claim up to *batch_size* pending jobs."""
    where = "WHERE status = 'pending'"
    params: list[Any] = []
    if module:
        where += " AND module = ?"
        params.append(module)
    params.append(batch_size)
    
    # Use BEGIN IMMEDIATE for atomic SELECT + UPDATE
    self.conn.execute("BEGIN IMMEDIATE")
    try:
        rows = self.conn.execute(
            f"""SELECT id, image_id, module, attempts
                FROM job_queue
                {where}
                ORDER BY priority DESC, queued_at ASC
                LIMIT ?""",
            params,
        ).fetchall()
        
        if not rows:
            self.conn.rollback()
            return []
        
        job_ids = [r["id"] for r in rows]
        placeholders = ",".join("?" * len(job_ids))
        self.conn.execute(
            f"""UPDATE job_queue
                SET status = 'running', started_at = ?,
                    last_node_id = ?, last_node_role = ?
                WHERE id IN ({placeholders})""",
            [_now(), node_id, node_role] + job_ids,
        )
        self.conn.commit()
    except Exception:
        self.conn.rollback()
        raise
    return [dict(r) for r in rows]
```

---

## 7. BLIP2 LOCAL MODEL CODE

**File: imganalyzer/analysis/ai/local.py (247 lines)**

### Model ID & Loading (Lines 19-75)

```python
_MODEL_ID = "Salesforce/blip2-flan-t5-xl"

class LocalAI:
    _processor = None
    _model = None
    
    @classmethod
    def _unload(cls) -> None:
        """Unload BLIP-2 model from GPU to free VRAM."""
        if cls._model is not None:
            del cls._model
            cls._model = None
        if cls._processor is not None:
            del cls._processor
            cls._processor = None
        try:
            from imganalyzer.device import empty_cache
            empty_cache()
        except Exception:
            pass

# Load model lazily
if LocalAI._processor is None:
    from rich.console import Console
    Console().print(f"[dim]Loading BLIP-2 model {_MODEL_ID}...[/dim]")
    LocalAI._processor = Blip2Processor.from_pretrained(
        _MODEL_ID, cache_dir=CACHE_DIR
    )
    from imganalyzer.device import get_device, supports_fp16
    device = get_device()
    LocalAI._model = Blip2ForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.float16 if supports_fp16() else torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    ).to(device)
```

### Single Image Inference (Lines 47-137)

```python
def analyze(self, image_data: dict[str, Any]) -> dict[str, Any]:
    rgb: np.ndarray = image_data["rgb_array"]
    pil_img = Image.fromarray(rgb)
    
    # (model loading code above)
    processor = LocalAI._processor
    model = LocalAI._model
    device = next(model.parameters()).device
    
    results: dict[str, Any] = {}
    
    # 1. Image captioning
    with torch.inference_mode():
        inputs = processor(pil_img, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=100)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()
    results["description"] = caption
    
    # 2. VQA — batch all 4 questions
    vqa_questions = [
        ("What type of scene is this? Answer in 1-3 words.", "scene_type"),
        ("What is the main subject of this image? Answer in 1-5 words.", "main_subject"),
        ("What is the lighting condition or time of day? Answer in 1-3 words.", "lighting"),
        ("What is the mood or aesthetic of this image? Answer in 1-3 words.", "mood"),
    ]
    
    try:
        with torch.inference_mode():
            questions = [q for q, _ in vqa_questions]
            images = [pil_img] * len(questions)
            inputs = processor(images, questions, return_tensors="pt", padding=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=30)
            
            for idx, (_, key) in enumerate(vqa_questions):
                answer = processor.decode(outputs[idx], skip_special_tokens=True).strip()
                if answer:
                    results[key] = answer
    except Exception:
        # Fallback: sequential VQA if batch fails
        for question, key in vqa_questions:
            try:
                with torch.inference_mode():
                    inputs = processor(pil_img, question, return_tensors="pt").to(device)
                    out = model.generate(**inputs, max_new_tokens=30)
                    answer = processor.decode(out[0], skip_special_tokens=True).strip()
                if answer:
                    results[key] = answer
            except Exception:
                pass
    
    # 3. Keywords from caption
    results["keywords"] = _extract_keywords(caption)
    
    return results
```

### Batch Inference (Lines 139-238)

```python
def analyze_batch(self, image_data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run BLIP-2 on multiple images in batched forward passes.
    
    Batch_size=2 is safe max (beam search memory scales linearly).
    Falls back to sequential on OOM.
    """
    if not image_data_list:
        return []
    if len(image_data_list) == 1:
        return [self.analyze(image_data_list[0])]
    
    # (model loading code)
    
    pil_images = [Image.fromarray(d["rgb_array"]) for d in image_data_list]
    n = len(pil_images)
    
    vqa_questions = [
        ("What type of scene is this? Answer in 1-3 words.", "scene_type"),
        ("What is the main subject of this image? Answer in 1-5 words.", "main_subject"),
        ("What is the lighting condition or time of day? Answer in 1-3 words.", "lighting"),
        ("What is the mood or aesthetic of this image? Answer in 1-3 words.", "mood"),
    ]
    
    try:
        # 1. Batched captioning
        with torch.inference_mode():
            inputs = processor(pil_images, return_tensors="pt", padding=True).to(device)
            out = model.generate(**inputs, max_new_tokens=100)
            captions = [processor.decode(out[i], skip_special_tokens=True).strip() for i in range(n)]
        
        # 2. Batched VQA (N images × 4 questions = N*4 items)
        vqa_images = []
        vqa_texts = []
        for img in pil_images:
            for q, _ in vqa_questions:
                vqa_images.append(img)
                vqa_texts.append(q)
        
        vqa_answers: list[str] = []
        with torch.inference_mode():
            inputs = processor(vqa_images, vqa_texts, return_tensors="pt", padding=True).to(device)
            out = model.generate(**inputs, max_new_tokens=30)
            vqa_answers = [processor.decode(out[i], skip_special_tokens=True).strip() for i in range(len(vqa_images))]
        
        # 3. Assemble per-image results
        all_results: list[dict[str, Any]] = []
        for img_idx in range(n):
            results: dict[str, Any] = {"description": captions[img_idx]}
            for q_idx, (_, key) in enumerate(vqa_questions):
                answer = vqa_answers[img_idx * len(vqa_questions) + q_idx]
                if answer:
                    results[key] = answer
            results["keywords"] = _extract_keywords(captions[img_idx])
            all_results.append(results)
        
        return all_results
    
    except Exception:
        # OOM or error — fall back to sequential
        return [self.analyze(d) for d in image_data_list]
```

---

## REPLACEMENT STEPS FOR QWEN3.5 + SIGLIP-V2.5

### 1. Measure VRAM Usage

Create imganalyzer/analysis/ai/qwen_siglip.py:

```python
class QwenSigLIPAnalyzer:
    _qwen_model = None
    _qwen_processor = None
    _siglip_model = None
    _siglip_processor = None
    
    @classmethod
    def _unload(cls) -> None:
        """Unload models from GPU."""
        for attr in ("_qwen_model", "_qwen_processor", "_siglip_model", "_siglip_processor"):
            if getattr(cls, attr) is not None:
                setattr(cls, attr, None)
        try:
            from imganalyzer.device import empty_cache
            empty_cache()
        except Exception:
            pass
    
    def analyze(self, image_data: dict[str, Any]) -> dict[str, Any]:
        """Single image inference."""
        rgb: np.ndarray = image_data["rgb_array"]
        pil_img = Image.fromarray(rgb)
        
        # Load models lazily
        self._ensure_loaded()
        
        # Qwen captioning + 4 VQA questions (same as BLIP2)
        # Return: {description, scene_type, main_subject, lighting, mood, keywords}
        pass
    
    def analyze_batch(self, image_data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch inference."""
        # Returns list of result dicts
        pass
```

### 2. Update VRAM Budget (vram_budget.py, lines 19-30)

```python
_MODULE_VRAM_GB: dict[str, float] = {
    "objects":    2.4,
    "blip2":     5.5,    # <-- Update to Qwen3.5 + SigLIP combined VRAM
    "ocr":       1.3,
    "faces":     1.0,
    "embedding": 0.95,
    "perception": 13.8,
}

# If Qwen3.5 + SigLIP can coexist with Phase 2 models:
# Move from Phase 1 (exclusive) to Phase 2 (co-resident)
# KEEP _EXCLUSIVE_MODULES as-is if > 3 GB
_EXCLUSIVE_MODULES: frozenset[str] = frozenset({"blip2", "perception"})
```

### 3. Update Module Dispatch (modules.py, lines 377-385)

```python
def _run_blip2(self, image_id: int, path: Path) -> dict[str, Any]:
    image_data = self._cached_read_image(path, image_id)
    
    from imganalyzer.pipeline.passes.qwen_siglip import run_qwen_siglip_batch
    result = run_qwen_siglip_batch([image_data], self.repo, [image_id], self.conn)[0]
    
    if self.verbose:
        console.print(f"  [dim]Qwen-SigLIP done for image {image_id}[/dim]")
    return result
```

### 4. Create Batch Pass (imganalyzer/pipeline/passes/qwen_siglip.py)

```python
def run_qwen_siglip_batch(
    image_data_list: list[dict[str, Any]],
    repo: Repository,
    image_ids: list[int],
    conn: sqlite3.Connection,
) -> None:
    """Batch Qwen + SigLIP inference."""
    from imganalyzer.analysis.ai.qwen_siglip import QwenSigLIPAnalyzer
    
    analyzer = QwenSigLIPAnalyzer()
    results = analyzer.analyze_batch(image_data_list)
    
    # Write to analysis_blip2 table (same schema)
    from imganalyzer.pipeline.modules import _transaction
    with _transaction(conn):
        for image_id, result in zip(image_ids, results):
            repo.upsert_blip2(image_id, result)
```

### 5. Update Worker Batch Dispatch (worker.py, lines 714-718)

```python
elif module == "blip2":
    from imganalyzer.pipeline.passes.qwen_siglip import run_qwen_siglip_batch
    run_qwen_siglip_batch(
        valid_image_data, repo, valid_image_ids, runner.conn,
    )
```

### 6. Update Batch Sizes (worker.py, lines 595-601)

```python
_GPU_BATCH_SIZES: dict[str, int] = {
    "objects":   4,
    "blip2":     2,    # <-- Test; may support higher with Qwen3.5
    "embedding": 16,
    "faces":     8,
    "ocr":       4,
}
```

### 7. Optional: Replace Cloud AI with Local Qwen Evaluation

If cloud_ai should also use Qwen locally:

```python
# In modules.py _run_cloud_ai (line 424)
def _run_cloud_ai(self, image_id: int, path: Path) -> dict[str, Any]:
    # Skip cloud API; aesthetic/description computed by _run_aesthetic
    return {}

# In modules.py _run_aesthetic (line 463)
def _run_aesthetic(self, image_id: int, path: Path) -> dict[str, Any]:
    # Privacy gate
    local_data = self.repo.get_analysis(image_id, "local_ai")
    if local_data and local_data.get("has_people"):
        return {}
    
    image_data = self._cached_read_image(path, image_id)
    from imganalyzer.analysis.ai.qwen_siglip import QwenSigLIPAnalyzer
    
    result = QwenSigLIPAnalyzer().analyze(image_data)
    
    aesthetic_data = {
        "aesthetic_score": result.get("aesthetic_score", 0.5),
        "aesthetic_label": result.get("aesthetic_label", ""),
        "aesthetic_reason": result.get("aesthetic_reason", ""),
        "provider": "local_qwen",
    }
    
    with _transaction(self.conn):
        self.repo.upsert_aesthetic(image_id, aesthetic_data)
    
    return aesthetic_data
```

---

## KEY CONSTRAINTS & MIGRATION CHECKLIST

| # | Constraint | Current | Action Required |
|---|-----------|---------|-----------------|
| 1 | **GPU Phase** | BLIP2 in Phase 1 (exclusive) | Measure Qwen VRAM; if ≤3 GB move to Phase 2 |
| 2 | **Batch Size** | blip2=1 (beam search) | Test Qwen batch sizes (2-8 likely safe) |
| 3 | **Output Schema** | description, scene_type, main_subject, lighting, mood, keywords | Map Qwen output to exact field names |
| 4 | **has_people Gate** | Cloud AI / aesthetic skip if has_people=true | Qwen must accept has_people flag |
| 5 | **VRAM Budget** | blip2=6.0 GB | Update to Qwen+SigLIP combined VRAM |
| 6 | **Exclusive Lock** | blip2 marked exclusive | Keep exclusive if > 3 GB, remove if < 3 GB |
| 7 | **DB Upsert** | Cloud AI writes BOTH cloud_ai + aesthetic tables | Qwen writes analysis_blip2; aesthetic separate |
| 8 | **FTS Index** | "blip2" in _FTS_MODULES | Keep "blip2" (table unchanged) |
| 9 | **Model Unload** | Worker calls LocalAI._unload() at phase boundary | Update to QwenSigLIPAnalyzer._unload() |
| 10 | **Batch Pass** | run_blip2_batch() in pipeline/passes/blip2.py | Create run_qwen_siglip_batch() |

