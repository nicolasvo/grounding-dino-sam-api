from __future__ import annotations

import modal

try:
    from fastapi import Request  # noqa: F401
except ImportError:
    pass

SAM_MODEL = "facebook/sam2.1-hiera-tiny"
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "git+https://github.com/facebookresearch/sam2.git",
        "transformers>=4.40,<5",
        "huggingface-hub<1",
        "timm",
        "einops",
        "fastapi[standard]",
        "pillow",
        "opencv-python",
        "numpy<2",
    )
    .run_commands(
        f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{SAM_MODEL}')\"",
        f"python -c \"from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; AutoProcessor.from_pretrained('{GROUNDING_DINO_MODEL}'); AutoModelForZeroShotObjectDetection.from_pretrained('{GROUNDING_DINO_MODEL}')\"",
    )
    .add_local_python_source("image", copy=True)
)

app = modal.App("grounding-dino-sam")


@app.cls(
    image=image,
    gpu="T4",
    memory=8192,
    enable_memory_snapshot=True,
    scaledown_window=60,
)
class GroundingDinoSam:
    @modal.enter()
    def load(self):
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = SAM2ImagePredictor.from_pretrained(SAM_MODEL, device=self.device)
        self.gd_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
        self.gd_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL)
            .to(self.device)
            .eval()
        )

    @modal.fastapi_endpoint(method="POST")
    async def sticker(self, request: Request):
        import base64
        import json
        import logging
        import os
        import tempfile

        from image import make_sticker, segment

        log = logging.getLogger("gsa")
        raw = await request.body()
        log.info("incoming: ct=%s len=%d", request.headers.get("content-type"), len(raw))
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            return {"error": "invalid json", "detail": str(e)}

        image_b64 = payload.get("image") if isinstance(payload, dict) else None
        text_prompt = payload.get("text_prompt") if isinstance(payload, dict) else None
        mode = payload.get("mode", "sticker") if isinstance(payload, dict) else "sticker"
        if not image_b64 or not text_prompt:
            return {"image": None, "error": "missing 'image' or 'text_prompt'"}

        with tempfile.TemporaryDirectory(dir="/tmp/") as tmpdir:
            input_path = os.path.join(tmpdir, "input.jpeg")
            output_path = os.path.join(tmpdir, "output.png")
            input_then_path = os.path.join(tmpdir, "input_then.jpeg")
            with open(input_path, "wb") as f:
                f.write(base64.b64decode(image_b64))

            if mode == "segment":
                import cv2

                img = segment(
                    input_path,
                    text_prompt,
                    self.sam,
                    self.gd_processor,
                    self.gd_model,
                    self.device,
                )
                if img is False:
                    return {"image": None, "error": "not_detected"}
                cv2.imwrite(output_path, img)
            else:
                ok = make_sticker(
                    input_path,
                    output_path,
                    text_prompt,
                    input_then_path,
                    self.sam,
                    self.gd_processor,
                    self.gd_model,
                    self.device,
                )
                if not ok:
                    return {"image": None, "error": "not_detected"}

            with open(output_path, "rb") as f:
                return {"image": base64.b64encode(f.read()).decode()}
