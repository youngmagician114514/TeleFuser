"""Single-session adapter for the official Matrix-Game 2.0 chunk runtime.

The upstream repository keeps the causal block loop inside its interactive
``inference`` function.  This module reuses its model components but owns one
session's condition tensors, KV caches, and causal VAE decoder cache so
TeleFuser can call one block at a time.  Imports are intentionally lazy: the
Matrix environment (apex/FlashAttention and model weights) is optional.
"""

from __future__ import annotations

import copy
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from telefuser.service.livekit.model_contracts import ModelServingCapabilities

from .action import MatrixGameAction
from .backend import (
    MatrixGameBackendConfig,
    MatrixGameBackendError,
    MatrixGameSessionBackend,
    MatrixGameStep,
)


@dataclass
class _MatrixSession:
    session_id: str
    mode: str
    conditional_dict: dict[str, Any]
    vae_cache: list[Any]
    current_start_frame: int
    max_latent_frames: int
    closed: bool = False


class MatrixGameOfficialBackend(MatrixGameSessionBackend):
    """A real, single-session Matrix-Game 2.0 chunk backend.

    The backend deliberately rejects a second active session.  The official
    model's caches are single-batch tensors; sharing or copying them between
    sessions would silently corrupt causal state.  Multi-session batching can
    be added later by making this state explicit in the model runtime.
    """

    max_sessions = 1

    def __init__(self, config: MatrixGameBackendConfig) -> None:
        self.config = config
        self.capabilities = ModelServingCapabilities(
            model_name="matrix-game-2.0",
            output_fps=float(config.output_fps),
            frames_per_job=4 * int(config.num_frame_per_block),
            supports_cross_session_batching=False,
            supports_session_state_transfer=False,
        )
        self._started = False
        self._active_session: _MatrixSession | None = None
        self._root: Path | None = None
        self._torch: Any = None
        self._pipeline: Any = None
        self._vae: Any = None
        self._decoder: Any = None
        self._zero_vae_cache: Any = None
        self._dtype: Any = None
        self._device: Any = None

    def start(self) -> None:
        if self._started:
            return
        root = Path(self.config.model_root or "").expanduser().resolve()
        if not root.is_dir():
            raise MatrixGameBackendError(
                "MATRIX_GAME_ROOT must point to the Matrix-Game-2 source/model directory; "
                f"got {root}"
            )
        pretrained_root = Path(self.config.pretrained_model_path or root).expanduser().resolve()
        config_path = Path(self.config.config_path or root / "configs/inference_yaml/inference_universal.yaml")
        checkpoint_path = self.config.checkpoint_path
        if not config_path.is_file():
            raise MatrixGameBackendError(f"Matrix inference config does not exist: {config_path}")
        if not pretrained_root.is_dir():
            raise MatrixGameBackendError(f"Matrix pretrained model directory does not exist: {pretrained_root}")
        if not checkpoint_path:
            raise MatrixGameBackendError(
                "MATRIX_GAME_CHECKPOINT must point to the Matrix-Game 2.0 universal/templerun checkpoint; "
                "refusing to start an uninitialized random model"
            )
        if not Path(checkpoint_path).expanduser().is_file():
            raise MatrixGameBackendError(f"Matrix checkpoint does not exist: {checkpoint_path}")

        try:
            self._load_runtime(root, pretrained_root, config_path, checkpoint_path)
        except MatrixGameBackendError:
            raise
        except Exception as exc:  # pragma: no cover - requires optional CUDA runtime
            self._release_runtime()
            raise MatrixGameBackendError(f"Failed to initialize Matrix-Game 2.0 runtime: {exc}") from exc
        self._started = True

    def stop(self) -> None:
        self._active_session = None
        self._release_runtime()
        self._started = False

    def create_session(self, session_id: str, config: dict[str, Any]) -> _MatrixSession:
        self._require_started()
        if self._active_session is not None and not self._active_session.closed:
            raise MatrixGameBackendError("Matrix-Game official backend supports one active session")
        assert self._torch is not None and self._vae is not None and self._decoder is not None
        assert self._pipeline is not None and self._zero_vae_cache is not None

        image = self._load_image(config)
        mode = str(config.get("mode", self.config.mode)).strip().lower()
        if mode not in {"universal", "templerun"}:
            raise ValueError(f"Matrix-Game mode must be 'universal' or 'templerun', got {mode!r}")
        block_size = int(self.config.num_frame_per_block)
        if block_size <= 0:
            raise ValueError("MATRIX_GAME_FRAMES_PER_BLOCK must be positive")
        max_latent_frames = int(config.get("max_latent_frames", os.getenv("MATRIX_GAME_MAX_LATENT_FRAMES", "15")))
        if max_latent_frames < block_size:
            raise ValueError("Matrix-Game max_latent_frames must cover at least one frame block")

        image_tensor = self._image_tensor(image)
        with self._torch.inference_mode():
            # Encode only the first image frame.  The remaining causal latent
            # slots are zero-padded in latent space, avoiding a large VAE pass
            # over an artificial sequence of blank RGB frames.
            image_latent = self._vae.encode(
                image_tensor,
                device=self._device,
                tiled=True,
                tile_size=[44, 80],
                tile_stride=[23, 38],
            ).to(device=self._device, dtype=self._dtype)
            if image_latent.shape[2] != 1:
                raise MatrixGameBackendError(
                    f"Expected one latent image frame, got temporal shape {tuple(image_latent.shape)}"
                )
            padding = self._torch.zeros(
                image_latent.shape[0],
                image_latent.shape[1],
                max_latent_frames - 1,
                image_latent.shape[3],
                image_latent.shape[4],
                device=self._device,
                dtype=image_latent.dtype,
            )
            img_cond = self._torch.cat([image_latent, padding], dim=2)
            mask_cond = self._torch.zeros_like(img_cond)
            mask_cond[:, :, :1] = 1
            cond_concat = self._torch.cat([mask_cond[:, :4], img_cond], dim=1)
            visual_context = self._vae.clip.encode_video(image_tensor)

        frame_count = 1 + 4 * (max_latent_frames - 1)
        keyboard_dim = 4 if mode == "universal" else 7
        conditional: dict[str, Any] = {
            "cond_concat": cond_concat,
            "visual_context": visual_context.to(device=self._device, dtype=self._dtype),
            "keyboard_cond": self._torch.zeros(
                1, frame_count, keyboard_dim, device=self._device, dtype=self._dtype
            ),
        }
        if mode != "templerun":
            conditional["mouse_cond"] = self._torch.zeros(
                1, frame_count, 2, device=self._device, dtype=self._dtype
            )
        session = _MatrixSession(
            session_id=session_id,
            mode=mode,
            conditional_dict=conditional,
            vae_cache=copy.deepcopy(self._zero_vae_cache),
            current_start_frame=0,
            max_latent_frames=max_latent_frames,
        )
        self._pipeline._initialize_kv_cache(batch_size=1, dtype=self._dtype, device=self._device)
        self._pipeline._initialize_kv_cache_mouse_and_keyboard(
            batch_size=1, dtype=self._dtype, device=self._device
        )
        self._pipeline._initialize_crossattn_cache(batch_size=1, dtype=self._dtype, device=self._device)
        self._active_session = session
        return session

    def step(self, session: _MatrixSession, action: MatrixGameAction) -> MatrixGameStep:
        self._require_started()
        if session.closed or session is not self._active_session:
            raise MatrixGameBackendError("Matrix-Game session is closed or does not belong to this backend")
        block_size = int(self.config.num_frame_per_block)
        if session.current_start_frame + block_size > session.max_latent_frames:
            raise MatrixGameBackendError(
                "Matrix-Game session reached max_latent_frames; create a new session or increase the limit"
            )
        torch = self._torch
        pipeline = self._pipeline
        condition, _ = self._condition_for_step(session, action, block_size)
        noise = torch.randn(
            1, 16, block_size, 44, 80, device=self._device, dtype=self._dtype
        )
        current_start = session.current_start_frame
        with torch.inference_mode():
            noisy_input = noise
            denoised_pred = None
            for index, current_timestep in enumerate(pipeline.denoising_step_list):
                timestep = torch.ones(
                    1,
                    block_size,
                    device=self._device,
                    dtype=torch.int64,
                ) * current_timestep.to(device=self._device)
                _, denoised_pred = pipeline.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=condition,
                    timestep=timestep,
                    kv_cache=pipeline.kv_cache1,
                    kv_cache_mouse=pipeline.kv_cache_mouse,
                    kv_cache_keyboard=pipeline.kv_cache_keyboard,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start * pipeline.frame_seq_length,
                )
                if index < len(pipeline.denoising_step_list) - 1:
                    next_timestep = pipeline.denoising_step_list[index + 1].to(device=self._device)
                    noisy_input = pipeline.scheduler.add_noise(
                        denoised_pred.reshape(-1, denoised_pred.shape[1], denoised_pred.shape[3], denoised_pred.shape[4]),
                        torch.randn_like(
                            denoised_pred.reshape(
                                -1, denoised_pred.shape[1], denoised_pred.shape[3], denoised_pred.shape[4]
                            )
                        ),
                        next_timestep * torch.ones(
                            1 * block_size, device=self._device, dtype=torch.long
                        ),
                    ).reshape(1, denoised_pred.shape[1], block_size, denoised_pred.shape[3], denoised_pred.shape[4])

            assert denoised_pred is not None
            context_timestep = torch.ones(
                1, block_size, device=self._device, dtype=torch.int64
            ) * int(pipeline.args.context_noise)
            pipeline.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=condition,
                timestep=context_timestep,
                kv_cache=pipeline.kv_cache1,
                kv_cache_mouse=pipeline.kv_cache_mouse,
                kv_cache_keyboard=pipeline.kv_cache_keyboard,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start * pipeline.frame_seq_length,
            )
            video, session.vae_cache = self._decoder(
                denoised_pred.transpose(1, 2).half(), *session.vae_cache
            )
        session.current_start_frame += block_size
        frames = ((video[0].float() + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
        return MatrixGameStep(frames=tuple(frames), index=session.current_start_frame, fps=self.capabilities.output_fps)

    def close_session(self, session: _MatrixSession) -> None:
        session.closed = True
        if self._active_session is session:
            self._active_session = None

    def _load_runtime(
        self,
        root: Path,
        pretrained_root: Path,
        config_path: Path,
        checkpoint_path: str | None,
    ) -> None:
        self._root = root
        sys.path.insert(0, str(root))
        try:
            torch = importlib.import_module("torch")
            omega_conf = importlib.import_module("omegaconf").OmegaConf
            wan_wrapper = importlib.import_module("utils.wan_wrapper")
            causal = importlib.import_module("pipeline.causal_inference")
            vae_block3 = importlib.import_module("demo_utils.vae_block3")
            constant = importlib.import_module("demo_utils.constant")
            wanx_vae = importlib.import_module("wan.vae.wanx_vae")
            safetensors = importlib.import_module("safetensors.torch")
        except Exception as exc:  # pragma: no cover - optional Matrix environment
            raise MatrixGameBackendError(
                "Matrix-Game dependencies are unavailable; install the official Matrix-Game-2 requirements "
                f"in the runtime used by TeleFuser: {exc}"
            ) from exc

        if not torch.cuda.is_available():
            raise MatrixGameBackendError("Matrix-Game 2.0 official backend requires CUDA")
        self._torch = torch
        self._device = torch.device(self.config.device)
        self._dtype = torch.bfloat16
        args = omega_conf.load(str(config_path))
        model_config = Path(str(args.model_kwargs.model_config))
        if not model_config.is_absolute():
            model_config = root / model_config
        generator = wan_wrapper.WanDiffusionWrapper(
            model_config=str(model_config),
            timestep_shift=float(args.model_kwargs.get("timestep_shift", 5.0)),
            is_causal=True,
        )
        decoder = vae_block3.VAEDecoderWrapper()
        vae_checkpoint = pretrained_root / "Wan2.1_VAE.pth"
        if not vae_checkpoint.is_file():
            raise MatrixGameBackendError(f"Matrix VAE checkpoint does not exist: {vae_checkpoint}")
        vae_state = torch.load(str(vae_checkpoint), map_location="cpu")
        decoder_state = {
            key: value for key, value in vae_state.items() if "decoder." in key or "conv2" in key
        }
        decoder.load_state_dict(decoder_state)
        decoder.to(self._device, torch.float16).eval().requires_grad_(False)
        if checkpoint_path:
            generator.load_state_dict(safetensors.load_file(str(Path(checkpoint_path)), device="cpu"))
        pipeline = causal.CausalInferenceStreamingPipeline(args, generator=generator, vae_decoder=decoder)
        self._pipeline = pipeline.to(device=self._device, dtype=self._dtype)
        self._decoder = decoder
        self._vae = wanx_vae.get_wanx_vae_wrapper(str(pretrained_root), torch.float16)
        self._vae.requires_grad_(False).eval().to(self._device, self._dtype)
        self._zero_vae_cache = constant.ZERO_VAE_CACHE

    def _release_runtime(self) -> None:
        self._pipeline = None
        self._vae = None
        self._decoder = None
        self._zero_vae_cache = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    @staticmethod
    def _load_image(config: dict[str, Any]) -> Any:
        from PIL import Image

        image = config.get("image")
        image_path = config.get("image_path") or os.getenv("MATRIX_GAME_IMAGE")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if image_path:
            return Image.open(image_path).convert("RGB")
        raise ValueError("Matrix-Game session requires image or image_path")

    def _image_tensor(self, image: Any) -> Any:
        torch = self._torch
        image = image.resize((640, 352))
        values = torch.from_numpy(__import__("numpy").asarray(image, dtype="float32"))
        values = values.permute(2, 0, 1).div(127.5).sub(1.0)
        return values[None, :, None].to(device=self._device, dtype=self._dtype)

    def _condition_for_step(
        self, session: _MatrixSession, action: MatrixGameAction, block_size: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        current = session.current_start_frame
        final_frame = 1 + 4 * (current + block_size - 1)
        last_frame_num = 1 + 4 * (block_size - 1) if current == 0 else 4 * block_size
        start = final_frame - last_frame_num
        keyboard = self._keyboard_vector(session.mode, action)
        session.conditional_dict["keyboard_cond"][:, start:final_frame] = keyboard
        if session.mode != "templerun":
            mouse = self._torch.tensor(
                [action.mouse_dy, action.mouse_dx], device=self._device, dtype=self._dtype
            )
            session.conditional_dict["mouse_cond"][:, start:final_frame] = mouse
        condition = {
            "cond_concat": session.conditional_dict["cond_concat"][:, :, current : current + block_size],
            "visual_context": session.conditional_dict["visual_context"],
            "keyboard_cond": session.conditional_dict["keyboard_cond"][:, :final_frame],
        }
        if session.mode != "templerun":
            condition["mouse_cond"] = session.conditional_dict["mouse_cond"][:, :final_frame]
        return condition, session.conditional_dict

    def _keyboard_vector(self, mode: str, action: MatrixGameAction) -> Any:
        if mode == "universal":
            keys = set(action.keyboard)
            return self._torch.tensor(
                [float("w" in keys), float("s" in keys), float("a" in keys), float("d" in keys)],
                device=self._device,
                dtype=self._dtype,
            )
        keys = set(action.keyboard)
        index = {
            "q": 0,
            "w": 1,
            "s": 2,
            "z": 3,
            "c": 4,
            "a": 5,
            "d": 6,
        }
        vector = self._torch.zeros(7, device=self._device, dtype=self._dtype)
        selected = next((key for key in keys if key in index), "q")
        vector[index[selected]] = 1
        return vector

    def _require_started(self) -> None:
        if not self._started or self._pipeline is None:
            raise MatrixGameBackendError("Matrix-Game official backend has not been started")
