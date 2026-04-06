{
  description = "Optional local wrapper flake for WhisperX-WebUI using the external gfx803 compatibility runtime";

  nixConfig = {
    extra-substituters = [ "https://gfx803-rocm.cachix.org" ];
    extra-trusted-public-keys = [ "gfx803-rocm.cachix.org-1:UTaIREqPZa9yjY7hiMBYG556OrGR6WEhWPjqX4Us3us=" ];
  };

  inputs = {
    gfx803-compat.url = "path:/home/c/Documents/code/__OTHER/gfx803_compat_graph/gfx803_flake_v1";
    nixpkgs.follows = "gfx803-compat/nixpkgs";
  };

  outputs = { self, nixpkgs, ... }:
  let
    system = "x86_64-linux";

    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        rocmSupport = true;
      };
    };

    compatRepoRoot = "/home/c/Documents/code/__OTHER/gfx803_compat_graph";
    webuiRepoRoot = builtins.toString ./.;

    gfx803EnvText = ''
      export HSA_OVERRIDE_GFX_VERSION=8.0.3
      export ROC_ENABLE_PRE_VEGA=1
      export PYTORCH_ROCM_ARCH=gfx803
      export ROCM_ARCH=gfx803
      export TORCH_BLAS_PREFER_HIPBLASLT=0
      export MIOPEN_DEBUG_CONV_WINOGRAD=0
      export MIOPEN_DEBUG_CONV_FFT=0
      export CUBLAS_WORKSPACE_CONFIG=:4096:8
    '';

    commonInputs = with pkgs; [
      bash
      coreutils
      curl
      ffmpeg
      findutils
      git
      gnugrep
      gawk
      jq
      libsndfile
      pkg-config
      portaudio
      python312
      rubberband
      which
      rocmPackages.clr
      rocmPackages.rocblas
      rocmPackages.miopen
      rocmPackages.rocminfo
      rocmPackages.rocm-smi
    ];

    launchWebUI = pkgs.writeShellApplication {
      name = "start-whisperx-webui-gfx803";
      runtimeInputs = commonInputs;
      text = ''
        set -euo pipefail

        export WEBUI_ROOT="${webuiRepoRoot}"
        export GFX803_COMPAT_ROOT="''${GFX803_COMPAT_ROOT:-${compatRepoRoot}}"
        export EXTRACTED_OUTDIR="''${EXTRACTED_OUTDIR:-$GFX803_COMPAT_ROOT}"
        export JOBLIB_MULTIPROCESSING="''${JOBLIB_MULTIPROCESSING:-0}"
        export HIP_LAUNCH_BLOCKING="''${HIP_LAUNCH_BLOCKING:-1}"
        export XDG_CACHE_HOME="''${XDG_CACHE_HOME:-$WEBUI_ROOT/.cache}"
        export TORCH_HOME="''${TORCH_HOME:-$WEBUI_ROOT/.cache/torch}"
        export HF_HOME="''${HF_HOME:-$WEBUI_ROOT/.cache/huggingface}"
        export HUGGINGFACE_HUB_CACHE="''${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
        export PIP_DISABLE_PIP_VERSION_CHECK=1
        mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$WEBUI_ROOT/models" "$WEBUI_ROOT/outputs"

        if [[ ! -x "$GFX803_COMPAT_ROOT/scripts/host-docker-python.sh" ]]; then
          echo "ERROR: compatibility wrapper missing: $GFX803_COMPAT_ROOT/scripts/host-docker-python.sh" >&2
          exit 1
        fi

        if [[ ! -d "$EXTRACTED_OUTDIR/lib-compat" || ! -d "$EXTRACTED_OUTDIR/docker-venv" ]]; then
          echo "ERROR: extracted runtime missing under $EXTRACTED_OUTDIR" >&2
          echo "Expected: lib-compat/ and docker-venv/" >&2
          exit 1
        fi

        cd "$WEBUI_ROOT"
        exec bash "$GFX803_COMPAT_ROOT/scripts/host-docker-python.sh" "$WEBUI_ROOT/app.py" "$@"
      '';
    };

    bootstrapSilero = pkgs.writeShellApplication {
      name = "bootstrap-whisperx-webui-silero-cache";
      runtimeInputs = with pkgs; [ bash coreutils git gnugrep ];
      text = ''
        set -euo pipefail

        export WEBUI_ROOT="${webuiRepoRoot}"
        export GFX803_COMPAT_ROOT="''${GFX803_COMPAT_ROOT:-${compatRepoRoot}}"
        export TORCH_HOME="''${TORCH_HOME:-$WEBUI_ROOT/.cache/torch}"
        mkdir -p "$TORCH_HOME"

        exec bash "$GFX803_COMPAT_ROOT/scripts/bootstrap-silero-vad-cache.sh"
      '';
    };

    verifyRuntime = pkgs.writeShellApplication {
      name = "verify-whisperx-webui-gfx803-runtime";
      runtimeInputs = commonInputs;
      text = ''
        set -euo pipefail

        export GFX803_COMPAT_ROOT="''${GFX803_COMPAT_ROOT:-${compatRepoRoot}}"
        export EXTRACTED_OUTDIR="''${EXTRACTED_OUTDIR:-$GFX803_COMPAT_ROOT}"

        echo "== compatibility root =="
        echo "$GFX803_COMPAT_ROOT"
        echo
        echo "== extracted runtime =="
        ls -ld "$EXTRACTED_OUTDIR/lib-compat" "$EXTRACTED_OUTDIR/docker-venv"
        echo
        echo "== GPU visibility =="
        rocminfo | sed -n '1,40p' || true
        echo
        echo "== quick torch probe =="
        bash "$GFX803_COMPAT_ROOT/scripts/host-docker-python.sh" -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available(), "count", torch.cuda.device_count())'
      '';
    };
  in {
    packages.${system} = {
      webui = launchWebUI;
      bootstrap-silero-cache = bootstrapSilero;
      verify-runtime = verifyRuntime;
    };

    apps.${system} = {
      default = {
        type = "app";
        program = "${launchWebUI}/bin/start-whisperx-webui-gfx803";
      };
      webui = {
        type = "app";
        program = "${launchWebUI}/bin/start-whisperx-webui-gfx803";
      };
      bootstrap-silero-cache = {
        type = "app";
        program = "${bootstrapSilero}/bin/bootstrap-whisperx-webui-silero-cache";
      };
      verify-runtime = {
        type = "app";
        program = "${verifyRuntime}/bin/verify-whisperx-webui-gfx803-runtime";
      };
    };

    devShells.${system}.default = pkgs.mkShell {
      buildInputs = commonInputs ++ [ launchWebUI bootstrapSilero verifyRuntime ];
      shellHook = ''
        ${gfx803EnvText}
        export WEBUI_ROOT="${webuiRepoRoot}"
        export GFX803_COMPAT_ROOT="''${GFX803_COMPAT_ROOT:-${compatRepoRoot}}"
        export EXTRACTED_OUTDIR="''${EXTRACTED_OUTDIR:-$GFX803_COMPAT_ROOT}"
        export JOBLIB_MULTIPROCESSING="''${JOBLIB_MULTIPROCESSING:-0}"
        export HIP_LAUNCH_BLOCKING="''${HIP_LAUNCH_BLOCKING:-1}"
        export XDG_CACHE_HOME="''${XDG_CACHE_HOME:-$WEBUI_ROOT/.cache}"
        export TORCH_HOME="''${TORCH_HOME:-$WEBUI_ROOT/.cache/torch}"
        export HF_HOME="''${HF_HOME:-$WEBUI_ROOT/.cache/huggingface}"
        export HUGGINGFACE_HUB_CACHE="''${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
        mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$WEBUI_ROOT/models" "$WEBUI_ROOT/outputs"

        echo "WhisperX-WebUI gfx803 shell"
        echo "Compatibility root: $GFX803_COMPAT_ROOT"
        echo "Extracted runtime:  $EXTRACTED_OUTDIR"
        echo "Launch WebUI with:  start-whisperx-webui-gfx803 --server_name 0.0.0.0"
        echo "Warm Silero cache:  bootstrap-whisperx-webui-silero-cache"
        echo "Verify runtime with: verify-whisperx-webui-gfx803-runtime"
      '';
    };
  };
}
