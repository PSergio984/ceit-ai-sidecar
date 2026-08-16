#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="$HOME/.local/share/sonarqube-cli/bin"
BINARY_NAME="sonar"
TMP_DIR=""

cleanup() {
  [[ -n "$TMP_DIR" ]] && rm -rf "$TMP_DIR"
}
trap cleanup EXIT

BASE_URL="https://binaries.sonarsource.com/Distribution/sonarqube-cli"
version="1.5.0.4158"

detect_os() {
  local os
  os="$(uname -s)"
  case "$os" in
    Linux*)  echo "linux" ;;
    Darwin*) echo "macos" ;;
    *)
      echo "Unsupported operating system: $os" >&2
      exit 1
      ;;
  esac
}

detect_platform() {
  case "$(detect_os)" in
    linux)
      case "$(uname -m)" in
        aarch64 | arm64) echo "linux-arm64" ;;
        x86_64 | amd64) echo "linux-x86-64" ;;
        *)
          echo "Unsupported Linux architecture: $(uname -m)" >&2
          exit 1
          ;;
      esac
      ;;
    macos)
      case "$(uname -m)" in
        aarch64 | arm64) echo "macos-arm64" ;;
        x86_64 | amd64) echo "macos-x86-64" ;;
        *)
          echo "Unsupported macOS architecture: $(uname -m)" >&2
          exit 1
          ;;
      esac
      ;;
  esac
}

# Optional third argument "quiet": return 1 on failure instead of exiting (stderr suppressed).
download() {
  local url="$1"
  local dest="$2"
  local quiet="${3:-}"
  # curl is guaranteed on GitHub runners; HTTPS is enforced with --proto.
  if [[ -n "$quiet" ]]; then
    curl -fsSL --proto '=https' "$url" -o "$dest" 2>/dev/null
  else
    curl -fsSL --proto '=https' "$url" -o "$dest"
  fi
  return
}

# Fetches the CLI from binaries.sonarsource.com (sonar self-update runs this script from GitHub).
# Tries .bin first, then .exe for legacy CDN builds. Remove .exe fallback once .bin is released.
# SHA-256 values recorded at vendoring time for version 1.5.0.4158 (.bin artifacts).
# Artifacts without a recorded checksum (e.g. the .exe fallback) are refused.
ARTIFACT_SHA256_LINUX_X86_64="DBD4EE20257F73010AD7F8A2C2552373039EE3610AF252416E6F13F7FF915460"
ARTIFACT_SHA256_LINUX_ARM64="596018EC03F6282588E6BDE56904625A4ABD0C65C3E4DC3E05A9ECD28381C644"
ARTIFACT_SHA256_MACOS_ARM64="181DF66A25B6CAB6A4428F782FE279E01374088C1F3240FD946F7B91D35457E6"

verify_checksum() {
  local dest="$1"
  local platform="$2"
  local expected=""
  case "$platform" in
    linux-x86-64) expected="$ARTIFACT_SHA256_LINUX_X86_64" ;;
    linux-arm64) expected="$ARTIFACT_SHA256_LINUX_ARM64" ;;
    macos-arm64) expected="$ARTIFACT_SHA256_MACOS_ARM64" ;;
  esac
  if [[ -z "$expected" ]]; then
    echo "Error: no recorded checksum for platform $platform; refusing to install an unverified artifact." >&2
    exit 1
  fi
  printf '%s  %s\n' "$expected" "$dest" | sha256sum -c - >/dev/null
}

download_cli_artifact() {
  local version="$1"
  local platform="$2"
  local os="$3"
  local dest="$4"
  local base="sonarqube-cli-${version}-${platform}"
  local url_bin="$BASE_URL/$version/$os/${base}.bin"
  local url_exe="$BASE_URL/$version/$os/${base}.exe"

  if download "$url_bin" "$dest" quiet; then
    echo "  $url_bin"
    verify_checksum "$dest" "$platform"
    return 0
  fi
  if download "$url_exe" "$dest" quiet; then
    echo "  $url_exe"
    verify_checksum "$dest" "$platform"
    return 0
  fi

  echo "Error: could not download sonarqube-cli (tried .bin and .exe):" >&2
  echo "  $url_bin" >&2
  echo "  $url_exe" >&2
  exit 1
}

# Detect the best shell profile file to update (inspired by nvm_detect_profile).
# Honors $PROFILE override, detects shell from $SHELL, respects $ZDOTDIR for zsh.
# Outputs exactly one file path, or nothing if no profile is found.
detect_profile() {
  if [[ "${PROFILE:-}" == "/dev/null" ]]; then
    return
  fi
  if [[ -n "${PROFILE:-}" && -f "$PROFILE" ]]; then
    echo "$PROFILE"
    return
  fi

  local detected=""
  case "${SHELL+${SHELL##*/}}" in
    bash)
      if [[ -f "$HOME/.bashrc" ]]; then
        detected="$HOME/.bashrc"
      elif [[ -f "$HOME/.bash_profile" ]]; then
        detected="$HOME/.bash_profile"
      fi
      ;;
    zsh)
      if [[ -f "${ZDOTDIR:-$HOME}/.zshrc" ]]; then
        detected="${ZDOTDIR:-$HOME}/.zshrc"
      elif [[ -f "${ZDOTDIR:-$HOME}/.zprofile" ]]; then
        detected="${ZDOTDIR:-$HOME}/.zprofile"
      fi
      ;;
  esac

  if [[ -z "$detected" ]]; then
    for f in ".profile" ".bashrc" ".bash_profile" ".zprofile" ".zshrc"; do
      if [[ -f "$HOME/$f" ]]; then
        detected="$HOME/$f"
        break
      fi
    done
  fi

  [[ -n "$detected" ]] && echo "$detected"
}

# Appends the sonarqube-cli PATH export to the best shell profile,
# skipping if it is already present. Uses detect_profile() to choose
# the target file and reports the outcome on stdout.
update_profile() {
  local path_line='export PATH="$HOME/.local/share/sonarqube-cli/bin:$PATH"'
  local detected_profile
  detected_profile="$(detect_profile || true)"

  if [[ -z "$detected_profile" ]]; then
    echo "No shell profile files found. Add the following line to your shell profile manually:"
    echo "  $path_line"
  elif grep -qF 'sonarqube-cli/bin' "$detected_profile" 2>/dev/null; then
    echo "Already present in $detected_profile, skipping."
  else
    printf '\n# Added by sonarqube-cli installer\n%s\n' "$path_line" >> "$detected_profile"
    echo "Updated PATH in: $detected_profile"
  fi
}

main() {
  local platform
  platform="$(detect_platform)"

  # Pinned for reproducible, checksum-verified installs (stable.version is
  # deliberately not fetched at runtime).
  local version="1.5.0.4158"
  echo "Installing version: $version"

  local os
  os="$(detect_os)"

  local artifact_basename="sonarqube-cli-${version}-${platform}"
  local dest="$INSTALL_DIR/$BINARY_NAME"
  TMP_DIR="$(mktemp -d -t 'sonarqube-cli-install.XXXXXX')"

  echo "Detected platform: $platform"
  echo "Downloading sonarqube-cli from:"

  mkdir -p "$INSTALL_DIR"

  local tmp_bin="$TMP_DIR/$artifact_basename"

  download_cli_artifact "$version" "$platform" "$os" "$tmp_bin"

  mv "$tmp_bin" "$dest"
  chmod +x "$dest"

  if [[ "$platform" == macos-* ]]; then
    xattr -d com.apple.quarantine "$dest" 2>/dev/null || true
  fi

  echo "Installed sonar to: $dest"

  update_profile

  echo ""
  echo "Installation complete!"
  echo ""
  echo "sonar has been installed to: $dest"
  echo ""
  echo "What happens next:"
  echo "  - Any NEW terminal window you open will have 'sonar' available automatically."
  echo "  - This current terminal window won't see it yet — you have two options:"
  echo ""
  echo "    Option 1: Open a new terminal window (recommended)"
  echo ""
  echo "    Option 2: Activate it in this window right now by running:"
  echo "      export PATH=\"$INSTALL_DIR:\$PATH\""
  echo "      (This only applies to this window — you won't need to run it again.)"
  echo ""
  echo "Once ready, run 'sonar --help' to get started."
}

main "$@"
