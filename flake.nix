{
  description = "AITX Hackathon 2025 - Next.js and Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        
        # Python 3 (packages will be managed by uv)
        pythonEnv = pkgs.python3;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Node.js toolchain
            nodejs_20
            nodePackages.npm
            nodePackages.pnpm
            nodePackages.yarn
            ngrok
            
            # Python toolchain
            pythonEnv
            python3Packages.pip
            python3Packages.setuptools
            python3Packages.wheel
            
            # uv - Fast Python package installer and resolver
            # Note: uv manages Python packages, so we don't need to include them in nix
            uv
          ];

          shellHook = ''
            echo "Node.js version: $(node --version)"
            echo "npm version: $(npm --version)"
            echo "Python version: $(python3 --version)"
            echo "uv version: $(uv --version 2>/dev/null || echo 'uv not found')"
            
            # Configure uv to use system Python instead of downloading
            export UV_PYTHON_DOWNLOADS=never
            export PATH="$HOME/.local/bin:$PATH"
            
            # Set up uv if not already configured
            if ! [ -d .venv ]; then
              echo ""
              echo "To set up Python environment with uv:"
              echo "  uv venv"
              echo "  source .venv/bin/activate"
              echo "  uv pip install -r requirements.txt"
            fi
          '';
        };
      }
    );
}
