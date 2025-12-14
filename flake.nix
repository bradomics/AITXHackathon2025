{
  description = "AITX Hackathon 2025 - Next.js development environment";

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
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            nodejs_20
            nodePackages.npm
            nodePackages.pnpm
            nodePackages.yarn
            ngrok
          ];

          shellHook = ''
            echo "Node.js version: $(node --version)"
            echo "npm version: $(npm --version)"
          '';
        };
      }
    );
}
