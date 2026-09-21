{
  description = "Dev environment for forecast-v3 (sales analysis, Jupyter notebooks)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = { nixpkgs, flake-parts, ... }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];

      perSystem = { system, ... }: {
        devShells.default =
          let
            pkgs = nixpkgs.legacyPackages.${system};
          in
          pkgs.mkShell {
            packages = with pkgs; [
              uv          # Python environment and dependency management (using uv.lock)
              git
              pandoc      # notebook conversion to LaTeX
              (texliveSmall.withPackages (tex: with tex; [
                adjustbox caption enumitem eurosym grffile jknapltx
                parskip pdfcol soul tcolorbox titling ulem upquote
                collection-fontsrecommended
              ]))          # XeLaTeX and packages for Jupyter's PDF template
            ];

            shellHook = ''
              echo "[forecast-v3] nix devShell: uv $(uv --version | cut -d' ' -f2)"
              echo "[forecast-v3] start: uv sync --frozen && uv run jupyter lab"
              echo "[forecast-v3] PDF: uv run jupyter nbconvert --to pdf notebooks/prognoza_sprzedazy.ipynb"
            '';
          };
      };
    };
}
