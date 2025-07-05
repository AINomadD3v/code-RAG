{
  description = "CocoIndex dev shell with system dependencies and automated Python venv setup";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in {
        devShells.default = pkgs.mkShell {
          name = "cocoindex-dev-env";

          buildInputs = with pkgs; [
            python3
            python3Packages.pip
            python3Packages.setuptools
            python3Packages.wheel
            python3Packages.virtualenv

            stdenv.cc.cc.lib
            gcc-unwrapped.lib
            zlib

            rustc
            cargo
            docker
            docker-compose
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.gcc-unwrapped.lib
            pkgs.zlib
          ];

          # This environment variable can be used by the shellHook
          # to know which Python executable from Nix to use for creating the venv.
          PYTHON_NIX_EXEC = "${pkgs.python3}/bin/python3";

          shellHook = ''
            echo "✅ CocoIndex development shell activating..."
            echo "   Nix has provided Python, pip, virtualenv, and essential system libraries."
            echo "💡 LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"
            echo "💡 Python from Nix: $PYTHON_NIX_EXEC"
            echo ""

            # --- Automated venv setup and dependency installation ---
            VENV_DIR=".venv"

            if [ ! -d "$VENV_DIR" ]; then
                echo "📦 Virtual environment '$VENV_DIR' not found. Creating..."
                $PYTHON_NIX_EXEC -m venv "$VENV_DIR"
                echo "Virtual environment created."
            else
                echo "Found existing virtual environment '$VENV_DIR'."
            fi

            echo "🐍 Activating virtual environment..."
            source "$VENV_DIR/bin/activate"
            echo "   Active Python: $(which python)"
            echo "   Python version: $(python --version)"

            echo "📦 Upgrading pip in virtual environment..."
            pip install --upgrade pip --quiet # --quiet to reduce noise

            echo "📦 Installing Python dependencies into virtual environment..."
            # Using a requirements file is more robust, but direct install is fine for now.
            # Ensure quotes are handled correctly if your package list gets complex.
            pip install \
                "cocoindex[cocoinsight]==0.1.36" \
                "sentence-transformers" \
                "python-dotenv" \
                "sqlalchemy" \
                "psycopg2-binary" \
                "fastapi[all]" \
                --quiet # --quiet to reduce noise during routine activation

            echo "🔍 Verifying NumPy installation within virtual environment:"
            if python -c "import numpy; print(f'NumPy {numpy.__version__} loaded from: {numpy.__file__}')" &> /dev/null; then
                echo "   ✅ NumPy imported successfully."
            else
                echo "   ❌ ERROR: NumPy import failed within the virtual environment."
                echo "      This might indicate an issue with LD_LIBRARY_PATH or pip installation."
            fi
            # --- End Automated Setup ---

            echo ""
            echo "✨ Python environment configured and virtual environment '$VENV_DIR' is active."
            echo "---------------------------------------------------------------------"
            echo "🚀 Next Steps:"
            echo "   1. Ensure PostgreSQL is running (e.g., execute ./dev.sh in another terminal or ensure it's already up)."
            echo "      Your .env should point to it: COCOINDEX_DATABASE_URL=postgresql://cocoindex:cocoindex@localhost:5543/cocoindex"
            echo ""
            echo "   2. Manage CocoIndex (run these in this terminal):"
            echo "      python main.py cocoindex setup   # To create DB tables (if first time or schema changed)"
            echo "      python main.py cocoindex update  # To process/index your uiautomator2 code"
            echo "      python main.py cocoindex ls      # To list defined flows"
            echo ""
            echo "   3. Start the search API server for your LLM assistant:"
            echo "      python main.py serve-api"
            echo ""
            echo "   4. (Optional) Start CocoInsight UI server:"
            echo "      python main.py cocoindex server -ci"
            echo "---------------------------------------------------------------------"
          '';
        };
      }
    );
}
