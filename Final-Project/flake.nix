{
    description = "catifier - classifier for cat breed (cosc522 final project)";

    inputs = {
        nixpkgs.url = github:NixOS/nixpkgs/22.05;
    };

    outputs = { self, nixpkgs }: rec {
        arch = "x86_64-linux";
        pkgs = import nixpkgs {
            system = "${arch}";
        };

        py = pkgs.python39;
        pyenv = py.withPackages (p: with p; [
            ipykernel
            notebook
            scikit-learn
            seaborn
            pandas
            numpy
            pypandoc
            librosa
            opencv3
            pyfiglet
            sounddevice
            pillow
            imbalanced-learn
        ]);

        # Make the shell.
        devShells.${arch} = rec {
            default = cat-dev;

            cat-dev = pkgs.mkShell rec {
                name = "cat-dev";

                packages = [
                    pyenv
                    pkgs.git
                    pkgs.gnumake
                ];

                inputsFrom = [
                    # NOP
                ];

                shellHook = ''
                    PYTHONPATH=${pyenv}/${pyenv.sitePackages}
                    export PS1='\n\[\033[1;36m\][${name}:\W]\$\[\033[0m\] '
                '';
            };
        };
    };
}
