{
    description = "sound_class - a sound classifier model (cosc522 lab 2)";

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
        ]);

        # Make the shell.
        devShells.${arch} = rec {
            default = sc-dev;

            sc-dev = pkgs.mkShell rec {
                name = "sc-dev";

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
