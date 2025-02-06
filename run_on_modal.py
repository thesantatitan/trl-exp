import modal
import sys
import os

def run_file_in_sandbox(python_file):
    if not os.path.exists(python_file):
        print(f"Error: File {python_file} does not exist")
        return

    if not os.path.exists("requirements.txt"):
        print("Warning: requirements.txt not found in current directory")
        return

    # Create app
    app = modal.App.lookup("sandbox-runner", create_if_missing=True)

    # Define image
    image = (
        modal.Image.from_registry('pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel')
        .apt_install(["git", "build-essential", "libcairo2", "libvips"])
        .run_commands('uv pip install huggingface datasets transformers cairosvg accelerate einops pyvips --compile-bytecode --system')
        .add_local_file(python_file, f"/{python_file}", copy=True)
    )
    envs = modal.Secret.from_dict({"HF_TOKEN": os.getenv('HF_TOKEN'),"TRANSFORMERS_VERBOSITY":"info"})

    # Enable output streaming to see build logs
    with modal.enable_output():
        print("Creating sandbox with PyTorch image...")
        # Create sandbox with the image
        sb = modal.Sandbox.create(
            image=image,
            app=app,
            timeout=60*60,
            gpu="a10g",
            secrets=[envs],
            workdir='/'
        )

        try:
            print(f"\nExecuting {python_file}...")
            # Run the Python file with output streaming
            from modal.stream_type import StreamType
            process = sb.exec(
                "python",
                python_file,
                stdout=StreamType.STDOUT,  # Stream directly to stdout
                stderr=StreamType.STDOUT   # Stream directly to stderr
            )

            # Wait for the process to complete
            process.wait()

        finally:
            # Clean up
            sb.terminate()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sandbox_runner.py <python_file>")
        sys.exit(1)

    run_file_in_sandbox(sys.argv[1])
