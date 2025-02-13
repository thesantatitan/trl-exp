import modal
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from modal.stream_type import StreamType
from queue import Queue
import time

def stream_reader(stream, queue, stream_name):
    """Read from a stream and put lines into a queue"""
    try:
        for line in stream:
            queue.put((stream_name, line.encode('utf-8')))
    finally:
        queue.put((stream_name, None))  # Signal EOF

def ensure_logs_directory():
    """Create logs directory if it doesn't exist"""
    if not os.path.exists("logs"):
        os.makedirs("logs")

def run_sandbox(index, python_file):
    """Run a single sandbox instance with its own log file"""
    log_file = f"logs/sandbox_{index}.log"

    # Create app
    app = modal.App.lookup(f"sandbox-runner-{index}", create_if_missing=True)

    # Define image
    image = (
        modal.Image.from_registry('pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel')
        .apt_install(["git", "build-essential", "libcairo2", "libvips"])
        .run_commands('uv pip install huggingface datasets transformers cairosvg accelerate einops pyvips --compile-bytecode --system')
        .add_local_file(python_file, f"/{python_file}", copy=True)
    )

    envs = modal.Secret.from_dict({
        "HF_TOKEN": os.getenv('HF_TOKEN'),
        "TRANSFORMERS_VERBOSITY": "info"
    })
    vols = modal.Volume.from_name("svg-dataset", create_if_missing=True)

    # Open log file for writing
    with open(log_file, 'w') as log:
        log.write(f"=== Starting sandbox {index} ===\n")
        log.flush()

        # Create sandbox with the image
        sb = modal.Sandbox.create(
            image=image,
            app=app,
            timeout=12*60*60,
            gpu="a10g",
            secrets=[envs],
            workdir='/',
            volumes={"/dataset": vols}
        )

        try:
            log.write(f"\nExecuting {python_file} with index {index}...\n")
            log.flush()

            # Run the Python file with PIPE output
            process = sb.exec(
                "python",
                python_file,
                "--number_of_batches", "32",
                "--index", str(index),
                stdout=StreamType.PIPE,
                stderr=StreamType.PIPE
            )

            # Create a queue for the output
            output_queue = Queue()

            # Start threads to read from stdout and stderr
            stdout_thread = threading.Thread(
                target=stream_reader,
                args=(process.stdout, output_queue, "stdout")
            )
            stderr_thread = threading.Thread(
                target=stream_reader,
                args=(process.stderr, output_queue, "stderr")
            )

            stdout_thread.start()
            stderr_thread.start()

            # Keep track of which streams are still active
            active_streams = {"stdout", "stderr"}

            # Process output from both streams as it becomes available
            while active_streams:
                try:
                    stream_name, line = output_queue.get()
                    if line is None:
                        active_streams.remove(stream_name)
                        continue

                    # Write the line to the log file
                    log.write(f"[{stream_name}] {line}")
                    log.flush()
                except Exception as e:
                    log.write(f"\nError processing output: {str(e)}\n")
                    log.flush()
                    break

            # Wait for reader threads to finish
            stdout_thread.join()
            stderr_thread.join()

            # Wait for the process to complete
            process.wait()

        except Exception as e:
            log.write(f"\nError in sandbox {index}: {str(e)}\n")
            log.flush()

        finally:
            log.write(f"\n=== Finishing sandbox {index} ===\n")
            log.flush()
            # Clean up
            sb.terminate()

def run_parallel_sandboxes(python_file, num_sandboxes=32):
    """Run multiple sandboxes in parallel"""
    if not os.path.exists(python_file):
        print(f"Error: File {python_file} does not exist")
        return

    if not os.path.exists("requirements.txt"):
        print("Warning: requirements.txt not found in current directory")
        return

    # Create logs directory
    ensure_logs_directory()

    print(f"Starting {num_sandboxes} parallel sandboxes...")

    # Use ThreadPoolExecutor to run sandboxes in parallel
    with ThreadPoolExecutor(max_workers=num_sandboxes) as executor:
        # Submit all sandbox tasks
        futures = [
            executor.submit(run_sandbox, i, python_file)
            for i in range(num_sandboxes)
        ]

        # Wait for all tasks to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error in sandbox task: {str(e)}")

    print(f"\nAll {num_sandboxes} sandboxes completed. Check logs directory for output.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sandbox_runner.py <python_file>")
        sys.exit(1)

    run_parallel_sandboxes(sys.argv[1])
