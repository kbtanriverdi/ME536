import subprocess
import atexit
import signal
import os

# Create a list to keep track of subprocesses
processes = []

def terminate_processes():
    for process in processes:
        try:
            # Send SIGTERM to the subprocess to terminate it
            process.terminate()
        except Exception as e:
            print(f"Error terminating process: {e}")

# Register terminate_processes to be called at exit
atexit.register(terminate_processes)

# Start the first subprocess (image.py)
process_image = subprocess.Popen(['python', 'imageopen.py'])
processes.append(process_image)

# Start the second subprocess (main.py itself or another script)
process_main = subprocess.Popen(['python', 'classifier.py'])  # Replace with the correct script if needed
processes.append(process_main)

# Continue running the main program
print("Main program is running. Press Ctrl+C to exit.")
try:
    # You can put your main program logic here
    process_main.wait()  # Wait for subprocesses to complete
except KeyboardInterrupt:
    print("Exiting main program...")
