import subprocess

# Set the parameter values you want to pass
input_file = '/path/to/your/input/file.txt'
num_iterations = 20

# Construct the command to run the script with specific parameter values
command = [
    '/bin/python3',
    '/home/justin/FedLearning/script_using_flags.py',
    f'--input_file={input_file}',
    f'--num_iterations={num_iterations}',
]

# Use subprocess.run() to run the command and capture the returncode
result = subprocess.run(command, capture_output=True, text=True)
return_code = result.returncode
output = result.stdout

# Check if the subprocess ran successfully
# if result.returncode == 0:
#     # Parse the float value from the standard output
#     float_value = float(result.stdout.strip())
#     print(f"Float value from subprocess: {float_value}")
# else:
#     print(f"Subprocess failed with return code {result.returncode}")
print(result.stderr)
