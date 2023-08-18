import subprocess
import argparse

def call_c_program(file1, file2, number):
    command = ["./quantizer", "-f", file1, "-g", file2, "-n", str(number)]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Error:", result.stderr)
    except FileNotFoundError:
        print("Error: The C program executable 'quantizer' was not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python wrapper for C program")
    parser.add_argument("-f", "--file1", required=True, help="Path to input model file")
    parser.add_argument("-g", "--file2", required=True, help="Path to output model file")
    parser.add_argument("-n", "--number", type=int, required=True, help="Quantization type")

    args = parser.parse_args()

    call_c_program(args.file1, args.file2, args.number)
