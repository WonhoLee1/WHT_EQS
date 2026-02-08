import os

files = ['part1.py', 'part2.py', 'part3.py', 'part4.py']
output = 'main_verification.py'

print("Building main_verification.py...")
try:
    with open(output, 'w', encoding='utf-8') as outfile:
        for fname in files:
            if os.path.exists(fname):
                with open(fname, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    outfile.write('\n') # Ensure clean separation
                print(f"  Added {fname}")
            else:
                print(f"Error: {fname} not found!")
                exit(1)
    print(f"Successfully built {output}")
except Exception as e:
    print(f"Build failed: {e}")
    exit(1)
