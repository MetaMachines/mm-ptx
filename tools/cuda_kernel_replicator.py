import argparse

def generate_kernels(n, template_file, output_file):
    # Read the template code
    with open(template_file, 'r') as f:
        template = f.read()
    
    # Generate all kernel code in a single string
    combined_code = ""
    for i in range(n):
        # Format the number as a 6-digit zero-padded string
        num_str = f"{i:06d}"
        # Replace '_000000' with the current number
        new_code = template.replace('_000000', num_str)
        combined_code += new_code + "\n"
    
    # Write all generated code to the specified output file
    with open(output_file, 'w') as f:
        f.write(combined_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replicate a CUDA kernel file with _000000 to N copies into a single file')
    parser.add_argument('-n', '--num-kernels', type=int, default=128,
                        help='Number of kernel copies to generate (default: 128)')
    parser.add_argument('-t', '--template', type=str, required=True,
                        help='Template file name (required)')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output file name (required)')
    
    args = parser.parse_args()
    generate_kernels(args.num_kernels, args.template, args.output)