#
# Copyright (c) 2025 [MetaMachines, Charlie Durham]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

# Program to take a file and inline it to a c header file as a static const char*

import argparse

def escape_c_string(s):
    """Escape special characters for C string literals."""
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\r', '\\r').replace('\t', '\\t')

def main():
    parser = argparse.ArgumentParser(description="Convert a file's content to a static const char* in a C99 header file.")
    parser.add_argument('input_file', help="Path to the input file to convert.")
    parser.add_argument('output_file', help="Path to the output header file.")
    parser.add_argument('--varname', default='file_content', help="Name of the const char* variable (default: file_content).")
    parser.add_argument('--guard', default='FILE_CONTENT_H', help="Include guard macro name (default: FILE_CONTENT_H).")
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.splitlines()
    ends_with_newline = content.endswith('\n')

    with open(args.output_file, 'w', encoding='utf-8') as out:
        out.write(f"#ifndef {args.guard}\n")
        out.write(f"#define {args.guard}\n\n")
        out.write(f"static const char *{args.varname} =\n")
        
        for i, line in enumerate(lines):
            escaped = escape_c_string(line)
            suffix = '\\n' if i < len(lines) - 1 or (i == len(lines) - 1 and ends_with_newline) else ''
            out.write(f'    "{escaped}{suffix}"\n')
        
        out.write(";\n\n")
        out.write("#endif\n")

if __name__ == "__main__":
    main()