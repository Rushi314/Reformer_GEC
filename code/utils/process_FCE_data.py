import difflib
import sys
import argparse

def get_binary(ground_truth, actual_string):
    diffs = difflib.SequenceMatcher(None, actual_string, ground_truth)
    binary_out = [0] * len(actual_string)
    for tag, i1, i2, j1, j2 in diffs.get_opcodes():
        if tag == "delete":
            binary_out[i1:i2] = [1]*(i2-i1)
    return binary_out
    

def main():
    with open(args.output_file, "w+", encoding='utf-8') as out_file:
        with open(args.truth, encoding='utf-8') as gt , \
                open(args.actual, encoding='utf-8') as og:
            for line_gt, line_og in zip(gt, og):
                out_file.write("".join(map(str, get_binary(line_gt, line_og)))+"\n")
            

if __name__ == '__main__':
    # Read files
    # Get binary out
    # Append to output file
    parser = argparse.ArgumentParser(description=(
        "Convert FCE data into required input format for Reformer"))
    parser.add_argument('output_file',
                        help='Output file path')
    parser.add_argument('--truth',
                        help='Path to the ground truth file')
    parser.add_argument('--actual',
                        help='Path to the actual file to be compared')
    args = parser.parse_args()
    main()
