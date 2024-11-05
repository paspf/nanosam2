# Autohor: paspf
#
# Usage:
# Download two chunks into directory `data` and extract them:
# python download_sa_dataset.py links.txt -d data -n 2 -e


import argparse
import os
import requests
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser(description='Download and extract tar archives listed in a text file.')
parser.add_argument('filename', help='the name of the file containing the links to the tar archives')
parser.add_argument('-d', '--destination', help='destination directory for extracted files')
parser.add_argument('-e', '--extract', action='store_true', help='extract the tar archives (disable if files are non-tar)')
parser.add_argument('-r', '--remove', action='store_true', help='delete the tar archives after extraction')
parser.add_argument('-n', '--num-lines', type=int, help='number of lines to process')
args = parser.parse_args()

dest_dir = Path("")
if args.destination is not None:
    dest_dir = Path(args.destination)
    dest_dir.mkdir(parents=True, exist_ok=True)

with open(args.filename, 'r') as f:
    lines_processed = 0
    for line in f:
        if line.startswith("file_name"):
            continue
        if args.num_lines is not None and lines_processed >= args.num_lines:
            break

        name, link = line.strip().split('\t')
        name = dest_dir/Path(name)
        subdir = Path(name.stem)

        if (dest_dir/subdir).exists() and (dest_dir/subdir).is_dir():
            print(f"Skipping {name}, directory already exists.")
            lines_processed += 1
            continue

        print(f'Downloading {name}...', end='')
        try:
            response = requests.get(link, stream=True)
            response.raise_for_status()
            total_length = response.headers.get('content-length')
            downloaded_length = 0
            with open(name, 'wb') as archive:
                for chunk in response.iter_content(chunk_size=2048):
                    if chunk:
                        downloaded_length += len(chunk)
                        if total_length is not None:
                            print(f'\rDownloading {name}: {downloaded_length//1048576}/{int(total_length)//1048576} MB', end='', flush=True)
                        archive.write(chunk)
            print("...done")
            if args.extract:
                print(f'Extracting {name}...')
                subdir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(name, 'r') as tar:
                    tar.extractall(dest_dir/subdir)
            if args.remove:
                os.remove(name)
        except Exception as e:
            print(f'--> Failed to download {name}: {str(e)}')
        lines_processed += 1
    
