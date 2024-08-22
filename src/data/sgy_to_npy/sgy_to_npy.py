import argparse

import numpy as np
import segyio
import pathlib


def main(sgy_paths: list[pathlib.Path], out_dir: list[pathlib.Path],
         xline_s: int, slow:bool):
    #https://github.com/equinor/segyio-notebooks/blob/master/notebooks/basic/03_basic_segy_editing.ipynb
    print(f"[LOG] Crossline size used: {xline_s}")
    data = None
    for path_idx, sgy_path in enumerate(sgy_paths):
        with segyio.open(sgy_path, strict=False) as f:
            print(f"[LOG] Current file name: {sgy_path.name}")
            # Get basic attributes
            print(f"[LOG] Unstructured: {f.unstructured}")
            print(f"[LOG] Inline size: {f.ilines}")
            print(f"[LOG] Crossline size: {f.xlines}")
            print(f"[LOG] Num traces: {f.tracecount}")
            print(f"[LOG] Sample rate: {segyio.tools.dt(f) / 1000}")
            print(f"[LOG] Num samples per trace: {f.samples.size}")
            print(f"[LOG] Trace start:end: {min(f.samples)}:{max(f.samples)}")
            print(f"[LOG] Headers len: {len(f.header)}")
            print(f"[LOG] First header: {f.header[0]}")
            print("====================")
            inline_size = f.tracecount / xline_s
            if inline_size % 1 != 0:
                err_msg = "The xline size provided does not result in an integer sized inline!"
                err_msg += f" xline: {xline_s}, traces: {f.tracecount}, resulting inline: {inline_size}"
                raise ValueError(err_msg)

            inline_size = int(inline_size)
            trace_size = f.samples.size

            data_shape = (inline_size, xline_s, trace_size)
            if data is None or data.shape != data_shape:
                data = np.empty(data_shape)
            else:
                # Reuse already created data for speed
                pass

            if slow:
                for inline_idx, xline_start in enumerate(range(0, f.tracecount, xline_s)):
                    data[inline_idx, :, :] = f.trace.raw[xline_start:xline_start+xline_s]
            else:
                data = f.trace.raw[:].reshape(data_shape)
            
            target_path = out_dir[path_idx] / (sgy_path.stem + ".npy")
            print(f"[LOG] Saving data to {target_path}")
            np.save(target_path, data)


def config_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='This script transforms sgy files to npy files')

    parser.add_argument(
        '--files',
        nargs="+",
        dest='files',
        required=True,
        type=pathlib.Path,
        help="The paths to sgy files to be transformed." \
        " Example: --files path1 path2 path3 or --files path1, path2, path3")

    parser.add_argument(
        '--target_dir',
        required=False,
        default=None,
        type=pathlib.Path,
        help="The target dir path where to save the npy files." \
            "If not provided, use the dir of each file provided for --files."
    )

    parser.add_argument(
        '--xline',
        required=True,
        type=int,
        help="The crossline size used to evenly divide the number of traces.")

    parser.add_argument(
        '-slow',
        action='store_true',
        help=
        "Indicates that we shouldnt load all the seismic data into memory to reshape it."
    )

    return parser


if __name__ == "__main__":
    parser = config_arg_parser()
    args = parser.parse_args()

    if args.target_dir is None:
        target_dirs = [file.parents[0].resolve() for file in args.files]
    else:
        target_dirs = [
            args.target_dir.resolve() for _ in range(len(args.files))
        ]

    print(args.files)
    print(target_dirs)
    main(args.files, target_dirs, args.xline, args.slow)
