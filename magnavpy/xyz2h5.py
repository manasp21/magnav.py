import argparse
import os
import pandas as pd
import h5py
import numpy as np

def _ensure_extension(filepath, extension):
    """Ensures the filepath has the given extension."""
    name, ext = os.path.splitext(filepath)
    if ext.lower() != extension.lower():
        return name + extension
    return filepath

def xyz2h5(xyz_filepath, h5_filepath,
           lines_filter=None, lines_type='exclude',
           tt_sort=True, downsample_160=False):
    """
    Convert XYZ data file to HDF5 format.

    Args:
        xyz_filepath (str): Path to the input XYZ file.
        h5_filepath (str): Path to the output HDF5 file.
        lines_filter (list, optional): List of tuples for line filtering.
            Each tuple: (line_number, start_time, end_time). Defaults to None.
        lines_type (str, optional): Type of line filtering: 'include' or 'exclude'.
            Defaults to 'exclude'.
        tt_sort (bool, optional): If True, sort data by 'tt' column. Defaults to True.
        downsample_160 (bool, optional): If True, attempt to downsample 160Hz-like
            data to 10Hz based on 'tt' column. Defaults to False.
    """
    xyz_filepath = _ensure_extension(xyz_filepath, ".xyz")
    h5_filepath = _ensure_extension(h5_filepath, ".h5")

    if not os.path.exists(xyz_filepath):
        print(f"Error: Input XYZ file not found: {xyz_filepath}")
        return

    print(f"Reading XYZ file: {xyz_filepath}")
    try:
        # Assuming space/tab delimited, first line is header, '*' is NaN
        df = pd.read_csv(xyz_filepath, delim_whitespace=True, na_values=['*'])
    except Exception as e:
        print(f"Error reading XYZ file: {e}")
        return

    if df.empty:
        print("XYZ file is empty or could not be parsed.")
        return

    # Convert columns to numeric where possible, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Initial data shape: {df.shape}")

    # Downsample 160Hz data to 10Hz (if applicable and 'tt' column exists)
    if downsample_160 and 'tt' in df.columns:
        print("Applying 160Hz to 10Hz downsampling...")
        # Ensure 'tt' is numeric and not all NaN
        if pd.api.types.is_numeric_dtype(df['tt']) and not df['tt'].isnull().all():
            # Approximate Julia logic: (par(split(line)[ind_tt])+1e-6) % 0.1 < 1e-3
            df = df[((df['tt'].astype(float) + 1e-6) % 0.1) < 1e-3].copy()
            print(f"Data shape after downsampling: {df.shape}")
        else:
            print("Warning: 'tt' column not suitable for downsampling or not found.")

    # Filter lines based on line number and time ranges
    if lines_filter and 'line' in df.columns and 'tt' in df.columns:
        print(f"Applying line filtering (type: {lines_type})...")
        if not (pd.api.types.is_numeric_dtype(df['line']) and pd.api.types.is_numeric_dtype(df['tt'])):
            print("Warning: 'line' or 'tt' columns are not numeric. Skipping line filtering.")
        else:
            combined_condition = pd.Series([False] * len(df), index=df.index)
            for line_num, start_time, end_time in lines_filter:
                condition = (df['line'] == line_num) & \
                            (df['tt'] >= start_time) & \
                            (df['tt'] <= end_time)
                combined_condition |= condition

            if lines_type == 'exclude':
                df = df[~combined_condition].copy()
            elif lines_type == 'include':
                df = df[combined_condition].copy()
            else:
                print(f"Warning: Unknown lines_type '{lines_type}'. Skipping line filtering.")
            print(f"Data shape after line filtering: {df.shape}")

    # Sort data by time ('tt' column)
    if tt_sort and 'tt' in df.columns:
        print("Sorting data by 'tt' column...")
        if pd.api.types.is_numeric_dtype(df['tt']) and not df['tt'].isnull().all():
            df = df.sort_values(by='tt').reset_index(drop=True)
        else:
            print("Warning: 'tt' column not suitable for sorting or not found.")


    if df.empty:
        print("No data remaining after processing. HDF5 file will not be created.")
        return

    print(f"Writing to HDF5 file: {h5_filepath}")
    try:
        with h5py.File(h5_filepath, 'w') as hf:
            # Write N (number of data rows)
            hf.create_dataset('N', data=len(df))

            # Write dt (time step) if 'tt' is available
            if 'tt' in df.columns and len(df) > 1 and pd.api.types.is_numeric_dtype(df['tt']) and not df['tt'].isnull().all():
                # Calculate dt similar to Julia: round(data[ind,ind_tt][2]-data[ind,ind_tt][1],digits=9)
                # Using median of differences for robustness
                dt = np.round(df['tt'].diff().median(), 9)
                if pd.isna(dt): # if only one point after diff or other issues
                    dt = 0.1 # default from Julia for N > 1 case with no second point
                hf.create_dataset('dt', data=dt)
            else:
                hf.create_dataset('dt', data=0.1) # Default dt if 'tt' not usable

            # Write other data fields (each column as a dataset)
            for col_name in df.columns:
                if col_name not in ['N', 'dt']: # Avoid overwriting N, dt if they were column names
                    # Ensure data is in a C-contiguous array and handle NaNs appropriately for h5py
                    col_data = df[col_name].values
                    if pd.api.types.is_numeric_dtype(col_data):
                         # h5py handles np.nan for float types. For int, NaNs might be an issue if not handled by pandas.
                        pass # Data is already numeric
                    else: # For object dtypes that might contain strings after failed numeric conversion
                        # Try to convert to string if not numeric, h5py handles string arrays
                        col_data = df[col_name].astype(str).values
                    hf.create_dataset(col_name, data=col_data)
        print("HDF5 file created successfully.")
    except Exception as e:
        print(f"Error writing HDF5 file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert XYZ data file to HDF5 format.")
    parser.add_argument("xyz_file", help="Path to the input XYZ file (e.g., data.xyz)")
    parser.add_argument("h5_file", help="Path for the output HDF5 file (e.g., data.h5)")
    parser.add_argument("--lines", type=str, default=None,
                        help="Line filtering criteria: 'line,start_t,end_t;line,start_t,end_t;...'. "
                             "Example: '1001.0,0,100;1002.0,50,200'")
    parser.add_argument("--lines_type", choices=['include', 'exclude'], default='exclude',
                        help="Type of line filtering: 'include' or 'exclude'. Default: exclude.")
    parser.add_argument("--no_sort", action='store_true',
                        help="Disable sorting by 'tt' column.")
    parser.add_argument("--downsample_160", action='store_true',
                        help="Enable 160Hz to 10Hz downsampling based on 'tt' column.")

    args = parser.parse_args()

    lines_filter_parsed = None
    if args.lines:
        lines_filter_parsed = []
        try:
            segments = args.lines.split(';')
            for seg in segments:
                parts = seg.split(',')
                if len(parts) == 3:
                    lines_filter_parsed.append((float(parts[0]), float(parts[1]), float(parts[2])))
                else:
                    raise ValueError("Each line segment must have 3 parts: line,start_time,end_time")
        except ValueError as e:
            parser.error(f"Invalid format for --lines: {e}. Use 'line,start,end;...'")

    xyz2h5(args.xyz_file, args.h5_file,
           lines_filter=lines_filter_parsed,
           lines_type=args.lines_type,
           tt_sort=not args.no_sort,
           downsample_160=args.downsample_160)

if __name__ == '__main__':
    main()