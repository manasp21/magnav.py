import sys

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while importing {module_name}: {e}")
        return False

if __name__ == "__main__":
    modules_to_check = [
        "magnavpy",
        "magnavpy.map_utils",
        "magnavpy.compensation",
        "magnavpy.ekf",
        "magnavpy.create_xyz",
        "magnavpy.common_types"
    ]

    print("Starting import checks...")
    all_successful = True
    for module in modules_to_check:
        if not check_import(module):
            all_successful = False
    
    if all_successful:
        print("All specified modules imported successfully.")
    else:
        print("Some modules failed to import.")