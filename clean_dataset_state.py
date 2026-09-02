import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def clean_database():
    print("=== Cleaning Database Tables ===")
    db_paths = [
        ROOT / "backend" / "instance" / "smartxsense.db",
        ROOT / "instance" / "smartxsense.db",
    ]
    for db_path in db_paths:
        if db_path.exists():
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                tables = [r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
                if "dataset_preprocessing" in tables:
                    cursor.execute("DELETE FROM dataset_preprocessing")
                if "datasets" in tables:
                    cursor.execute("DELETE FROM datasets")
                conn.commit()
                conn.close()
                print(f"  Cleaned {db_path}")
            except Exception as e:
                print(f"  Error cleaning {db_path}: {e}")

def clean_feature_dirs():
    print("=== Cleaning Stale Feature Directories ===")
    demo_dirs = {
        "features_1c0c92ff-264e-4398-b83c-e0e082b956b4",
        "features_23889e57-aa6d-4a50-930c-874f6a963a25",
    }
    
    for domain in ["retail", "electricity"]:
        feat_root = ROOT / "ml_engine" / "data" / "features" / domain
        if feat_root.exists():
            for child in feat_root.iterdir():
                if child.is_dir() and child.name not in demo_dirs:
                    print(f"  Removing stale feature dir: {child.relative_to(ROOT)}")
                    shutil.rmtree(child, ignore_errors=True)

def clean_tmp_folders():
    print("=== Cleaning Temporary Test Folders ===")
    for folder_name in ["test_e2e_tmp", "test_tmp", ".test-tmp"]:
        folder = ROOT / folder_name
        if folder.exists():
            print(f"  Removing tmp folder: {folder_name}")
            shutil.rmtree(folder, ignore_errors=True)

if __name__ == "__main__":
    clean_database()
    clean_feature_dirs()
    clean_tmp_folders()
    print("[SUCCESS] Dataset state cleanup complete!")

