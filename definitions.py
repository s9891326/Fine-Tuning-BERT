from pathlib import Path, PurePath

ROOT_DIR = PurePath(__file__).parent
LOGS_DIR = Path(ROOT_DIR / "logs")
DATASET = Path(ROOT_DIR / "datasets")
OLD_FOLDER = Path(DATASET / "old_sentiment")
NEW_FOLDER = Path(DATASET / "new_sentiment")
