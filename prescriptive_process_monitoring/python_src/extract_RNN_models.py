from pathlib import Path
import shutil


def extract(source, destination):
    if not Path(source).exists() or not Path(source).is_dir():
        raise FileExistsError("Path does not exist as a directory: " + str(source))
    Path(destination).mkdir(parents=True, exist_ok=True)
    i = 0
    files = Path(source).rglob("**/0-results.edited.csv")
    for f in files:
        shutil.copy2(str(f), str(destination) + "/" + str(i) + "_results.csv")
        i += 1
        if i % 10 == 0:
            print("Completed: " + str(i) + "_results.csv")


if __name__ == '__main__':
    source = "models-traffic"
    extract("../" + source, "../" + source + "-extracted")
