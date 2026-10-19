"""CSV readback independent of the receiver write path."""
import csv


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.reader(stream))
