import logging

from schema_knowledge import rebuild

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")


def main():
    count = rebuild()
    print(f"Indexed {count} document(s).")


if __name__ == "__main__":
    main()