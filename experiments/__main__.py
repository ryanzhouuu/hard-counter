import importlib.util


def main() -> None:
    if importlib.util.find_spec("clash_sos") is None:
        raise RuntimeError("install the backend before running experiments")
    print("Experiment workspace imports successfully")


if __name__ == "__main__":
    main()
