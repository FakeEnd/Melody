import os
from pathlib import Path

pdir: str = os.path.dirname(os.path.realpath(__file__))


if __name__ == '__main__':
    print(pdir)