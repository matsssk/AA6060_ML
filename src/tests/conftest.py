import os
from io import StringIO

import numpy as np
import pytest


@pytest.fixture()
def load_data():
    with open("raw_data/ph2,0.DTA", "r", encoding="ISO-8859-1") as f:
        data = f.read().replace(",", ".")
    data = StringIO(data)
    df = np.loadtxt(data, skiprows=51, usecols=(2, 3))
    yield df
