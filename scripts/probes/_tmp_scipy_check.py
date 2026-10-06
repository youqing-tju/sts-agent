import sys
try:
    import scipy
    with open("_tmp_scipy.txt", "w", encoding="utf-8") as f:
        f.write("SCIPY_OK " + scipy.__version__)
except Exception as e:
    with open("_tmp_scipy.txt", "w", encoding="utf-8") as f:
        f.write("NO_SCIPY " + repr(e))
