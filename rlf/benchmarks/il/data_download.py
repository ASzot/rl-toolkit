import os
import sys

import gdown

DIR = "./data/expert_datasets"

DEMOS = {
    "ant": [("reg", "1ST9_V_ddV4mdbhNnidx3r7BNabHki33m")],
    "hand": [("reg", "1NsZ8FrTIyVvxEiAyTRDtHyzcKfCZNlZu")],
    "maze2d": [
        ("25", "1Il1SWb0nX8RT796izf-YkqvYyb3ls8yO"),
        ("50", "1xfrhsFQEY__pCYe-6xYmPSkPdyrw-reD"),
        ("75", "1A4F3eammJaLWiV2HxAh9lKiij4dDgj6h"),
        ("100", "1Eocidtv_BUwmXQlVgF17rkRerxOX-mvM"),
    ],
    "nav": [
        ("25", "15GFUYTdvliQHCmwmLPZcXgJhcZ_hDcSb"),
        ("50", "1vcEtO8oS0wWBx9a9O6QRRqI1ImwRlzie"),
        ("75", "1D-2db8f6kpTvFhX34vAyqvSOK4u8r9Dm"),
        ("100", "1d-fhU9GOC5aHiWJVk1u5lua_g4uMnvhE"),
    ],
    "pick": [
        ("reg", "1xrAw_ic0DOjfBSl6P6btP4oVGsXFmKNB"),
    ],
    "push": [
        ("reg", "1kV48YTLdYO3SYN8OQk6KNWa12aCjqJcB"),
    ],
}


def check_data(tasks):
    os.makedirs(DIR, exist_ok=True)

    for task in tasks:
        for postfix, id in DEMOS[task]:
            url = "https://drive.google.com/uc?id=" + id
            target_path = "%s/%s_%s.pt" % (DIR, task, postfix)
            if os.path.exists(target_path):
                print("%s is already downloaded." % target_path)
            else:
                print("Downloading demo (%s_%s) from %s" % (task, postfix, url))
                gdown.download(url, target_path)


if __name__ == "__main__":
    tasks = []
    if len(sys.argv) > 1:
        tasks = sys.argv[1:]
    else:
        tasks = ["ant", "hand", "maze2d", "nav", "pick", "push"]

    check_data(tasks)
