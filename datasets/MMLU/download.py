import os
import requests
import tarfile
from KVCOMM.utils.log import logger


def download():

    this_file_path = os.path.split(__file__)[0]
    tar_path = os.path.join(this_file_path, "data.tar")
    if not os.path.exists(tar_path):
        url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
        logger.info("Downloading {}", url)
        r = requests.get(url, allow_redirects=True)
        with open(tar_path, 'wb') as f:
            f.write(r.content)
        logger.info("Saved to {}", tar_path)

    data_path = os.path.join(this_file_path, "data")
    if not os.path.exists(data_path):
        tar = tarfile.open(tar_path)
        tar.extractall(this_file_path)
        tar.close()
        logger.info("Saved to {}", data_path)


if __name__ == "__main__":
    download()
